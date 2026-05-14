import cv2
import numpy as np
import json
import sys
import os
import math

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT_DIR     = os.path.dirname(FEATURES_DIR)
OUTPUT_DIR   = os.path.join(FEATURES_DIR, "outputs")

SAMPLES = {
    "N001": {
        "ocr_json": os.path.join(FEATURES_DIR, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json"),
        "views": {
            "front": os.path.join(ROOT_DIR, "edges_N001_front.png"),
            "rear":  os.path.join(ROOT_DIR, "edges_N001_rear.png"),
        }
    },
    "BEV820": {
        "ocr_json": os.path.join(FEATURES_DIR, "samples", "sample_02_BEV820", "blueprint_page_parsed.json"),
        "views": {
            "front": os.path.join(ROOT_DIR, "edges_BEV820_front.png"),
            "rear":  os.path.join(ROOT_DIR, "edges_BEV820_rear.png"),
        }
    }
}

LEGACY_EDGE = os.path.join(ROOT_DIR, "debug_final_edges.png")

MATCH_TOL_FRAC = 0.15   # 15% of OCR diameter
MATCH_TOL_MIN  = 0.50   # floor in mm

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# STEP 1: Preprocess edge image — close gaps so holes become solid blobs
# ---------------------------------------------------------------------------

def _prepare(edge_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(edge_gray, 20, 255, cv2.THRESH_BINARY)
    # Use smaller kernel (5x5) to avoid over-expanding small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)


# ---------------------------------------------------------------------------
# STEP 2: Extract contour features — no filtering here, just measure everything
# ---------------------------------------------------------------------------

def extract_shape_features(edge_image_path: str):
    img = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {edge_image_path}")

    filled = _prepare(img)
    contours, hierarchy = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    seen = {}   # grid_key -> area, for dedup

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 25:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Dedup: same centroid grid cell + similar area → skip
        gk = (round(cx / 8) * 8, round(cy / 8) * 8)
        if gk in seen and abs(seen[gk] - area) / max(seen[gk], 1) < 0.25:
            continue
        seen[gk] = area

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        hull        = cv2.convexHull(cnt)
        hull_area   = cv2.contourArea(hull)
        solidity    = float(area) / hull_area if hull_area > 0 else 0
        (_, _), enc_r = cv2.minEnclosingCircle(cnt)
        eq_d        = math.sqrt(4 * area / math.pi)

        # For large contours use ellipse fitting — more stable than area-based diameter
        if len(cnt) >= 5 and eq_d > 20:
            try:
                axes = cv2.fitEllipse(cnt)[1]
                eq_d = float(np.mean(axes))
            except Exception:
                pass

        aspect = float(w) / h if h != 0 else 0
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        verts  = len(approx)
        angle  = cv2.minAreaRect(cnt)[-1]

        if   circularity > 0.85 and solidity > 0.85: shape = "circle"
        elif circularity > 0.65 and solidity > 0.75: shape = "ellipse"
        elif verts == 3:                              shape = "triangle"
        elif verts == 4:                              shape = "rectangle"
        elif aspect > 3.5:                            shape = "slot"
        else:                                         shape = "polygon"

        has_parent = bool(hierarchy is not None and i < len(hierarchy[0]) and hierarchy[0][i][3] >= 0)

        features.append({
            "id":                  i,
            "shape_type":          shape,
            "centroid":            {"x": cx, "y": cy},
            "area":                round(float(area), 2),
            "perimeter":           round(float(perimeter), 2),
            "bounding_box":        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "aspect_ratio":        round(aspect, 4),
            "circularity":         round(circularity, 4),
            "solidity":            round(solidity, 4),
            "equivalent_diameter": round(eq_d, 4),
            "enc_radius":          round(float(enc_r), 4),
            "rotation_angle":      round(float(angle), 2),
            "vertices":            int(verts),
            "has_parent":          has_parent,
        })

    return features, img


# ---------------------------------------------------------------------------
# STEP 3: RANSAC scale estimation
# Generate every possible (contour_diameter / ocr_diameter) scale hypothesis,
# find the cluster with the most votes within a 10% band, return its median.
# ---------------------------------------------------------------------------

def estimate_scale(features: list, ocr_diameters: list) -> float:
    candidates = sorted(features, key=lambda f: f["area"], reverse=True)
    if not candidates or not ocr_diameters:
        return None

    all_scales = []
    for c in candidates[:50]:
        px_d = c["equivalent_diameter"]
        for ocr_d in ocr_diameters:
            if ocr_d < 1.0:
                continue
            s = px_d / ocr_d
            if 0.2 < s < 10000:
                all_scales.append(s)

    if not all_scales:
        return None

    best_scale   = None
    best_inliers = 0
    best_score   = 999.0

    for s_cand in all_scales:
        inliers = [s for s in all_scales if abs(s - s_cand) / s_cand < 0.10]
        n = len(inliers)
        s_med = float(np.median(inliers))
        # Score: mean relative error — weight larger OCR features more heavily
        # because they are more reliably detected and anchor the scale better
        errors = []
        for c in candidates[:50]:
            mm = c["equivalent_diameter"] / s_med
            closest = min(ocr_diameters, key=lambda d: abs(d - mm))
            rel_err = abs(closest - mm) / max(closest, 1.0)
            # Weight by sqrt(ocr_diameter) — larger features get more weight
            weight = math.sqrt(closest)
            errors.append(rel_err * weight)
        score = sum(errors) / len(errors)

        if n > best_inliers or (n == best_inliers and score < best_score):
            best_inliers = n
            best_scale   = s_med
            best_score   = score

    return best_scale


# ---------------------------------------------------------------------------
# STEP 4: Match contours to OCR features
# Only filter on circularity >= 0.45 — keep it permissive, let tolerance do the work
# ---------------------------------------------------------------------------

def _tol(ocr_d: float) -> float:
    # Small holes get wider relative tolerance because morphological ops expand them
    if ocr_d <= 3.0:
        return max(1.2, ocr_d * 0.60)
    if ocr_d <= 6.0:
        return max(0.8, ocr_d * 0.25)
    return max(MATCH_TOL_MIN, ocr_d * MATCH_TOL_FRAC)


def compare_with_ocr(features: list, ocr_json_path: str, scale: float):
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr = json.load(f)

    # Build ocr_holes dict — one entry per unique diameter.
    # hole_patterns take priority over plain holes for the same diameter
    # so we don't double-count (e.g. "12x Ø5.2" and a plain "Ø5.2" are the same feature).
    ocr_holes = {}
    for h in ocr.get("holes", []):
        d = h["diameter"]
        if d not in ocr_holes:
            ocr_holes[d] = {"source": "hole", "through": h.get("through"), "depth": h.get("depth")}
    for hp in ocr.get("hole_patterns", []):
        d = hp["diameter"]
        # Always overwrite with the pattern entry — it carries count info and is more specific
        ocr_holes[d] = {"source": f"hole_pattern ({hp['count']}x)", "through": hp.get("through"), "depth": hp.get("depth")}
    for p in ocr.get("pcd_features", []):
        d = p["diameter"]
        if d not in ocr_holes:
            ocr_holes[d] = {"source": "pcd", "through": None, "depth": None}

    ocr_diameters = sorted(ocr_holes.keys())
    min_ocr_d   = min(ocr_diameters) if ocr_diameters else 1.0
    min_px_diam = (min_ocr_d / 3.0) * (scale if scale else 1.0)

    # --- Pass 1: collect all (contour, ocr_diameter, error) candidates ---
    candidates = []
    for f in features:
        if f["circularity"] < 0.45:
            continue
        if scale and f["equivalent_diameter"] < min_px_diam:
            continue
        px_d = f["equivalent_diameter"]
        mm_d = round(px_d / scale, 3) if scale else None
        if mm_d is None:
            continue
        for ocr_d in ocr_diameters:
            err = abs(ocr_d - mm_d)
            if err <= _tol(ocr_d):
                candidates.append((err, f, ocr_d))

    # --- Pass 2: greedy one-to-one assignment — best error wins ---
    # Sort by error ascending so smallest errors get priority
    candidates.sort(key=lambda x: x[0])
    claimed_ocr      = set()   # each OCR diameter claimed at most once
    claimed_contour  = set()   # each contour claimed at most once
    assignments      = {}      # contour_id -> (ocr_d, error, info)

    for err, f, ocr_d in candidates:
        if f["id"] in claimed_contour:
            continue
        if ocr_d in claimed_ocr:
            continue
        claimed_contour.add(f["id"])
        claimed_ocr.add(ocr_d)
        assignments[f["id"]] = (ocr_d, round(err, 4), ocr_holes[ocr_d])

    # --- Build results list ---
    results = []
    for f in features:
        if f["circularity"] < 0.45:
            continue
        if scale and f["equivalent_diameter"] < min_px_diam:
            continue
        px_d = f["equivalent_diameter"]
        mm_d = round(px_d / scale, 3) if scale else None

        if f["id"] in assignments:
            ocr_d, error_mm, info = assignments[f["id"]]
            matched_dia, matched_info = ocr_d, info
        else:
            matched_dia = matched_info = error_mm = None

        results.append({
            "contour_id":       f["id"],
            "shape_type":       f["shape_type"],
            "circularity":      f["circularity"],
            "solidity":         f["solidity"],
            "eq_diameter_px":   round(px_d, 2),
            "eq_diameter_mm":   mm_d,
            "matched_ocr_dia":  matched_dia,
            "ocr_feature_info": matched_info,
            "error_mm":         error_mm,
            "status":           "MATCH" if matched_dia is not None else "UNMATCHED",
            "centroid":         f["centroid"],
        })

    matched_dias = {r["matched_ocr_dia"] for r in results if r["matched_ocr_dia"]}
    missing_ocr  = [
        {"diameter": d, "info": info}
        for d, info in ocr_holes.items()
        if d not in matched_dias
    ]

    return results, ocr, missing_ocr


# ---------------------------------------------------------------------------
# STEP 5: Annotated output image
# ---------------------------------------------------------------------------

def draw_annotations(img_gray, features, results, missing_ocr, scale):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    rmap = {r["contour_id"]: r for r in results}
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    C_MATCH, C_FP, C_MISS = (0, 200, 0), (0, 120, 255), (0, 0, 220)

    for f in features:
        r = rmap.get(f["id"])
        if r is None:
            continue
        cx, cy = f["centroid"]["x"], f["centroid"]["y"]
        rad = max(4, int(f["enc_radius"]))
        if r["status"] == "MATCH":
            cv2.circle(vis, (cx, cy), rad, C_MATCH, 2)
            cv2.putText(vis, f"O{r['matched_ocr_dia']}", (cx-16, cy-rad-5), FONT, 0.38, C_MATCH, 1, cv2.LINE_AA)
        else:
            cv2.circle(vis, (cx, cy), rad, C_FP, 1)
            mm = r["eq_diameter_mm"]
            cv2.putText(vis, f"~{mm:.1f}" if mm else "?", (cx-14, cy-rad-5), FONT, 0.35, C_FP, 1, cv2.LINE_AA)

    h_img, w_img = vis.shape[:2]
    xc, y0 = max(w_img - 180, 10), 28
    cv2.putText(vis, "NOT DETECTED:", (xc, y0), FONT, 0.42, C_MISS, 1, cv2.LINE_AA)
    for idx, m in enumerate(sorted(missing_ocr, key=lambda x: x["diameter"])):
        cv2.putText(vis, f"  O{m['diameter']}  {m['info'].get('source','')}",
                    (xc, y0 + 20 + idx * 17), FONT, 0.35, C_MISS, 1, cv2.LINE_AA)

    lx, ly = 8, h_img - 72
    cv2.rectangle(vis, (lx-4, ly-14), (lx+230, ly+60), (25,25,25), -1)
    cv2.putText(vis, "GREEN  = Matched to OCR hole",   (lx, ly),    FONT, 0.37, C_MATCH, 1, cv2.LINE_AA)
    cv2.putText(vis, "ORANGE = Detected, not in OCR",  (lx, ly+18), FONT, 0.37, C_FP,    1, cv2.LINE_AA)
    cv2.putText(vis, "RED    = OCR hole not detected", (lx, ly+36), FONT, 0.37, C_MISS,  1, cv2.LINE_AA)
    cv2.putText(vis, f"Scale: {scale:.3f} px/mm" if scale else "Scale: unknown",
                (lx, ly+54), FONT, 0.37, (180,180,180), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# STEP 6: Report — one line per sample
# ---------------------------------------------------------------------------

def print_report(label, results, missing_ocr, ocr):
    matched      = [r for r in results if r["status"] == "MATCH"]
    # Deduplicate: a diameter in both holes and hole_patterns is one unique feature
    all_ocr_dias = set(
        [h["diameter"]  for h in ocr.get("holes", [])] +
        [hp["diameter"] for hp in ocr.get("hole_patterns", [])]
    )
    total_ocr    = len(all_ocr_dias)
    matched_dias = sorted(set(r["matched_ocr_dia"] for r in matched if r["matched_ocr_dia"] and r["matched_ocr_dia"] in all_ocr_dias))
    coverage     = len(matched_dias)
    dias_str     = ", ".join(f"Ø{d}mm" for d in matched_dias)
    miss_dias    = sorted(all_ocr_dias - set(matched_dias))
    miss_str     = f", missing: {', '.join(f'Ø{d}mm' for d in miss_dias)}" if miss_dias else ""
    print(f"[{label}] {coverage}/{total_ocr} features detected — {dias_str}{miss_str}")


# ---------------------------------------------------------------------------
# Run one edge image
# ---------------------------------------------------------------------------

def run_single(edge_image_path: str, ocr_json_path: str, label: str) -> dict:
    safe_label = label.replace(" ", "_").replace("/", "_")

    features, img_gray = extract_shape_features(edge_image_path)

    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_preview = json.load(f)

    ocr_diameters = sorted(set(
        [h["diameter"]  for h in ocr_preview.get("holes", [])] +
        [hp["diameter"] for hp in ocr_preview.get("hole_patterns", [])] +
        [p["diameter"]  for p in ocr_preview.get("pcd_features", [])]
    ))

    scale = estimate_scale(features, ocr_diameters)
    results, ocr, missing_ocr = compare_with_ocr(features, ocr_json_path, scale)

    vis = draw_annotations(img_gray, features, results, missing_ocr, scale)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{safe_label}_annotated.png"), vis)

    with open(os.path.join(OUTPUT_DIR, f"{safe_label}_comparison.json"), "w") as f:
        json.dump({
            "label":          label,
            "edge_image":     edge_image_path,
            "ocr_json":       ocr_json_path,
            "scale_px_mm":    round(scale, 4) if scale else None,
            "total_features": len(features),
            "results":        results,
            "missing_ocr":    missing_ocr,
        }, f, indent=4)

    print_report(label, results, missing_ocr, ocr)

    matched_count = sum(1 for r in results if r["status"] == "MATCH")
    return {"label": label, "matched": matched_count, "total": len(results), "missing_ocr": len(missing_ocr)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if sys.platform == "win32":
        os.system("color")

    if len(sys.argv) >= 3:
        edge_path = sys.argv[1]
        ocr_path  = sys.argv[2]
        label     = sys.argv[3] if len(sys.argv) > 3 else "custom"
        if not os.path.exists(edge_path):
            print(f"{RED}[ERROR] Edge image not found: {edge_path}{RESET}"); sys.exit(1)
        if not os.path.exists(ocr_path):
            print(f"{RED}[ERROR] OCR JSON not found: {ocr_path}{RESET}"); sys.exit(1)
        run_single(edge_path, ocr_path, label)
        return

    summary, missing_files = [], []

    for sample_name, config in SAMPLES.items():
        ocr_json = config["ocr_json"]
        if not os.path.exists(ocr_json):
            continue
        for view, edge_path in config["views"].items():
            label = f"{sample_name}_{view}"
            if not os.path.exists(edge_path):
                missing_files.append((label, edge_path)); continue
            summary.append(run_single(edge_path, ocr_json, label))

    if not summary and os.path.exists(LEGACY_EDGE):
        summary.append(run_single(LEGACY_EDGE, SAMPLES["N001"]["ocr_json"], "N001_legacy"))

    if not summary:
        print(f"{YELLOW}No edge images processed.{RESET}")
    for lbl, path in missing_files:
        print(f"{YELLOW}[MISSING]{RESET} {lbl} — run: python quick_test.py cad.png {os.path.basename(path)}")


if __name__ == "__main__":
    main()

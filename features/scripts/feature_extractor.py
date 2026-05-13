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
        "ocr_json": os.path.join(
            FEATURES_DIR, "samples", "sample_01_N001",
            "1-02-1065-N001_parsed.json"
        ),
        "views": {
            "front": os.path.join(ROOT_DIR, "edges_N001_front.png"),
            "rear":  os.path.join(ROOT_DIR, "edges_N001_rear.png"),
        }
    },
    "BEV820": {
        "ocr_json": os.path.join(
            FEATURES_DIR, "samples", "sample_02_BEV820",
            "blueprint_page_parsed.json"
        ),
        "views": {
            "front": os.path.join(ROOT_DIR, "edges_BEV820_front.png"),
            "rear":  os.path.join(ROOT_DIR, "edges_BEV820_rear.png"),
        }
    }
}

LEGACY_EDGE = os.path.join(ROOT_DIR, "debug_final_edges.png")

MATCH_TOL_FRAC = 0.12
MATCH_TOL_MIN  = 0.30

SEP = "=" * 62

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def _fill_enclosed_regions(edge_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(edge_gray, 30, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    return closed


def extract_shape_features(edge_image_path: str):
    img = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load edge image: {edge_image_path}")

    filled = _fill_enclosed_regions(img)

    contours, hierarchy = cv2.findContours(
        filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    features   = []
    seen_cents = {}

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        grid_key = (round(cx / 6) * 6, round(cy / 6) * 6)
        if grid_key in seen_cents:
            prev_area = seen_cents[grid_key]
            if abs(prev_area - area) / max(prev_area, 1) < 0.20:
                continue
        seen_cents[grid_key] = area

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        hull        = cv2.convexHull(cnt)
        hull_area   = cv2.contourArea(hull)
        solidity    = float(area) / hull_area if hull_area > 0 else 0
        (_, _), enc_radius = cv2.minEnclosingCircle(cnt)
        eq_diameter = math.sqrt(4 * area / math.pi)
        extent      = float(area) / (w * h) if (w * h) > 0 else 0
        rect        = cv2.minAreaRect(cnt)
        angle       = rect[-1]
        approx      = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices    = len(approx)

        hu_raw = cv2.HuMoments(M).flatten()
        hu = [
            float(-np.sign(v) * np.log10(abs(v))) if v != 0 else 0.0
            for v in hu_raw
        ]

        if circularity > 0.85 and solidity > 0.85:
            shape_type = "circle"
        elif circularity > 0.65 and solidity > 0.75:
            shape_type = "ellipse"
        elif vertices == 3:
            shape_type = "triangle"
        elif vertices == 4:
            shape_type = "rectangle"
        elif aspect_ratio > 3.5:
            shape_type = "slot"
        else:
            shape_type = "polygon"

        features.append({
            "id":                  i,
            "shape_type":          shape_type,
            "centroid":            {"x": cx, "y": cy},
            "area":                round(float(area), 2),
            "perimeter":           round(float(perimeter), 2),
            "bounding_box":        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "aspect_ratio":        round(aspect_ratio, 4),
            "circularity":         round(circularity, 4),
            "solidity":            round(solidity, 4),
            "extent":              round(extent, 4),
            "equivalent_diameter": round(eq_diameter, 4),
            "enc_radius":          round(float(enc_radius), 4),
            "rotation_angle":      round(float(angle), 2),
            "vertices":            int(vertices),
            "hu_moments":          hu,
        })

    return features, img


def estimate_scale(features: list, ocr_diameters: list):
    candidates = sorted(
        [f for f in features if f["circularity"] > 0.60],
        key=lambda f: f["area"], reverse=True
    )
    if not candidates or not ocr_diameters:
        return None

    best_scale = None
    best_score = 999.0

    for c in candidates[:20]:
        px_d = c["equivalent_diameter"]
        for ocr_d in ocr_diameters:
            if ocr_d < 1.0:
                continue
            scale = px_d / ocr_d
            if not (0.5 < scale < 1000):
                continue
            errors = []
            for other in candidates[:30]:
                mm = other["equivalent_diameter"] / scale
                closest = min(ocr_diameters, key=lambda d: abs(d - mm))
                errors.append(abs(closest - mm))
            avg_err = sum(errors) / len(errors)
            if avg_err < best_score:
                best_score = avg_err
                best_scale = scale

    return best_scale


def _tol(ocr_d: float) -> float:
    return max(MATCH_TOL_MIN, ocr_d * MATCH_TOL_FRAC)


def compare_with_ocr(features: list, ocr_json_path: str, scale: float):
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr = json.load(f)

    ocr_holes = {}
    for h in ocr.get("holes", []):
        d = h["diameter"]
        ocr_holes[d] = {
            "source": "hole",
            "through": h.get("through"),
            "depth":   h.get("depth"),
        }
    for hp in ocr.get("hole_patterns", []):
        d = hp["diameter"]
        ocr_holes[d] = {
            "source": f"hole_pattern ({hp['count']}x)",
            "through": hp.get("through"),
            "depth":   hp.get("depth"),
        }

    ocr_diameters = sorted(ocr_holes.keys())

    min_ocr_d   = min(ocr_diameters) if ocr_diameters else 1.0
    min_px_diam = (min_ocr_d / 3.0) * (scale if scale else 1.0)

    results = []
    for f in features:
        if f["circularity"] < 0.45:
            continue
        if scale and f["equivalent_diameter"] < min_px_diam:
            continue

        px_d = f["equivalent_diameter"]
        mm_d = round(px_d / scale, 3) if scale else None

        matched_dia  = None
        matched_info = None
        error_mm     = None

        if mm_d is not None:
            best_err = None
            for ocr_d in ocr_diameters:
                err = abs(ocr_d - mm_d)
                if err <= _tol(ocr_d):
                    if best_err is None or err < best_err:
                        best_err     = err
                        matched_dia  = ocr_d
                        matched_info = ocr_holes[ocr_d]
            if best_err is not None:
                error_mm = round(best_err, 4)

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


def draw_annotations(img_gray, features, results, missing_ocr, scale):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    result_map = {r["contour_id"]: r for r in results}

    C_MATCH   = (0,  200,   0)
    C_FP      = (0,  120, 255)
    C_MISSING = (0,   0,  220)
    FONT      = cv2.FONT_HERSHEY_SIMPLEX

    for f in features:
        r = result_map.get(f["id"])
        if r is None:
            continue
        cx     = f["centroid"]["x"]
        cy     = f["centroid"]["y"]
        radius = max(4, int(f["enc_radius"]))

        if r["status"] == "MATCH":
            cv2.circle(vis, (cx, cy), radius, C_MATCH, 2)
            cv2.putText(vis, f"O{r['matched_ocr_dia']}",
                        (cx - 16, cy - radius - 5),
                        FONT, 0.38, C_MATCH, 1, cv2.LINE_AA)
        else:
            cv2.circle(vis, (cx, cy), radius, C_FP, 1)
            mm = r["eq_diameter_mm"]
            cv2.putText(vis, f"~{mm:.1f}" if mm else "?",
                        (cx - 14, cy - radius - 5),
                        FONT, 0.35, C_FP, 1, cv2.LINE_AA)

    h_img, w_img = vis.shape[:2]
    x_col = max(w_img - 180, 10)
    y0    = 28
    cv2.putText(vis, "NOT DETECTED:", (x_col, y0),
                FONT, 0.42, C_MISSING, 1, cv2.LINE_AA)
    for idx, m in enumerate(sorted(missing_ocr, key=lambda x: x["diameter"])):
        y = y0 + 20 + idx * 17
        src = m["info"].get("source", "")
        cv2.putText(vis, f"  O{m['diameter']}  {src}",
                    (x_col, y), FONT, 0.35, C_MISSING, 1, cv2.LINE_AA)

    lx, ly = 8, h_img - 72
    cv2.rectangle(vis, (lx - 4, ly - 14), (lx + 230, ly + 60), (25, 25, 25), -1)
    cv2.putText(vis, "GREEN  = Matched to OCR hole",   (lx, ly),
                FONT, 0.37, C_MATCH,   1, cv2.LINE_AA)
    cv2.putText(vis, "ORANGE = Detected, not in OCR",  (lx, ly + 18),
                FONT, 0.37, C_FP,      1, cv2.LINE_AA)
    cv2.putText(vis, "RED    = OCR hole not detected", (lx, ly + 36),
                FONT, 0.37, C_MISSING, 1, cv2.LINE_AA)
    scale_txt = f"Scale: {scale:.3f} px/mm" if scale else "Scale: unknown"
    cv2.putText(vis, scale_txt, (lx, ly + 54),
                FONT, 0.37, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


def print_report(label, features, results, missing_ocr, scale, ocr):
    matched      = [r for r in results if r["status"] == "MATCH"]
    all_ocr_dias = set(
        [h["diameter"] for h in ocr.get("holes", [])] +
        [hp["diameter"] for hp in ocr.get("hole_patterns", [])]
    )
    total_ocr    = len(all_ocr_dias)
    matched_dias = sorted(r["matched_ocr_dia"] for r in matched if r["matched_ocr_dia"])
    coverage     = len(set(matched_dias))
    cov_pct      = f"{coverage / total_ocr * 100:.0f}%" if total_ocr else "N/A"
    dias_str     = ", ".join(f"Ø{d}" for d in sorted(set(matched_dias)))
    miss_str     = f"  {RED}{len(missing_ocr)} missing{RESET}" if missing_ocr else f"  {GREEN}all found{RESET}"
    print(f"[{label}] {GREEN}{coverage}/{total_ocr} features detected{RESET} ({cov_pct}) — {dias_str}{miss_str}")


def run_single(edge_image_path: str, ocr_json_path: str, label: str) -> dict:
    safe_label = label.replace(" ", "_").replace("/", "_")

    features, img_gray = extract_shape_features(edge_image_path)

    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_preview = json.load(f)
    ocr_diameters = sorted(set(
        [h["diameter"]  for h in ocr_preview.get("holes", [])] +
        [hp["diameter"] for hp in ocr_preview.get("hole_patterns", [])]
    ))
    scale = estimate_scale(features, ocr_diameters)

    results, ocr, missing_ocr = compare_with_ocr(features, ocr_json_path, scale)

    vis      = draw_annotations(img_gray, features, results, missing_ocr, scale)
    vis_path = os.path.join(OUTPUT_DIR, f"{safe_label}_annotated.png")
    cv2.imwrite(vis_path, vis)

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

    print_report(label, features, results, missing_ocr, scale, ocr)

    matched_count = sum(1 for r in results if r["status"] == "MATCH")
    return {
        "label":       label,
        "matched":     matched_count,
        "total":       len(results),
        "missing_ocr": len(missing_ocr),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if sys.platform == "win32":
        os.system("color")

    if len(sys.argv) >= 3:
        edge_path = sys.argv[1]
        ocr_path  = sys.argv[2]
        label     = sys.argv[3] if len(sys.argv) > 3 else "custom"
        if not os.path.exists(edge_path):
            print(f"{RED}[ERROR] Edge image not found: {edge_path}{RESET}")
            sys.exit(1)
        if not os.path.exists(ocr_path):
            print(f"{RED}[ERROR] OCR JSON not found: {ocr_path}{RESET}")
            sys.exit(1)
        run_single(edge_path, ocr_path, label)
        return

    summary       = []
    missing_files = []

    for sample_name, config in SAMPLES.items():
        ocr_json = config["ocr_json"]
        if not os.path.exists(ocr_json):
            continue

        for view, edge_path in config["views"].items():
            label = f"{sample_name}_{view}"
            if not os.path.exists(edge_path):
                missing_files.append((label, edge_path))
                continue
            result = run_single(edge_path, ocr_json, label)
            summary.append(result)

    if not summary and os.path.exists(LEGACY_EDGE):
        ocr_json = SAMPLES["N001"]["ocr_json"]
        result   = run_single(LEGACY_EDGE, ocr_json, "N001_legacy")
        summary.append(result)

    if not summary:
        print(f"{YELLOW}No edge images processed.{RESET}")
    if missing_files:
        for lbl, path in missing_files:
            print(f"{YELLOW}[MISSING]{RESET} {lbl} — run: python quick_test.py cad.png {os.path.basename(path)}")


if __name__ == "__main__":
    main()

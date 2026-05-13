"""
Phase 1 — Main Feature Extraction Pipeline
===========================================

Sample mapping:
  sample_01_N001  → cad2.png (front), cad4.png (rear)
                  → real2.png (front), real4.png (rear)
                  → OCR: 1-02-1065-N001_parsed.json

  sample_02_BEV820 → cad1.png (front), cad3.png (rear)
                   → real1.png (front), real3.png (rear)
                   → OCR: blueprint_page_parsed.json

Workflow:
  quick_test.py (alignment)
        ↓
  edges_<sample>_<view>.png   (e.g. edges_N001_front.png)
        ↓
  feature_extractor.py  (this file)
        ↓
  extract shape features + compare against OCR JSON
        ↓
  features/outputs/<sample>_<view>_comparison.json
  features/outputs/<sample>_<view>_annotated.png

Usage:
  # Run all samples and views:
  python features/scripts/feature_extractor.py

  # Run a specific edge image against a specific OCR JSON:
  python features/scripts/feature_extractor.py <edge_image> <ocr_json> [label]
"""

import cv2
import numpy as np
import json
import sys
import os
import math

# =====================================================
# PATHS
# =====================================================

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

# Fallback: old single debug_final_edges.png
LEGACY_EDGE = os.path.join(ROOT_DIR, "debug_final_edges.png")

MATCH_TOLERANCE = 0.15   # mm
SEP = "=" * 62


# =====================================================
# STEP 1 — EXTRACT SHAPE FEATURES
# =====================================================

def extract_shape_features(edge_image_path: str):
    img = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load edge image: {edge_image_path}")

    print(f"  [INFO] Image size     : {img.shape[1]} x {img.shape[0]} px")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"  [INFO] Raw contours   : {len(contours)}")

    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 20:
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

        hu_raw = cv2.HuMoments(M).flatten()
        hu = [
            float(-np.sign(v) * np.log10(abs(v))) if v != 0 else 0.0
            for v in hu_raw
        ]

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area) / hull_area if hull_area > 0 else 0
        (_, _), enc_radius = cv2.minEnclosingCircle(cnt)
        eq_diameter = math.sqrt(4 * area / math.pi)
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        rect  = cv2.minAreaRect(cnt)
        angle = rect[-1]
        approx   = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)

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

    print(f"  [INFO] Valid features : {len(features)}")
    return features, img


# =====================================================
# STEP 2 — ESTIMATE SCALE
# =====================================================

def estimate_scale(features: list, ocr_diameters: list):
    circles = sorted(
        [f for f in features if f["circularity"] > 0.75],
        key=lambda f: f["area"], reverse=True
    )
    if not circles or not ocr_diameters:
        return None

    best_scale, best_score = None, 999
    for c in circles[:10]:
        px_d = c["equivalent_diameter"]
        for ocr_d in ocr_diameters:
            if ocr_d < 1.0:
                continue
            scale = px_d / ocr_d
            if not (0.5 < scale < 500):
                continue
            errors = []
            for other in circles[:20]:
                mm = other["equivalent_diameter"] / scale
                closest = min(ocr_diameters, key=lambda d: abs(d - mm))
                errors.append(abs(closest - mm))
            avg_err = sum(errors) / len(errors)
            if avg_err < best_score:
                best_score = avg_err
                best_scale = scale

    return best_scale


# =====================================================
# STEP 3 — COMPARE WITH OCR
# =====================================================

def compare_with_ocr(features: list, ocr_json_path: str, scale: float):
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr = json.load(f)

    ocr_holes = {}
    for h in ocr.get("holes", []):
        d = h["diameter"]
        ocr_holes[d] = {"source": "hole", "through": h.get("through")}
    for hp in ocr.get("hole_patterns", []):
        d = hp["diameter"]
        ocr_holes[d] = {"source": f"hole_pattern ({hp['count']}x)", "through": hp.get("through")}

    ocr_diameters = sorted(ocr_holes.keys())

    results = []
    for f in features:
        if f["circularity"] < 0.55:
            continue

        px_d = f["equivalent_diameter"]
        mm_d = round(px_d / scale, 3) if scale else None

        matched_dia  = None
        matched_info = None
        error_mm     = None

        if mm_d is not None:
            best_err = MATCH_TOLERANCE + 1
            for ocr_d in ocr_diameters:
                err = abs(ocr_d - mm_d)
                if err < best_err:
                    best_err     = err
                    matched_dia  = ocr_d
                    matched_info = ocr_holes[ocr_d]
            if best_err > MATCH_TOLERANCE:
                matched_dia  = None
                matched_info = None
            else:
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

    return results, ocr


# =====================================================
# STEP 4 — VISUALISE
# =====================================================

def draw_annotations(img_gray, features, results):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    result_map = {r["contour_id"]: r for r in results}

    for f in features:
        r = result_map.get(f["id"])
        if r is None:
            continue
        cx     = f["centroid"]["x"]
        cy     = f["centroid"]["y"]
        radius = max(3, int(f["enc_radius"]))

        if r["status"] == "MATCH":
            color = (0, 220, 0)
            label = f"O{r['matched_ocr_dia']}"
        else:
            color = (0, 140, 255)
            mm    = r["eq_diameter_mm"]
            label = f"~{mm}mm" if mm else "?"

        cv2.circle(vis, (cx, cy), radius, color, 2)
        cv2.putText(vis, label, (cx - 15, cy - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


# =====================================================
# STEP 5 — PRINT REPORT
# =====================================================

def print_report(label, features, results, scale, ocr):
    matched   = [r for r in results if r["status"] == "MATCH"]
    unmatched = [r for r in results if r["status"] == "UNMATCHED"]
    circular  = [f for f in features if f["circularity"] > 0.55]
    match_pct = len(matched) / len(results) * 100 if results else 0

    print(f"\n{SEP}")
    print(f"RESULTS — {label}")
    print(SEP)
    print(f"  Total contours    : {len(features)}")
    print(f"  Circular features : {len(circular)}")
    print(f"  Scale estimate    : {f'{scale:.3f} px/mm' if scale else 'unknown'}")
    print(f"  OCR hole diameters: {sorted(set(r['matched_ocr_dia'] for r in matched if r['matched_ocr_dia']))}")

    print(f"\n  MATCHES:")
    if matched:
        for r in matched:
            info = r["ocr_feature_info"] or {}
            print(f"    [OK] contour {r['contour_id']:5d}  "
                  f"circ={r['circularity']:.3f}  "
                  f"Ø_mm={r['eq_diameter_mm']:6.3f}  "
                  f"→ OCR Ø{r['matched_ocr_dia']}  "
                  f"err={r['error_mm']}mm  [{info.get('source','')}]")
    else:
        print(f"    none")

    print(f"\n  UNMATCHED ({len(unmatched)}):")
    for r in unmatched[:8]:
        print(f"    [--] contour {r['contour_id']:5d}  "
              f"circ={r['circularity']:.3f}  "
              f"Ø_mm={r['eq_diameter_mm']}")

    print(f"\n  Match rate : {len(matched)}/{len(results)} ({match_pct:.1f}%)")
    print(f"  Part       : {ocr.get('part_number')}  Rev={ocr.get('revision')}")
    print(f"  Finish     : {ocr.get('surface_finish')}")
    print(f"  Units      : {ocr.get('units')}")
    print(f"  Threads    : {[t['thread'] for t in ocr.get('metric_threads', [])]}")


# =====================================================
# CORE RUNNER
# =====================================================

def run_single(edge_image_path: str, ocr_json_path: str, label: str):
    """Run the full pipeline for one edge image + OCR JSON pair."""

    print(f"\n{SEP}")
    print(f"PROCESSING: {label}")
    print(f"  Edge image : {os.path.basename(edge_image_path)}")
    print(f"  OCR JSON   : {os.path.basename(ocr_json_path)}")
    print(SEP)

    # Step 1 — extract
    features, img_gray = extract_shape_features(edge_image_path)

    # Step 2 — scale
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_preview = json.load(f)
    ocr_diameters = sorted(set(
        [h["diameter"] for h in ocr_preview.get("holes", [])] +
        [hp["diameter"] for hp in ocr_preview.get("hole_patterns", [])]
    ))
    scale = estimate_scale(features, ocr_diameters)
    print(f"  [INFO] Scale estimate : {f'{scale:.3f} px/mm' if scale else 'unknown'}")

    # Step 3 — compare
    results, ocr = compare_with_ocr(features, ocr_json_path, scale)

    # Step 4 — visualise
    vis = draw_annotations(img_gray, features, results)
    safe_label = label.replace(" ", "_").replace("/", "_")
    vis_path   = os.path.join(OUTPUT_DIR, f"{safe_label}_annotated.png")
    cv2.imwrite(vis_path, vis)

    # Step 5 — save JSON
    feat_path  = os.path.join(OUTPUT_DIR, f"{safe_label}_shape_features.json")
    match_path = os.path.join(OUTPUT_DIR, f"{safe_label}_comparison.json")

    with open(feat_path, "w") as f:
        json.dump(features, f, indent=4)

    with open(match_path, "w") as f:
        json.dump({
            "label":          label,
            "edge_image":     edge_image_path,
            "ocr_json":       ocr_json_path,
            "scale_px_mm":    round(scale, 4) if scale else None,
            "total_features": len(features),
            "results":        results,
        }, f, indent=4)

    # Step 6 — report
    print_report(label, features, results, scale, ocr)

    print(f"\n  [OK] {safe_label}_shape_features.json")
    print(f"  [OK] {safe_label}_comparison.json")
    print(f"  [OK] {safe_label}_annotated.png")

    matched = sum(1 for r in results if r["status"] == "MATCH")
    return {"label": label, "matched": matched, "total": len(results)}


# =====================================================
# MAIN
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- CLI mode: single custom run ---
    if len(sys.argv) >= 3:
        edge_path = sys.argv[1]
        ocr_path  = sys.argv[2]
        label     = sys.argv[3] if len(sys.argv) > 3 else "custom"
        if not os.path.exists(edge_path):
            print(f"[ERROR] Edge image not found: {edge_path}")
            sys.exit(1)
        if not os.path.exists(ocr_path):
            print(f"[ERROR] OCR JSON not found: {ocr_path}")
            sys.exit(1)
        run_single(edge_path, ocr_path, label)
        return

    # --- Auto mode: run all samples and views ---
    print(f"\n{SEP}")
    print("PHASE 1 — FEATURE EXTRACTION PIPELINE")
    print(f"{SEP}")
    print("Sample mapping:")
    print("  N001   → cad2 (front), cad4 (rear) | real2 (front), real4 (rear)")
    print("  BEV820 → cad1 (front), cad3 (rear) | real1 (front), real3 (rear)")
    print(f"{SEP}")

    summary = []
    missing = []

    for sample_name, config in SAMPLES.items():
        ocr_json = config["ocr_json"]

        if not os.path.exists(ocr_json):
            print(f"\n[SKIP] OCR JSON not found for {sample_name}: {ocr_json}")
            continue

        for view, edge_path in config["views"].items():
            label = f"{sample_name}_{view}"

            if not os.path.exists(edge_path):
                missing.append((label, edge_path))
                print(f"\n[SKIP] Edge image not found: {os.path.basename(edge_path)}")
                print(f"       Run quick_test.py first to generate it.")
                continue

            result = run_single(edge_path, ocr_json, label)
            summary.append(result)

    # --- Fallback: use legacy debug_final_edges.png if no named edges exist ---
    if not summary and os.path.exists(LEGACY_EDGE):
        print(f"\n[INFO] No named edge images found. Using legacy: debug_final_edges.png")
        print(f"[INFO] Defaulting to sample_01_N001 OCR JSON.")
        ocr_json = SAMPLES["N001"]["ocr_json"]
        result = run_single(LEGACY_EDGE, ocr_json, "N001_legacy")
        summary.append(result)

    # --- Final summary ---
    print(f"\n{SEP}")
    print("OVERALL SUMMARY")
    print(SEP)

    if summary:
        for s in summary:
            pct = s["matched"] / s["total"] * 100 if s["total"] else 0
            print(f"  {s['label']:20s}  matched={s['matched']:3d}/{s['total']:3d}  ({pct:.1f}%)")
    else:
        print("  No edge images processed.")

    if missing:
        print(f"\n  MISSING EDGE IMAGES (run quick_test.py to generate):")
        for label, path in missing:
            print(f"    {label:20s} → {os.path.basename(path)}")
        print(f"\n  How to generate edge images:")
        print(f"    N001  front : python quick_test.py cad2.png real2.png")
        print(f"    N001  rear  : python quick_test.py cad4.png real4.png")
        print(f"    BEV820 front: python quick_test.py cad1.png real1.png")
        print(f"    BEV820 rear : python quick_test.py cad3.png real3.png")
        print(f"  Then rename debug_final_edges.png to the appropriate name.")

    print(SEP)
    print("[DONE] Phase 1 complete.\n")


if __name__ == "__main__":
    main()

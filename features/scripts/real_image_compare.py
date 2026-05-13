"""
Phase 1 — Part 2: Real Image vs OCR Feature Comparison
Runs edge detection + contour extraction on real photos,
extracts geometric features, and compares against OCR parsed JSON.

Input images : real1.png ... real5.png  (in PHASE-1 root)
OCR JSON     : features/1-02-1065-N001_parsed.json
Outputs      :
  real_shape_features.json     — all contour features per image
  real_ocr_comparison.json     — match results
  real_annotated_*.png         — visualisation per image
"""

import cv2
import numpy as np
import json
import os
import math

# =====================================================
# PATHS
# =====================================================

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.dirname(BASE)
ROOT     = os.path.dirname(FEATURES)
OCR_JSON = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")

IMAGE_FILES = [
    os.path.join(ROOT, f"real{i}.png") for i in range(1, 6)
]
# Also try .jpeg variants
EXTRA = [os.path.join(ROOT, "real.jpeg"), os.path.join(ROOT, "real1.jpeg")]
IMAGE_FILES = [p for p in IMAGE_FILES + EXTRA if os.path.exists(p)]

MATCH_TOL_PX  = 0.12   # fraction of image diagonal for diameter tolerance
MATCH_TOL_MM  = 0.5    # mm tolerance when scale is known

# =====================================================
# STEP 1 — LOAD OCR FEATURES
# =====================================================

with open(OCR_JSON, encoding="utf-8") as f:
    ocr = json.load(f)

ocr_hole_diameters = sorted(set(
    [h["diameter"] for h in ocr.get("holes", [])] +
    [hp["diameter"] for hp in ocr.get("hole_patterns", [])]
))

ocr_radii = [r["radius"] for r in ocr.get("radii", [])]

# =====================================================
# STEP 2 — PREPROCESS IMAGE
# =====================================================

def preprocess(img):
    """Returns edge image suitable for contour detection."""
    # Upscale for better small-feature detection
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Morphological close to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Canny on top for clean edges
    edges = cv2.Canny(gray, 30, 100)
    combined = cv2.bitwise_or(thresh, edges)

    return img, combined   # return upscaled img + edge map


# =====================================================
# STEP 3 — EXTRACT CONTOUR FEATURES
# =====================================================

def extract_contours(edge_img, min_area=50):
    """
    Finds contours and computes shape features.
    Returns list of feature dicts.
    """
    contours, _ = cv2.findContours(
        edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h != 0 else 0

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Circularity
        circularity = (4 * math.pi * area) / (perimeter ** 2)

        # Equivalent diameter
        eq_diam = math.sqrt(4 * area / math.pi)

        # Min enclosing circle
        (ex, ey), enc_radius = cv2.minEnclosingCircle(cnt)

        # Convex hull solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Polygon approx
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        # Min area rect angle
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        # Shape classification
        if circularity > 0.80 and solidity > 0.85:
            shape = "circle"
        elif circularity > 0.60 and solidity > 0.75:
            shape = "ellipse"
        elif vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            ar = float(max(w, h)) / min(w, h) if min(w, h) > 0 else 0
            shape = "square" if ar < 1.3 else "rectangle"
        elif aspect > 3.5:
            shape = "slot"
        else:
            shape = "polygon"

        features.append({
            "id":                 i,
            "shape":              shape,
            "centroid":           {"x": cx, "y": cy},
            "area_px":            float(area),
            "perimeter_px":       float(perimeter),
            "circularity":        round(circularity, 4),
            "solidity":           round(solidity, 4),
            "eq_diameter_px":     round(eq_diam, 2),
            "enc_radius_px":      round(enc_radius, 2),
            "bounding_box":       {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "aspect_ratio":       round(aspect, 3),
            "vertices":           vertices,
            "rotation_angle":     round(angle, 2),
        })

    return features


# =====================================================
# STEP 4 — ESTIMATE SCALE (px/mm)
# =====================================================

def estimate_scale(features, ocr_hole_diameters):
    """
    Tries to estimate px/mm scale by matching the most
    circular detected feature to the largest known OCR hole diameter.
    Returns scale factor (px per mm) or None.
    """
    circles = [f for f in features if f["circularity"] > 0.75]
    if not circles or not ocr_hole_diameters:
        return None

    # Sort by area descending — largest circle likely a main bore
    circles_sorted = sorted(circles, key=lambda c: c["area_px"], reverse=True)

    # Try matching top circles to OCR diameters
    best_scale = None
    best_score = 999

    for c in circles_sorted[:8]:
        px_diam = c["eq_diameter_px"]
        for ocr_d in ocr_hole_diameters:
            if ocr_d < 1.0:
                continue
            scale = px_diam / ocr_d   # px per mm
            if 1.0 < scale < 200:     # sanity range
                # Score: how well does this scale explain other circles?
                errors = []
                for other in circles_sorted[:15]:
                    other_mm = other["eq_diameter_px"] / scale
                    closest = min(ocr_hole_diameters, key=lambda d: abs(d - other_mm))
                    errors.append(abs(closest - other_mm))
                avg_err = sum(errors) / len(errors)
                if avg_err < best_score:
                    best_score = avg_err
                    best_scale = scale

    return best_scale


# =====================================================
# STEP 5 — MATCH FEATURES TO OCR
# =====================================================

def match_to_ocr(features, scale, ocr_hole_diameters, ocr_radii_mm):
    """
    Converts pixel measurements to mm using scale,
    then matches against OCR values.
    """
    results = []
    tol = MATCH_TOL_MM if scale else None

    for f in features:
        if f["circularity"] < 0.55:
            continue   # only match circular features

        px_diam = f["eq_diameter_px"]
        mm_diam = round(px_diam / scale, 3) if scale else None

        best_ocr = None
        best_err = None

        if mm_diam is not None:
            for ocr_d in ocr_hole_diameters:
                err = abs(ocr_d - mm_diam)
                if best_err is None or err < best_err:
                    best_err = err
                    best_ocr = ocr_d
            if best_err > tol:
                best_ocr = None
                best_err = None

        results.append({
            "contour_id":      f["id"],
            "shape":           f["shape"],
            "circularity":     f["circularity"],
            "eq_diameter_px":  px_diam,
            "eq_diameter_mm":  mm_diam,
            "matched_ocr_dia": best_ocr,
            "error_mm":        round(best_err, 4) if best_err is not None else None,
            "status":          "MATCH" if best_ocr is not None else "UNMATCHED",
            "centroid":        f["centroid"],
        })

    return results


# =====================================================
# STEP 6 — VISUALISE
# =====================================================

def draw_annotations(img, features, match_results, scale):
    vis = img.copy()
    match_map = {r["contour_id"]: r for r in match_results}

    for f in features:
        if f["circularity"] < 0.55:
            continue
        cx = f["centroid"]["x"]
        cy = f["centroid"]["y"]
        r  = int(f["enc_radius_px"])

        mr = match_map.get(f["id"])
        if mr and mr["status"] == "MATCH":
            color = (0, 255, 0)   # green = matched
            label = f"Ø{mr['matched_ocr_dia']}"
        else:
            color = (0, 100, 255) # orange = unmatched
            mm = mr["eq_diameter_mm"] if mr and mr["eq_diameter_mm"] else "?"
            label = f"~{mm}mm" if isinstance(mm, float) else "?"

        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.putText(vis, label, (cx - 20, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis


# =====================================================
# STEP 7 — PROCESS ALL IMAGES
# =====================================================

all_results = []

for img_path in IMAGE_FILES:
    img_name = os.path.basename(img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[SKIP] Cannot load: {img_path}")
        continue

    img_up, edges = preprocess(img_bgr)
    h_orig, w_orig = img_bgr.shape[:2]

    features = extract_contours(edges, min_area=80)
    scale = estimate_scale(features, ocr_hole_diameters)
    matches = match_to_ocr(features, scale, ocr_hole_diameters, ocr_radii)
    matched_count = sum(1 for m in matches if m["status"] == "MATCH")

    matched_dias = sorted(set(m["matched_ocr_dia"] for m in matches if m["matched_ocr_dia"]))
    dias_str = ", ".join(f"Ø{d}" for d in matched_dias)
    print(f"[{img_name}] {matched_count} matches — {dias_str}")

    vis = draw_annotations(img_up, features, matches, scale)
    out_name = f"real_annotated_{os.path.splitext(img_name)[0]}.png"
    cv2.imwrite(os.path.join(FEATURES, "outputs", out_name), vis)

    all_results.append({
        "image":           img_name,
        "size":            {"w": w_orig, "h": h_orig},
        "scale_px_per_mm": round(scale, 4) if scale else None,
        "total_contours":  len(features),
        "circular_count":  len([f for f in features if f["circularity"] > 0.75]),
        "matched_count":   matched_count,
        "matches":         matches,
    })

with open(os.path.join(FEATURES, "outputs", "real_ocr_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=4)

total_matched  = sum(r["matched_count"] for r in all_results)
total_circular = sum(r["circular_count"] for r in all_results)
overall_pct    = total_matched / total_circular * 100 if total_circular else 0
print(f"[real_image_compare] {total_matched}/{total_circular} total matches ({overall_pct:.0f}%) across {len(all_results)} images → real_ocr_comparison.json")

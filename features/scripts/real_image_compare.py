import cv2
import numpy as np
import json
import os
import sys
import math

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.dirname(BASE)
ROOT     = os.path.dirname(FEATURES)
OCR_JSON = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")

MATCH_TOL_FRAC = 0.15
MATCH_TOL_MIN  = 0.50


# =====================================================
# STEP 1 — LOAD OCR FEATURES
# =====================================================

def load_ocr(ocr_json_path):
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr = json.load(f)

    ocr_hole_diameters = sorted(set(
        [h["diameter"] for h in ocr.get("holes", [])] +
        [hp["diameter"] for hp in ocr.get("hole_patterns", [])]
    ))
    ocr_radii = [r["radius"] for r in ocr.get("radii", [])]
    return ocr, ocr_hole_diameters, ocr_radii


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

def _tol(ocr_d):
    if ocr_d <= 3.0:
        return max(1.2, ocr_d * 0.60)
    if ocr_d <= 6.0:
        return max(0.8, ocr_d * 0.25)
    return max(MATCH_TOL_MIN, ocr_d * MATCH_TOL_FRAC)


def match_to_ocr(features, scale, ocr_hole_diameters):
    """
    Converts pixel measurements to mm using scale,
    then matches against OCR values with per-diameter tolerance.
    Uses greedy one-to-one assignment so each OCR diameter is claimed once.
    """
    if not scale:
        return []

    # Build candidates: (error, feature, ocr_diameter)
    candidates = []
    for f in features:
        if f["circularity"] < 0.55:
            continue
        px_diam = f["eq_diameter_px"]
        mm_diam = px_diam / scale
        for ocr_d in ocr_hole_diameters:
            err = abs(ocr_d - mm_diam)
            if err <= _tol(ocr_d):
                candidates.append((err, f, ocr_d))

    # Greedy assignment — smallest error wins
    candidates.sort(key=lambda x: x[0])
    claimed_ocr     = set()
    claimed_contour = set()
    assignments     = {}   # contour_id -> (ocr_d, error)

    for err, f, ocr_d in candidates:
        if f["id"] in claimed_contour or ocr_d in claimed_ocr:
            continue
        claimed_contour.add(f["id"])
        claimed_ocr.add(ocr_d)
        assignments[f["id"]] = (ocr_d, round(err, 4))

    results = []
    for f in features:
        if f["circularity"] < 0.55:
            continue
        px_diam = f["eq_diameter_px"]
        mm_diam = round(px_diam / scale, 3)

        if f["id"] in assignments:
            ocr_d, error = assignments[f["id"]]
            results.append({
                "contour_id":      f["id"],
                "shape":           f["shape"],
                "circularity":     f["circularity"],
                "eq_diameter_px":  px_diam,
                "eq_diameter_mm":  mm_diam,
                "matched_ocr_dia": ocr_d,
                "error_mm":        error,
                "status":          "MATCH",
                "centroid":        f["centroid"],
            })
        else:
            results.append({
                "contour_id":      f["id"],
                "shape":           f["shape"],
                "circularity":     f["circularity"],
                "eq_diameter_px":  px_diam,
                "eq_diameter_mm":  mm_diam,
                "matched_ocr_dia": None,
                "error_mm":        None,
                "status":          "UNMATCHED",
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
            label = f"O{mr['matched_ocr_dia']}"
        else:
            color = (0, 100, 255) # orange = unmatched
            mm = mr["eq_diameter_mm"] if mr and mr["eq_diameter_mm"] else "?"
            label = f"~{mm}mm" if isinstance(mm, float) else "?"

        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.putText(vis, label, (cx - 20, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis


# =====================================================
# STEP 7 — PROCESS A SINGLE IMAGE
# =====================================================

def process_image(img_path, ocr_json_path=OCR_JSON):
    img_name = os.path.basename(img_path)

    ocr, ocr_hole_diameters, ocr_radii = load_ocr(ocr_json_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot load: {img_path}")
        return None

    h_orig, w_orig = img_bgr.shape[:2]
    img_up, edges = preprocess(img_bgr)

    features = extract_contours(edges, min_area=80)
    scale = estimate_scale(features, ocr_hole_diameters)
    matches = match_to_ocr(features, scale, ocr_hole_diameters)
    matched_count = sum(1 for m in matches if m["status"] == "MATCH")

    matched_dias = sorted(set(m["matched_ocr_dia"] for m in matches if m["matched_ocr_dia"]))
    dias_str = ", ".join(f"O{d}" for d in matched_dias)
    print(f"[{img_name}] {matched_count} matches — {dias_str}")

    vis = draw_annotations(img_up, features, matches, scale)
    out_name = f"real_annotated_{os.path.splitext(img_name)[0]}.png"
    out_path = os.path.join(FEATURES, "outputs", out_name)
    cv2.imwrite(out_path, vis)
    print(f"[{img_name}] Annotated image saved: {out_path}")

    result = {
        "image":           img_name,
        "size":            {"w": w_orig, "h": h_orig},
        "scale_px_per_mm": round(scale, 4) if scale else None,
        "total_contours":  len(features),
        "circular_count":  len([f for f in features if f["circularity"] > 0.75]),
        "matched_count":   matched_count,
        "matches":         matches,
    }

    out_json = os.path.join(FEATURES, "outputs", "real_ocr_comparison.json")
    with open(out_json, "w") as f:
        json.dump([result], f, indent=4)
    print(f"[{img_name}] Results saved: {out_json}")

    return result


# =====================================================
# ENTRY POINT
# =====================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python real_image_compare.py <image_path> [ocr_json_path]")
        print("Example: python features\\scripts\\real_image_compare.py real2.png")
        sys.exit(1)

    img_arg = sys.argv[1]
    # Allow just filename (e.g. real2.png) — resolve relative to ROOT
    if not os.path.isabs(img_arg) and not os.path.exists(img_arg):
        img_arg = os.path.join(ROOT, img_arg)

    if not os.path.exists(img_arg):
        print(f"[ERROR] Image not found: {img_arg}")
        sys.exit(1)

    ocr_arg = sys.argv[2] if len(sys.argv) >= 3 else OCR_JSON
    if not os.path.exists(ocr_arg):
        print(f"[ERROR] OCR JSON not found: {ocr_arg}")
        sys.exit(1)

    os.makedirs(os.path.join(FEATURES, "outputs"), exist_ok=True)
    process_image(img_arg, ocr_arg)


if __name__ == "__main__":
    main()

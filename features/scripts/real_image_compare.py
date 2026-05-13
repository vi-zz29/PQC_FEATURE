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

# Accept a single image path from CLI, or fall back to scanning the folder
if len(sys.argv) >= 2:
    IMAGE_FILES = [sys.argv[1]] if os.path.exists(sys.argv[1]) else []
    if not IMAGE_FILES:
        print(f"[ERROR] Image not found: {sys.argv[1]}")
        sys.exit(1)
else:
    IMAGE_FILES = [os.path.join(ROOT, f"real{i}.png") for i in range(1, 6)]
    EXTRA       = [os.path.join(ROOT, "real.jpeg"), os.path.join(ROOT, "real1.jpeg")]
    IMAGE_FILES = [p for p in IMAGE_FILES + EXTRA if os.path.exists(p)]

# FIX 1: Adaptive tolerance — 15% of OCR diameter, floor 0.5mm
MATCH_TOL_FRAC = 0.15
MATCH_TOL_MIN  = 0.50

with open(OCR_JSON, encoding="utf-8") as f:
    ocr = json.load(f)

# Include PCD features as large-circle scale anchors
ocr_hole_diameters = sorted(set(
    [h["diameter"]  for h in ocr.get("holes", [])] +
    [hp["diameter"] for hp in ocr.get("hole_patterns", [])] +
    [p["diameter"]  for p in ocr.get("pcd_features", [])]
))
ocr_radii = [r["radius"] for r in ocr.get("radii", [])]


# ---------------------------------------------------------------------------
# FIX 4: Improved preprocessing — CLAHE + open + close + Canny
# ---------------------------------------------------------------------------

def preprocess(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bilateral filter — denoises while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    # Morphological open: remove tiny specks before closing
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open, iterations=1)

    # Close gaps in hole rings
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Canny with lower thresholds to catch faint edges
    edges = cv2.Canny(gray, 20, 80)
    combined = cv2.bitwise_or(thresh, edges)

    return img, combined


# ---------------------------------------------------------------------------
# FIX 9: Adaptive thresholds based on image resolution and contour density
# ---------------------------------------------------------------------------

def _compute_adaptive_thresholds(img_shape, n_contours):
    h, w = img_shape[:2]
    area = h * w
    if area > 1_000_000:
        circ_thresh, solid_thresh = 0.70, 0.80
    elif area > 400_000:
        circ_thresh, solid_thresh = 0.63, 0.76
    else:
        circ_thresh, solid_thresh = 0.56, 0.70

    if n_contours > 300:
        circ_thresh  = min(circ_thresh + 0.05, 0.88)
        solid_thresh = min(solid_thresh + 0.03, 0.93)

    return circ_thresh, solid_thresh


# ---------------------------------------------------------------------------
# FIX 10: Large contour refinement — ellipse fitting
# ---------------------------------------------------------------------------

def _refined_diameter(cnt, eq_diameter):
    if len(cnt) >= 5 and eq_diameter > 20:
        try:
            ellipse = cv2.fitEllipse(cnt)
            return float(np.mean(ellipse[1]))
        except Exception:
            pass
    return eq_diameter


# ---------------------------------------------------------------------------
# FIX 2: Spatial deduplication — cluster by centroid, keep highest circularity
# ---------------------------------------------------------------------------

def _deduplicate(features):
    if not features:
        return features
    kept = []
    used = set()
    sorted_feats = sorted(features, key=lambda f: f["circularity"], reverse=True)
    for i, f in enumerate(sorted_feats):
        if i in used:
            continue
        cx, cy = f["centroid"]["x"], f["centroid"]["y"]
        kept.append(f)
        for j, other in enumerate(sorted_feats):
            if j <= i or j in used:
                continue
            ox, oy = other["centroid"]["x"], other["centroid"]["y"]
            dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            d_ratio = abs(f["eq_diameter_px"] - other["eq_diameter_px"]) / max(f["eq_diameter_px"], 1.0)
            if dist < 12 and d_ratio < 0.25:
                used.add(j)
    return kept


# ---------------------------------------------------------------------------
# FIX 3: Combined contour validation
# ---------------------------------------------------------------------------

def _is_valid_candidate(circ, solid, enc_r, eq_d, circ_thresh, solid_thresh):
    if circ < circ_thresh or solid < solid_thresh:
        return False
    enc_d = enc_r * 2
    if enc_d > 0 and min(eq_d, enc_d) / max(eq_d, enc_d) < 0.60:
        return False
    return True


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours(edge_img, img_shape, min_area=50):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circ_thresh, solid_thresh = _compute_adaptive_thresholds(img_shape, len(contours))

    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h != 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        hull        = cv2.convexHull(cnt)
        hull_area   = cv2.contourArea(hull)
        solidity    = float(area) / hull_area if hull_area > 0 else 0
        (_, _), enc_radius = cv2.minEnclosingCircle(cnt)
        eq_diam     = math.sqrt(4 * area / math.pi)
        eq_diam     = _refined_diameter(cnt, eq_diam)  # FIX 10

        approx   = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)
        rect     = cv2.minAreaRect(cnt)
        angle    = rect[-1]

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
            "id":             i,
            "shape":          shape,
            "centroid":       {"x": cx, "y": cy},
            "area_px":        float(area),
            "perimeter_px":   float(perimeter),
            "circularity":    round(circularity, 4),
            "solidity":       round(solidity, 4),
            "eq_diameter_px": round(eq_diam, 2),
            "enc_radius_px":  round(enc_radius, 2),
            "bounding_box":   {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "aspect_ratio":   round(aspect, 3),
            "vertices":       vertices,
            "rotation_angle": round(angle, 2),
        })

    # FIX 2: Spatial deduplication
    features = _deduplicate(features)
    return features, circ_thresh, solid_thresh


# ---------------------------------------------------------------------------
# FIX 1: RANSAC consensus scale estimation
# ---------------------------------------------------------------------------

def estimate_scale(features, ocr_hole_diameters):
    candidates = sorted(
        [f for f in features if f["circularity"] > 0.50],
        key=lambda c: c["area_px"], reverse=True
    )
    if not candidates or not ocr_hole_diameters:
        return None

    all_scales = []
    for c in candidates[:40]:
        px_diam = c["eq_diameter_px"]
        for ocr_d in ocr_hole_diameters:
            if ocr_d < 1.0:
                continue
            s = px_diam / ocr_d
            if 0.3 < s < 5000:
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
        errors = []
        for c in candidates[:40]:
            mm = c["eq_diameter_px"] / s_med
            closest = min(ocr_hole_diameters, key=lambda d: abs(d - mm))
            errors.append(abs(closest - mm) / max(closest, 1.0))
        score = sum(errors) / len(errors)
        if n > best_inliers or (n == best_inliers and score < best_score):
            best_inliers = n
            best_scale   = s_med
            best_score   = score

    return best_scale


# ---------------------------------------------------------------------------
# FIX 5: Hough circle cross-check
# ---------------------------------------------------------------------------

def get_hough_diameters(img_gray, scale):
    if scale is None or scale <= 0:
        return []
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=8,
        param1=50, param2=22,
        minRadius=3, maxRadius=int(min(img_gray.shape) * 0.55)
    )
    if circles is None:
        return []
    return [round(r * 2 / scale, 3) for _, _, r in circles[0]]


# ---------------------------------------------------------------------------
# FIX 6: Match features to OCR with adaptive tolerance
# ---------------------------------------------------------------------------

def match_to_ocr(features, scale, ocr_hole_diameters, circ_thresh, solid_thresh,
                 hough_diameters=None):
    if hough_diameters is None:
        hough_diameters = []
    results = []

    for f in features:
        # FIX 3: Combined validation
        if not _is_valid_candidate(
            f["circularity"], f["solidity"],
            f["enc_radius_px"], f["eq_diameter_px"],
            circ_thresh, solid_thresh
        ):
            continue

        px_diam = f["eq_diameter_px"]
        mm_diam = round(px_diam / scale, 3) if scale else None

        best_ocr = None
        best_err = None

        if mm_diam is not None:
            for ocr_d in ocr_hole_diameters:
                err = abs(ocr_d - mm_diam)
                tol = max(MATCH_TOL_MIN, ocr_d * MATCH_TOL_FRAC)
                if err <= tol:
                    if best_err is None or err < best_err:
                        best_err = err
                        best_ocr = ocr_d

        # FIX 5: Hough confirmation
        hough_confirmed = False
        if mm_diam is not None and hough_diameters:
            hough_confirmed = any(
                abs(h - mm_diam) / max(mm_diam, 1.0) < 0.15
                for h in hough_diameters
            )

        results.append({
            "contour_id":      f["id"],
            "shape":           f["shape"],
            "circularity":     f["circularity"],
            "solidity":        f["solidity"],
            "eq_diameter_px":  px_diam,
            "eq_diameter_mm":  mm_diam,
            "matched_ocr_dia": best_ocr,
            "error_mm":        round(best_err, 4) if best_err is not None else None,
            "status":          "MATCH" if best_ocr is not None else "UNMATCHED",
            "hough_confirmed": hough_confirmed,
            "centroid":        f["centroid"],
        })

    return results


def draw_annotations(img, features, match_results, scale):
    vis = img.copy()
    match_map = {r["contour_id"]: r for r in match_results}

    for f in features:
        if f["circularity"] < 0.50:
            continue
        cx = f["centroid"]["x"]
        cy = f["centroid"]["y"]
        r  = int(f["enc_radius_px"])
        mr = match_map.get(f["id"])
        if mr and mr["status"] == "MATCH":
            color = (0, 255, 0)
            label = f"Ø{mr['matched_ocr_dia']}"
        else:
            color = (0, 100, 255)
            mm = mr["eq_diameter_mm"] if mr and mr["eq_diameter_mm"] else "?"
            label = f"~{mm}mm" if isinstance(mm, float) else "?"
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.putText(vis, label, (cx - 20, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


# ---------------------------------------------------------------------------
# Process all images
# ---------------------------------------------------------------------------

all_results = []

for img_path in IMAGE_FILES:
    img_name = os.path.basename(img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[SKIP] Cannot load: {img_path}")
        continue

    h_orig, w_orig = img_bgr.shape[:2]
    img_up, edges  = preprocess(img_bgr)

    features, circ_thresh, solid_thresh = extract_contours(edges, img_up.shape)
    scale           = estimate_scale(features, ocr_hole_diameters)

    # FIX 5: Hough on grayscale of upscaled image
    gray_up         = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    hough_diameters = get_hough_diameters(gray_up, scale)

    matches       = match_to_ocr(features, scale, ocr_hole_diameters,
                                  circ_thresh, solid_thresh, hough_diameters)
    matched_count = sum(1 for m in matches if m["status"] == "MATCH")

    matched_dias = sorted(set(m["matched_ocr_dia"] for m in matches if m["matched_ocr_dia"]))
    dias_str     = ", ".join(f"Ø{d}" for d in matched_dias)

    vis      = draw_annotations(img_up, features, matches, scale)
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
all_dias = sorted(set(
    d for r in all_results for m in r["matches"]
    if m.get("matched_ocr_dia") for d in [m["matched_ocr_dia"]]
))
dias_str = ", ".join(f"Ø{d}" for d in all_dias)
print(f"[real_image_compare] {total_matched} features detected across {len(all_results)} images — {dias_str}")

"""
Phase 1 — DXF vs OCR Feature Comparison
Extracts geometric features from all DXF files and compares
them against the OCR-parsed JSON (1-02-1065-N001_parsed.json).

Outputs:
  dxf_ocr_comparison.json   — full structured result
  Console report            — human-readable match/mismatch table
"""

import ezdxf
import json
import os
import math
from collections import defaultdict

# =====================================================
# PATHS
# =====================================================

BASE        = os.path.dirname(os.path.abspath(__file__))
FEATURES    = os.path.dirname(BASE)
DXF_DIR     = os.path.join(FEATURES, "dxf")
OCR_JSON    = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")
OUTPUT_JSON = os.path.join(FEATURES, "outputs", "dxf_ocr_comparison.json")

# =====================================================
# TOLERANCE FOR NUMERIC MATCHING (mm)
# =====================================================

MATCH_TOL = 0.15   # within 0.15 mm → considered a match

# =====================================================
# STEP 1 — EXTRACT DXF GEOMETRY
# =====================================================

def extract_dxf_features(dxf_path):
    """
    Reads a DXF file and returns:
      circles  : list of {diameter, center_x, center_y}
      arcs     : list of {radius, start_angle, end_angle, center_x, center_y}
      lines    : list of {length}
      texts    : list of str
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # INSUNITS=4 → mm, INSUNITS=1 → inch (convert to mm)
    insunits = doc.header.get("$INSUNITS", 4)
    scale = 25.4 if insunits == 1 else 1.0

    circles, arcs, lines, texts = [], [], [], []

    for entity in msp:
        etype = entity.dxftype()

        if etype == "CIRCLE":
            r = round(entity.dxf.radius * scale, 4)
            cx = round(entity.dxf.center.x * scale, 4)
            cy = round(entity.dxf.center.y * scale, 4)
            circles.append({
                "diameter": round(r * 2, 4),
                "radius":   r,
                "center":   {"x": cx, "y": cy}
            })

        elif etype == "ARC":
            r  = round(entity.dxf.radius * scale, 4)
            cx = round(entity.dxf.center.x * scale, 4)
            cy = round(entity.dxf.center.y * scale, 4)
            sa = round(entity.dxf.start_angle, 2)
            ea = round(entity.dxf.end_angle, 2)
            # arc span
            span = ea - sa if ea >= sa else (360 - sa + ea)
            arcs.append({
                "radius":      r,
                "diameter":    round(r * 2, 4),
                "start_angle": sa,
                "end_angle":   ea,
                "span_deg":    round(span, 2),
                "center":      {"x": cx, "y": cy}
            })

        elif etype == "LINE":
            x1 = entity.dxf.start.x * scale
            y1 = entity.dxf.start.y * scale
            x2 = entity.dxf.end.x   * scale
            y2 = entity.dxf.end.y   * scale
            length = round(math.hypot(x2 - x1, y2 - y1), 4)
            lines.append({"length": length})

        elif etype in ("TEXT", "MTEXT"):
            try:
                txt = entity.dxf.text.strip()
                if txt:
                    texts.append(txt)
            except Exception:
                pass

    return {
        "file":    os.path.basename(dxf_path),
        "circles": sorted(circles, key=lambda c: c["diameter"]),
        "arcs":    sorted(arcs,    key=lambda a: a["radius"]),
        "lines":   sorted(lines,   key=lambda l: l["length"]),
        "texts":   texts,
    }


# =====================================================
# STEP 2 — LOAD ALL DXF FILES
# =====================================================

dxf_files = [
    os.path.join(DXF_DIR, f)
    for f in sorted(os.listdir(DXF_DIR))
    if f.endswith(".dxf")
]

all_dxf = [extract_dxf_features(p) for p in dxf_files]

# Merge all DXF circles and arcs into one pool
merged_circles = []
merged_arcs    = []
for d in all_dxf:
    merged_circles.extend(d["circles"])
    merged_arcs.extend(d["arcs"])

# Unique diameters from DXF
dxf_diameters = sorted(set(round(c["diameter"], 3) for c in merged_circles))
dxf_radii     = sorted(set(round(a["radius"],   3) for a in merged_arcs))
dxf_all_radii = sorted(set(
    [round(c["radius"], 3) for c in merged_circles] +
    [round(a["radius"], 3) for a in merged_arcs]
))

# =====================================================
# STEP 3 — LOAD OCR PARSED JSON
# =====================================================

with open(OCR_JSON, encoding="utf-8") as f:
    ocr = json.load(f)

# =====================================================
# STEP 4 — MATCHING HELPERS
# =====================================================

def find_match(value, candidates, tol=MATCH_TOL):
    """Return closest candidate within tolerance, or None."""
    best, best_err = None, tol + 1
    for c in candidates:
        err = abs(c - value)
        if err < best_err:
            best_err = err
            best = c
    if best_err <= tol:
        return best, round(best_err, 4)
    return None, None


def match_label(err):
    if err is None:
        return "NO MATCH"
    if err == 0.0:
        return "EXACT"
    if err <= 0.05:
        return "NEAR EXACT"
    return f"CLOSE ({err} mm off)"


# =====================================================
# STEP 5 — COMPARE HOLES
# =====================================================

hole_results = []

# From OCR: simple holes
for h in ocr.get("holes", []):
    dia = h["diameter"]
    matched, err = find_match(dia, dxf_diameters)
    hole_results.append({
        "source":          "OCR hole",
        "ocr_diameter":    dia,
        "dxf_diameter":    matched,
        "error_mm":        err,
        "status":          match_label(err),
    })

# From OCR: hole patterns
for hp in ocr.get("hole_patterns", []):
    dia = hp["diameter"]
    matched, err = find_match(dia, dxf_diameters)
    hole_results.append({
        "source":          f"OCR hole_pattern ({hp['count']}x)",
        "ocr_diameter":    dia,
        "dxf_diameter":    matched,
        "error_mm":        err,
        "status":          match_label(err),
    })

# =====================================================
# STEP 6 — COMPARE RADII
# =====================================================

radius_results = []

for r in ocr.get("radii", []):
    rad = r["radius"]
    matched, err = find_match(rad, dxf_all_radii)
    radius_results.append({
        "ocr_radius":  rad,
        "dxf_radius":  matched,
        "error_mm":    err,
        "status":      match_label(err),
    })

# =====================================================
# STEP 7 — COMPARE PCD
# =====================================================

pcd_results = []

for p in ocr.get("pcd_features", []):
    dia = p["diameter"]
    # PCD diameter → radius in DXF arcs
    pcd_radius = dia / 2
    matched_r, err_r = find_match(pcd_radius, dxf_all_radii)
    # Also check direct diameter match in circles
    matched_d, err_d = find_match(dia, dxf_diameters)

    if err_r is not None and (err_d is None or err_r <= err_d):
        pcd_results.append({
            "ocr_pcd_diameter": dia,
            "match_type":       "arc radius",
            "dxf_value":        matched_r,
            "dxf_diameter_eq":  round(matched_r * 2, 4) if matched_r else None,
            "error_mm":         err_r,
            "status":           match_label(err_r),
        })
    elif err_d is not None:
        pcd_results.append({
            "ocr_pcd_diameter": dia,
            "match_type":       "circle diameter",
            "dxf_value":        matched_d,
            "dxf_diameter_eq":  matched_d,
            "error_mm":         err_d,
            "status":           match_label(err_d),
        })
    else:
        pcd_results.append({
            "ocr_pcd_diameter": dia,
            "match_type":       "none",
            "dxf_value":        None,
            "dxf_diameter_eq":  None,
            "error_mm":         None,
            "status":           "NO MATCH",
        })

# =====================================================
# STEP 8 — COMPARE DIMENSIONS (general)
# =====================================================

ocr_dims = ocr.get("dimensions", [])
# All DXF numeric values: diameters + radii*2 + line lengths
dxf_all_dims = sorted(set(
    dxf_diameters +
    [round(r * 2, 4) for r in dxf_all_radii] +
    [round(l["length"], 3) for d in all_dxf for l in d["lines"]]
))

dim_results = []
for dim in ocr_dims:
    if dim < 0.1:   # skip pure tolerance values
        continue
    matched, err = find_match(dim, dxf_all_dims, tol=0.5)
    dim_results.append({
        "ocr_dim":  dim,
        "dxf_dim":  matched,
        "error_mm": err,
        "status":   match_label(err),
    })

# =====================================================
# STEP 9 — SUMMARY STATS
# =====================================================

def stats(results):
    total   = len(results)
    matched = sum(1 for r in results if r["status"] != "NO MATCH")
    missed  = total - matched
    pct     = matched / total * 100 if total else 0
    return total, matched, missed, round(pct, 1)

h_total, h_match, h_miss, h_pct   = stats(hole_results)
r_total, r_match, r_miss, r_pct   = stats(radius_results)
p_total, p_match, p_miss, p_pct   = stats(pcd_results)
d_total, d_match, d_miss, d_pct   = stats(dim_results)

overall_total   = h_total + r_total + p_total + d_total
overall_matched = h_match + r_match + p_match + d_match
overall_pct     = overall_matched / overall_total * 100 if overall_total else 0

# =====================================================
# STEP 10 — CONSOLE REPORT
# =====================================================

SEP = "=" * 65

print(f"\n{SEP}")
print("DXF FILES LOADED")
print(SEP)
for d in all_dxf:
    print(f"  {d['file']:30s}  circles={len(d['circles'])}  arcs={len(d['arcs'])}  lines={len(d['lines'])}")

print(f"\n  Merged unique circle diameters : {dxf_diameters}")
print(f"  Merged unique arc radii        : {dxf_radii}")

print(f"\n{SEP}")
print("HOLE COMPARISON  (OCR ↔ DXF)")
print(SEP)
for r in hole_results:
    print(f"  {r['source']:30s}  OCR={r['ocr_diameter']:7.3f}  DXF={str(r['dxf_diameter']):7}  err={str(r['error_mm']):6}  [{r['status']}]")
print(f"\n  Match rate: {h_match}/{h_total} ({h_pct}%)")

print(f"\n{SEP}")
print("RADIUS COMPARISON  (OCR ↔ DXF)")
print(SEP)
for r in radius_results:
    print(f"  OCR R={r['ocr_radius']:6.3f}  DXF R={str(r['dxf_radius']):7}  err={str(r['error_mm']):6}  [{r['status']}]")
print(f"\n  Match rate: {r_match}/{r_total} ({r_pct}%)")

print(f"\n{SEP}")
print("PCD COMPARISON  (OCR ↔ DXF)")
print(SEP)
for r in pcd_results:
    print(f"  OCR PCD={r['ocr_pcd_diameter']:6.1f}  via {r['match_type']:15s}  DXF={str(r['dxf_value']):7}  err={str(r['error_mm']):6}  [{r['status']}]")
print(f"\n  Match rate: {p_match}/{p_total} ({p_pct}%)")

print(f"\n{SEP}")
print("DIMENSION COMPARISON  (OCR ↔ DXF)  [tol=0.5mm]")
print(SEP)
for r in dim_results:
    flag = "" if r["status"] != "NO MATCH" else "  <-- not in DXF"
    print(f"  OCR={r['ocr_dim']:8.4f}  DXF={str(r['dxf_dim']):8}  err={str(r['error_mm']):6}  [{r['status']}]{flag}")
print(f"\n  Match rate: {d_match}/{d_total} ({d_pct}%)")

print(f"\n{SEP}")
print("OVERALL SUMMARY")
print(SEP)
print(f"  Holes      : {h_match}/{h_total} matched  ({h_pct}%)")
print(f"  Radii      : {r_match}/{r_total} matched  ({r_pct}%)")
print(f"  PCD        : {p_match}/{p_total} matched  ({p_pct}%)")
print(f"  Dimensions : {d_match}/{d_total} matched  ({d_pct}%)")
print(f"\n  TOTAL      : {overall_matched}/{overall_total} matched  ({overall_pct:.1f}%)")
print(SEP)

# =====================================================
# STEP 11 — SAVE JSON
# =====================================================

output = {
    "dxf_files":          [d["file"] for d in all_dxf],
    "dxf_circle_diameters": dxf_diameters,
    "dxf_arc_radii":        dxf_radii,
    "hole_comparison":      hole_results,
    "radius_comparison":    radius_results,
    "pcd_comparison":       pcd_results,
    "dimension_comparison": dim_results,
    "summary": {
        "holes":      {"matched": h_match, "total": h_total, "pct": h_pct},
        "radii":      {"matched": r_match, "total": r_total, "pct": r_pct},
        "pcd":        {"matched": p_match, "total": p_total, "pct": p_pct},
        "dimensions": {"matched": d_match, "total": d_total, "pct": d_pct},
        "overall":    {"matched": overall_matched, "total": overall_total,
                       "pct": round(overall_pct, 1)},
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=4)

print(f"\n[OK] Saved: {OUTPUT_JSON}")
print("[DONE] DXF vs OCR comparison complete.\n")

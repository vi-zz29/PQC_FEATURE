import ezdxf
import json
import os
import math

BASE        = os.path.dirname(os.path.abspath(__file__))
FEATURES    = os.path.dirname(BASE)
DXF_DIR     = os.path.join(FEATURES, "dxf")
OCR_JSON    = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")
OUTPUT_JSON = os.path.join(FEATURES, "outputs", "dxf_ocr_comparison.json")

MATCH_TOL = 0.5


def extract_dxf_features(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

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


dxf_files = [
    os.path.join(DXF_DIR, f)
    for f in sorted(os.listdir(DXF_DIR))
    if f.endswith(".dxf")
]

all_dxf = [extract_dxf_features(p) for p in dxf_files]

merged_circles = []
merged_arcs    = []
for d in all_dxf:
    merged_circles.extend(d["circles"])
    merged_arcs.extend(d["arcs"])

dxf_diameters = sorted(set(round(c["diameter"], 3) for c in merged_circles))
dxf_radii     = sorted(set(round(a["radius"],   3) for a in merged_arcs))
dxf_all_radii = sorted(set(
    [round(c["radius"], 3) for c in merged_circles] +
    [round(a["radius"], 3) for a in merged_arcs]
))

with open(OCR_JSON, encoding="utf-8") as f:
    ocr = json.load(f)


def find_match(value, candidates, tol=MATCH_TOL):
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


hole_results = []

for h in ocr.get("holes", []):
    dia = h["diameter"]
    matched, err = find_match(dia, dxf_diameters)
    hole_results.append({
        "source":       "OCR hole",
        "ocr_diameter": dia,
        "dxf_diameter": matched,
        "error_mm":     err,
        "status":       match_label(err),
    })

for hp in ocr.get("hole_patterns", []):
    dia = hp["diameter"]
    matched, err = find_match(dia, dxf_diameters)
    hole_results.append({
        "source":       f"OCR hole_pattern ({hp['count']}x)",
        "ocr_diameter": dia,
        "dxf_diameter": matched,
        "error_mm":     err,
        "status":       match_label(err),
    })

radius_results = []

for r in ocr.get("radii", []):
    rad = r["radius"]
    matched, err = find_match(rad, dxf_all_radii)
    radius_results.append({
        "ocr_radius": rad,
        "dxf_radius": matched,
        "error_mm":   err,
        "status":     match_label(err),
    })

pcd_results = []

for p in ocr.get("pcd_features", []):
    dia = p["diameter"]
    pcd_radius = dia / 2
    matched_r, err_r = find_match(pcd_radius, dxf_all_radii)
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

ocr_dims = ocr.get("dimensions", [])
dxf_all_dims = sorted(set(
    dxf_diameters +
    [round(r * 2, 4) for r in dxf_all_radii] +
    [round(l["length"], 3) for d in all_dxf for l in d["lines"]]
))

dim_results = []
for dim in ocr_dims:
    if dim < 0.1:
        continue
    matched, err = find_match(dim, dxf_all_dims, tol=1.0)
    dim_results.append({
        "ocr_dim":  dim,
        "dxf_dim":  matched,
        "error_mm": err,
        "status":   match_label(err),
    })


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

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=4)

detected = ", ".join([
    f"holes {h_match}/{h_total}",
    f"radii {r_match}/{r_total}",
    f"pcd {p_match}/{p_total}",
    f"dims {d_match}/{d_total}",
])
print(f"[dxf_ocr_compare] {overall_matched}/{overall_total} features matched ({overall_pct:.0f}%) — {detected} → {OUTPUT_JSON}")

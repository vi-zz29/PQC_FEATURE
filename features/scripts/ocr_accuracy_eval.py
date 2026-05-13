import json
import re
import os

GROUND_TRUTH_N001 = {
    "part_number":      "1-02-1065-N001",
    "surface_finish":   "ELECTROLESS NICKEL",
    "surface_area":     "38726.48",
    "thread":           "M4x0.7",
    "thread_tol":       "6H",
    "thread_depth":     "DEPTH 12.0",
    "hole_45_thru":     "4.5 THRU",
    "hole_52":          "5.2",
    "pcd_84":           "84.0 PCD",
    "pcd_96":           "96.0 PCD",
    "slot_10":          "10.0 SLOTS",
    "radius_R3":        "R3.0",
    "radius_R04":       "R0.4",
    "radius_R02":       "R0.2",
    "chamfer_02x45":    "0.2 x 45",
    "chamfer_05x45":    "0.5 x 45",
    "fit_H8":           "H8",
    "fit_deviation":    "+0.046",
    "tol_5401":         "54.01",
    "tol_54054":        "54.054",
    "dim_2375":         "23.75",
    "dim_855":          "85.5",
    "dim_800":          "80.0",
    "dim_830":          "83.0",
    "dim_960":          "96.0",
    "dim_160":          "16.0",
    "dim_180":          "18.0",
    "ra_value":         "1.6 Ra",
    "units":            "dimensions mm",
    "rohs":             "ROHS",
    "equi_spaced":      "EQUI-SPACED",
    "dim_540":          "54.0",
    "dim_580":          "58.0",
    "dim_600":          "60.0",
    "dim_2310":         "23.10",
    "dim_1120":         "11.20",
    "dim_1105":         "11.05",
    "dim_4105":         "41.05",
    "section_aa":       "SECTION A",
    "detail_b":         "DETAIL B",
    "depth_111":        "DEPTH 11.1",
    "angle_42":         "42",
    "angle_15":         "15",
    "angle_28":         "28",
    "brake_mag":        "BRAKE MAG",
    "confidential":     "CONFIDENTIAL",
    "3rd_angle":        "3rd ANGLE PROJECTION",
}

GROUND_TRUTH_BLUEPRINT = {
    "part_number":          "BEV820-BM-01",
    "revision":             "REV: B",
    "material":             "AISI 304",
    "surface_finish":       "ELECTROPOLISH",
    "astm_spec":            "ASTM B912",
    "part_title":           "BACKREST MOUNTING BLOCK",
    "unc_thread_1024":      "10-24 UNC",
    "unc_thread_3816":      "3/8-16 UNC",
    "thread_depth_059":     "0.59",
    "thread_depth_038":     "0.38",
    "thread_thru_bore":     "THROUGH TO BORE",
    "hole_015":             "0.150",
    "hole_pattern_2x":      "2X",
    "hole_024_x90":         "0.24 X 90",
    "hole_313_thru":        "0.313",
    "hole_043_x90":         "0.43 X 90",
    "hole_088":             "0.88",
    "dowel_dia":            "0.375",
    "dowel_len":            "1.00",
    "dowel_ream_lower":     "0.3735",
    "dowel_ream_upper":     "0.3745",
    "radius_044":           "RO.44",
    "radius_025":           "RO.25",
    "radius_013":           "RO.13",
    "radius_0281":          "RO.281",
    "radius_0469":          "RO.469",
    "radius_0250":          "RO.250",
    "chamfer_003x45":       "0.03 X 45",
    "dim_0922":             "0.922",
    "dim_0373":             "0.373",
    "dim_0750":             "0.750",
    "dim_0755":             "0.755",
    "dim_1005":             "1.005",
    "dim_1003":             "1.003",
    "dim_0502":             "0.502",
    "dim_0503":             "0.503",
    "dim_1188":             "1.188",
    "dim_0719":             "0.719",
    "dim_325":              "3.25",
    "dim_1250":             "1.250",
    "dim_1125":             "1.125",
    "dim_163":              "1.63",
    "dim_0625":             "0.625",
    "dim_1531":             "1.531",
    "dim_0375":             "0.375",
    "dim_100":              "1.00",
    "tol_plus_0005":        "+0.005",
    "tol_plus_0010":        "+0.010",
    "near_side":            "NEAR SIDE",
    "confidential":         "CONFIDENTIALITY NOTICE",
    "electropolish_note":   "NOTE #2",
    "dowel_note":           "NOTE #1",
    "alternate_method":     "ALTERNATE METHOD",
}


def evaluate_ocr(ocr_text, ground_truth, label):
    text_lower = ocr_text.lower()
    found, missing, corrupted = [], [], []

    for key, token in ground_truth.items():
        token_lower = token.lower()
        if token_lower in text_lower:
            found.append((key, token))
        else:
            fuzzy = token_lower.replace("0", "[0o]").replace("l", "[l1]").replace("i", "[i1]")
            if re.search(fuzzy, text_lower):
                corrupted.append((key, token, "fuzzy match"))
            else:
                missing.append((key, token))

    total       = len(ground_truth)
    exact       = len(found)
    fuzzy_count = len(corrupted)
    missed      = len(missing)
    exact_pct   = exact / total * 100
    partial_pct = (exact + fuzzy_count) / total * 100

    miss_tokens = ", ".join(t for _, t in missing[:5])
    miss_extra  = f" +{missed - 5} more" if missed > 5 else ""
    print(f"[{label}] exact={exact}/{total} ({exact_pct:.0f}%)  partial={exact+fuzzy_count}/{total} ({partial_pct:.0f}%)  missing={missed}" +
          (f"  [{miss_tokens}{miss_extra}]" if missing else ""))

    return {
        "file": label,
        "total": total,
        "exact": exact,
        "fuzzy": fuzzy_count,
        "missing": missed,
        "exact_accuracy_pct": round(exact_pct, 1),
        "partial_accuracy_pct": round(partial_pct, 1),
    }


BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.dirname(BASE)

results = []

ocr_path_1    = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_ocr.txt")
parsed_path_1 = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")

if os.path.exists(ocr_path_1):
    with open(ocr_path_1, encoding="utf-8") as f:
        ocr1 = f.read()
    results.append(evaluate_ocr(ocr1, GROUND_TRUTH_N001, "1-02-1065-N001"))
else:
    print(f"[SKIP] Not found: {ocr_path_1}")

ocr_path_2    = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_ocr.txt")
parsed_path_2 = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_parsed.json")

if os.path.exists(ocr_path_2):
    with open(ocr_path_2, encoding="utf-8") as f:
        ocr2 = f.read()
    results.append(evaluate_ocr(ocr2, GROUND_TRUTH_BLUEPRINT, "blueprint_page (BEV820-BM-01)"))
else:
    print(f"[SKIP] Not found: {ocr_path_2}")

if results:
    avg_exact   = sum(r["exact_accuracy_pct"]   for r in results) / len(results)
    avg_partial = sum(r["partial_accuracy_pct"] for r in results) / len(results)
    print(f"[ocr_accuracy_eval] avg exact={avg_exact:.1f}%  avg partial={avg_partial:.1f}%  ({len(results)} files)")

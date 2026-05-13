"""
OCR Accuracy Evaluation for Phase 1
Compares raw OCR text against known ground truth values
for both blueprint files.
"""

import json
import re
import os

# =====================================================
# GROUND TRUTH - manually verified from the drawings
# =====================================================

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
    # --- Identity ---
    "part_number":          "BEV820-BM-01",
    "revision":             "REV: B",
    "material":             "AISI 304",
    "surface_finish":       "ELECTROPOLISH",
    "astm_spec":            "ASTM B912",
    "part_title":           "BACKREST MOUNTING BLOCK",

    # --- Threads ---
    "unc_thread_1024":      "10-24 UNC",
    "unc_thread_3816":      "3/8-16 UNC",
    "thread_depth_059":     "0.59",
    "thread_depth_038":     "0.38",
    "thread_thru_bore":     "THROUGH TO BORE",

    # --- Holes ---
    "hole_015":             "0.150",
    "hole_pattern_2x":      "2X",
    "hole_024_x90":         "0.24 X 90",
    "hole_313_thru":        "0.313",
    "hole_043_x90":         "0.43 X 90",
    "hole_088":             "0.88",

    # --- Dowel pin ---
    "dowel_dia":            "0.375",
    "dowel_len":            "1.00",
    "dowel_ream_lower":     "0.3735",
    "dowel_ream_upper":     "0.3745",

    # --- Radii ---
    "radius_044":           "RO.44",
    "radius_025":           "RO.25",
    "radius_013":           "RO.13",
    "radius_0281":          "RO.281",
    "radius_0469":          "RO.469",
    "radius_0250":          "RO.250",

    # --- Chamfer ---
    "chamfer_003x45":       "0.03 X 45",

    # --- Key dimensions ---
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

    # --- Tolerances ---
    "tol_plus_0005":        "+0.005",
    "tol_plus_0010":        "+0.010",

    # --- Notes ---
    "near_side":            "NEAR SIDE",
    "confidential":         "CONFIDENTIALITY NOTICE",
    "electropolish_note":   "NOTE #2",
    "dowel_note":           "NOTE #1",
    "alternate_method":     "ALTERNATE METHOD",
}

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_ocr(ocr_text, ground_truth, label):
    text_lower = ocr_text.lower()

    found = []
    missing = []
    corrupted = []  # present but with OCR errors nearby

    for key, token in ground_truth.items():
        token_lower = token.lower()
        if token_lower in text_lower:
            found.append((key, token))
        else:
            # Check if a fuzzy version exists (common OCR substitutions)
            fuzzy = (token_lower
                     .replace("0", "[0o]")
                     .replace("l", "[l1]")
                     .replace("i", "[i1]"))
            if re.search(fuzzy, text_lower):
                corrupted.append((key, token, "fuzzy match"))
            else:
                missing.append((key, token))

    total = len(ground_truth)
    exact = len(found)
    fuzzy_count = len(corrupted)
    missed = len(missing)

    exact_pct = exact / total * 100
    partial_pct = (exact + fuzzy_count) / total * 100

    print(f"\n{'='*60}")
    print(f"FILE: {label}")
    print(f"{'='*60}")
    print(f"  Total ground truth tokens : {total}")
    print(f"  Exact matches             : {exact}  ({exact_pct:.1f}%)")
    print(f"  Fuzzy/partial matches     : {fuzzy_count}")
    print(f"  Exact + fuzzy             : {exact + fuzzy_count}  ({partial_pct:.1f}%)")
    print(f"  Completely missing        : {missed}  ({missed/total*100:.1f}%)")

    if corrupted:
        print(f"\n  --- FUZZY MATCHES (OCR noise but recoverable) ---")
        for key, token, reason in corrupted:
            print(f"    ~ {key:25s}  expected: '{token}'")

    if missing:
        print(f"\n  --- MISSING TOKENS (not found at all) ---")
        for key, token in missing:
            print(f"    x {key:25s}  expected: '{token}'")

    return {
        "file": label,
        "total": total,
        "exact": exact,
        "fuzzy": fuzzy_count,
        "missing": missed,
        "exact_accuracy_pct": round(exact_pct, 1),
        "partial_accuracy_pct": round(partial_pct, 1),
    }


# =====================================================
# CHARACTER-LEVEL OCR QUALITY (Tesseract confidence proxy)
# =====================================================

def text_quality_stats(ocr_text, label):
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    total_chars = len(ocr_text)
    total_words = len(ocr_text.split())

    # Count garbled tokens: strings with mixed symbols/letters
    garbled = len(re.findall(r'[A-Za-z]{1,3}[^A-Za-z0-9\s.,/\-°×xØ$]{2,}', ocr_text))

    # Count clean numeric tokens
    clean_nums = len(re.findall(r'\b\d+(?:\.\d+)?\b', ocr_text))

    # Count Ø symbol occurrences (diameter callouts)
    diameter_symbols = ocr_text.count('Ø')

    # Count lines that look like pure noise (< 3 real chars)
    noise_lines = sum(1 for l in lines if len(re.sub(r'[^A-Za-z0-9]', '', l)) < 3)
    noise_pct = noise_lines / len(lines) * 100 if lines else 0

    print(f"\n  --- TEXT QUALITY STATS: {label} ---")
    print(f"    Total characters    : {total_chars}")
    print(f"    Total words         : {total_words}")
    print(f"    Total lines         : {len(lines)}")
    print(f"    Noise lines         : {noise_lines} ({noise_pct:.1f}%)")
    print(f"    Clean numeric tokens: {clean_nums}")
    print(f"    Diameter (Ø) symbols: {diameter_symbols}")
    print(f"    Garbled token count : {garbled}")


# =====================================================
# PARSED JSON COMPLETENESS CHECK
# =====================================================

def check_parsed_completeness(parsed_path, label):
    if not os.path.exists(parsed_path):
        print(f"\n  [SKIP] Parsed JSON not found: {parsed_path}")
        return

    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = [
        "part_number", "revision", "material", "surface_finish",
        "units", "surface_area", "metric_threads", "unc_threads",
        "pcd_features", "slot_features", "hole_patterns", "holes",
        "angles", "radii", "chamfers", "fit_tolerances",
        "tolerance_groups", "dimensions", "depths", "classified_dimensions"
    ]

    print(f"\n  --- PARSED JSON COMPLETENESS: {label} ---")
    filled = 0
    for key in expected_keys:
        val = data.get(key)
        is_filled = val is not None and val != [] and val != {}
        status = "OK " if is_filled else "---"
        if is_filled:
            filled += 1
        count = f"({len(val)} items)" if isinstance(val, list) else ""
        print(f"    [{status}] {key:30s} {count}")

    pct = filled / len(expected_keys) * 100
    print(f"\n    Completeness: {filled}/{len(expected_keys)} fields filled ({pct:.1f}%)")


# =====================================================
# MAIN
# =====================================================

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.dirname(BASE)
ROOT     = os.path.dirname(FEATURES)

results = []

# --- FILE 1: 1-02-1065-N001 ---
ocr_path_1    = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_ocr.txt")
parsed_path_1 = os.path.join(FEATURES, "samples", "sample_01_N001", "1-02-1065-N001_parsed.json")

if os.path.exists(ocr_path_1):
    with open(ocr_path_1, encoding="utf-8") as f:
        ocr1 = f.read()
    r1 = evaluate_ocr(ocr1, GROUND_TRUTH_N001, "1-02-1065-N001")
    text_quality_stats(ocr1, "1-02-1065-N001")
    check_parsed_completeness(parsed_path_1, "1-02-1065-N001")
    results.append(r1)
else:
    print(f"[SKIP] Not found: {ocr_path_1}")

# --- FILE 2: blueprint_page ---
ocr_path_2    = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_ocr.txt")
parsed_path_2 = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_parsed.json")

if os.path.exists(ocr_path_2):
    with open(ocr_path_2, encoding="utf-8") as f:
        ocr2 = f.read()
    r2 = evaluate_ocr(ocr2, GROUND_TRUTH_BLUEPRINT, "blueprint_page (BEV820-BM-01)")
    text_quality_stats(ocr2, "blueprint_page")
    check_parsed_completeness(parsed_path_2, "blueprint_page")
    results.append(r2)
else:
    print(f"[SKIP] Not found: {ocr_path_2}")

# =====================================================
# OVERALL SUMMARY
# =====================================================

print(f"\n{'='*60}")
print("OVERALL ACCURACY SUMMARY")
print(f"{'='*60}")
for r in results:
    print(f"\n  {r['file']}")
    print(f"    Exact match accuracy   : {r['exact_accuracy_pct']}%")
    print(f"    Partial match accuracy : {r['partial_accuracy_pct']}%")
    print(f"    Missing tokens         : {r['missing']}/{r['total']}")

if results:
    avg_exact = sum(r['exact_accuracy_pct'] for r in results) / len(results)
    avg_partial = sum(r['partial_accuracy_pct'] for r in results) / len(results)
    print(f"\n  AVERAGE EXACT ACCURACY   : {avg_exact:.1f}%")
    print(f"  AVERAGE PARTIAL ACCURACY : {avg_partial:.1f}%")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")

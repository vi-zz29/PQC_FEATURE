"""
Detailed OCR Accuracy Report — Sample 02 (BEV820-BM-01)
Checks every engineering feature category individually.
"""

import json
import re
import os

BASE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.dirname(BASE)

OCR_TXT    = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_ocr.txt")
PARSED_JSON = os.path.join(FEATURES, "samples", "sample_02_BEV820", "blueprint_page_parsed.json")

with open(OCR_TXT, encoding="utf-8") as f:
    raw = f.read()

with open(PARSED_JSON, encoding="utf-8") as f:
    parsed = json.load(f)

text = raw.upper()
SEP  = "=" * 65
SEP2 = "-" * 65

def found(token):
    return token.upper() in text

def found_re(pattern):
    return bool(re.search(pattern, text, re.IGNORECASE))

def tick(val):
    return "PASS" if val else "FAIL"

results = {}   # category → {pass, fail, details}

def section(name):
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)

def row(label, check, ocr_val, parsed_val, note=""):
    status = tick(check)
    mark   = "✓" if check else "✗"
    print(f"  [{mark}] {label:35s}  OCR={str(ocr_val):20s}  Parsed={str(parsed_val):20s}  {note}")
    return check

# ─────────────────────────────────────────────
# 1. IDENTITY
# ─────────────────────────────────────────────
section("1. IDENTITY")
checks = []
checks.append(row("Part Number",
    found("BEV820-BM-01"),
    "BEV820-BM-01" if found("BEV820-BM-01") else "NOT FOUND",
    parsed.get("part_number")))

checks.append(row("Revision B",
    found("REV: B") or found("REV B") or found("REV:B"),
    "REV: B" if found("REV: B") else "NOT FOUND",
    parsed.get("revision")))

checks.append(row("Material AISI 304",
    found("AISI 304"),
    "AISI 304" if found("AISI 304") else "NOT FOUND",
    parsed.get("material")))

checks.append(row("Surface Finish ELECTROPOLISH",
    found("ELECTROPOLISH"),
    "ELECTROPOLISH" if found("ELECTROPOLISH") else "NOT FOUND",
    parsed.get("surface_finish")))

checks.append(row("ASTM B912 spec",
    found("ASTM B912"),
    "ASTM B912" if found("ASTM B912") else "NOT FOUND",
    "in notes"))

checks.append(row("Units (inch drawing)",
    parsed.get("units") == "inch",
    "inferred from dims",
    parsed.get("units")))

checks.append(row("Part Title",
    found("BACKREST MOUNTING BLOCK"),
    "BACKREST MOUNTING BLOCK" if found("BACKREST MOUNTING BLOCK") else "NOT FOUND",
    "in title block"))

results["identity"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 2. THREADS
# ─────────────────────────────────────────────
section("2. THREADS")
checks = []

unc_threads = [t["thread"] for t in parsed.get("unc_threads", [])]

checks.append(row("10-24 UNC-2B present in OCR",
    found("10-24 UNC"),
    "10-24 UNC - 2B" if found("10-24 UNC") else "NOT FOUND",
    next((t for t in unc_threads if "10-24" in t), "MISSING")))

checks.append(row("10-24 UNC extracted to JSON",
    any("10-24" in t for t in unc_threads),
    "10-24 UNC - 2B",
    next((t for t in unc_threads if "10-24" in t), "MISSING")))

checks.append(row("3/8-16 UNC-2B present in OCR",
    found("3/8-16 UNC"),
    "3/8-16 UNC - 2B" if found("3/8-16 UNC") else "NOT FOUND",
    next((t for t in unc_threads if "3/8" in t or "38" in t), "MISSING")))

checks.append(row("3/8-16 UNC extracted to JSON",
    any("3/8" in t or "38" in t for t in unc_threads),
    "3/8-16 UNC - 2B",
    next((t for t in unc_threads if "3/8" in t or "38" in t), "MISSING")))

checks.append(row("Thread depth 0.59 in OCR",
    found("0.59"),
    "0.59" if found("0.59") else "NOT FOUND",
    "depth callout"))

checks.append(row("Thread depth 0.38 in OCR",
    found("0.38"),
    "0.38" if found("0.38") else "NOT FOUND",
    "depth callout"))

checks.append(row("THROUGH TO BORE callout",
    found("THROUGH TO BORE"),
    "THROUGH TO BORE" if found("THROUGH TO BORE") else "NOT FOUND",
    "thread note"))

results["threads"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 3. HOLES
# ─────────────────────────────────────────────
section("3. HOLES & HOLE PATTERNS")
checks = []

holes        = parsed.get("holes", [])
hole_pats    = parsed.get("hole_patterns", [])
hole_dias    = [h["diameter"] for h in holes]
pat_dias     = [h["diameter"] for h in hole_pats]

checks.append(row("Ø0.150 in OCR",
    found("0.150") or found("0.15"),
    "0.150" if found("0.150") else "NOT FOUND",
    f"{0.15} in holes" if 0.15 in hole_dias else "MISSING from JSON"))

checks.append(row("2X Ø0.150 pattern in OCR",
    found("2X") and found("0.150"),
    "2X Ø0.150" if found("2X") else "NOT FOUND",
    f"count=2 dia=0.15" if any(h.get("count")==2 for h in hole_pats) else "MISSING"))

checks.append(row("Ø0.24 X 90° in OCR",
    found("0.24 X 90") or found("0.24X90"),
    "0.24 X 90°" if found("0.24") else "NOT FOUND",
    f"{0.24} in holes" if 0.24 in hole_dias else "MISSING from JSON"))

checks.append(row("Ø0.313 THROUGH TO BORE",
    found("0.313"),
    "0.313" if found("0.313") else "NOT FOUND",
    f"{0.313} in holes" if 0.313 in hole_dias else "MISSING from JSON"))

checks.append(row("Ø0.43 X 90° in OCR",
    found("0.43 X 90") or found("0.43X90"),
    "0.43 X 90°" if found("0.43") else "NOT FOUND",
    f"{0.43} in holes" if 0.43 in hole_dias else "MISSING from JSON"))

checks.append(row("Ø0.88 in OCR",
    found("0.88"),
    "0.88" if found("0.88") else "NOT FOUND",
    f"{0.88} in holes" if 0.88 in hole_dias else "MISSING from JSON"))

results["holes"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 4. DOWEL PIN
# ─────────────────────────────────────────────
section("4. DOWEL PIN FEATURES")
checks = []

dims = parsed.get("dimensions", [])

checks.append(row("Ream Ø0.3735 in OCR",
    found("0.3735"),
    "0.3735" if found("0.3735") else "NOT FOUND",
    f"{0.3735} in dims" if 0.3735 in dims else "MISSING"))

checks.append(row("Ream Ø0.3745 in OCR",
    found("0.3745"),
    "0.3745" if found("0.3745") else "NOT FOUND",
    f"{0.3745} in dims" if 0.3745 in dims else "MISSING"))

checks.append(row("Tolerance group 0.3735/0.3745",
    any(abs(t.get("lower_limit",0)-0.3735)<0.001 for t in parsed.get("tolerance_groups",[])),
    "0.3735/0.3745",
    str(parsed.get("tolerance_groups",[]))))

checks.append(row("Dowel Ø0.375 in OCR",
    found("0.375"),
    "0.375" if found("0.375") else "NOT FOUND",
    f"{0.375} in dims" if 0.375 in dims else "MISSING"))

checks.append(row("Dowel length 1.00 in OCR",
    found("1.00"),
    "1.00" if found("1.00") else "NOT FOUND",
    f"{1.0} in dims" if 1.0 in dims else "MISSING"))

results["dowel"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 5. RADII
# ─────────────────────────────────────────────
section("5. RADII")
checks = []

radii_vals = [r["radius"] for r in parsed.get("radii", [])]

expected_radii = [
    ("R0.44",  "RO.44",  0.44),
    ("R0.25",  "RO.25",  0.25),
    ("R0.13",  "RO.13",  0.13),
    ("R0.281", "RO.281", 0.281),
    ("R0.469", "RO.469", 0.469),
    ("R0.250", "RO.250", 0.25),
]

for label, ocr_token, val in expected_radii:
    in_ocr    = found(ocr_token) or found(f"R{val}")
    in_parsed = any(abs(r - val) < 0.01 for r in radii_vals)
    checks.append(row(f"{label} in OCR",
        in_ocr,
        ocr_token if in_ocr else "NOT FOUND",
        f"{val} extracted" if in_parsed else "MISSING from JSON"))

results["radii"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 6. CHAMFERS
# ─────────────────────────────────────────────
section("6. CHAMFERS")
checks = []

chamfers = parsed.get("chamfers", [])

checks.append(row("0.03 X 45° in OCR",
    found("0.03 X 45") or found("0.03X45"),
    "0.03 X 45°" if found("0.03") else "NOT FOUND",
    str([c for c in chamfers if c.get("size")==0.03])))

checks.append(row("0.03 X 45° extracted",
    any(c.get("size")==0.03 and c.get("angle")==45 for c in chamfers),
    "0.03 x 45",
    str(chamfers)))

results["chamfers"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 7. KEY DIMENSIONS
# ─────────────────────────────────────────────
section("7. KEY DIMENSIONS")
checks = []

key_dims = [
    ("0.922", 0.922), ("0.373", 0.373), ("0.750", 0.75),
    ("0.755", 0.755), ("1.005", 1.005), ("1.003", 1.003),
    ("0.502", 0.502), ("0.503", 0.503), ("1.188", 1.188),
    ("0.719", 0.719), ("3.25",  3.25),  ("1.250", 1.25),
    ("1.125", 1.125), ("1.63",  1.63),  ("0.625", 0.625),
    ("1.531", 1.531), ("0.98",  0.98),  ("0.81",  0.81),
]

for label, val in key_dims:
    in_ocr    = found(label)
    in_parsed = val in dims or any(abs(d - val) < 0.001 for d in dims)
    checks.append(row(f"Dim {label}",
        in_ocr and in_parsed,
        label if in_ocr else "NOT IN OCR",
        f"extracted" if in_parsed else "MISSING"))

results["dimensions"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 8. TOLERANCES
# ─────────────────────────────────────────────
section("8. TOLERANCES")
checks = []

checks.append(row("+0.005 tolerance in OCR",
    found("+0.005") or found("0.005"),
    "+0.005" if found("+0.005") else "NOT FOUND",
    "in tolerance block"))

checks.append(row("+0.010 tolerance in OCR",
    found("+0.010") or found("0.010"),
    "+0.010" if found("+0.010") else "NOT FOUND",
    "in tolerance block"))

checks.append(row("0.3735/0.3745 tolerance group",
    found("0.3735") and found("0.3745"),
    "0.3735/0.3745",
    str(parsed.get("tolerance_groups"))))

results["tolerances"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# 9. NOTES & CALLOUTS
# ─────────────────────────────────────────────
section("9. NOTES & CALLOUTS")
checks = []

checks.append(row("NOTE #1 (Alternate method)",
    found("NOTE #1"),
    "NOTE #1" if found("NOTE #1") else "NOT FOUND", ""))

checks.append(row("NOTE #2 (Electropolish)",
    found("NOTE #2"),
    "NOTE #2" if found("NOTE #2") else "NOT FOUND", ""))

checks.append(row("NEAR SIDE callout",
    found("NEAR SIDE"),
    "NEAR SIDE" if found("NEAR SIDE") else "NOT FOUND", ""))

checks.append(row("ALTERNATE METHOD callout",
    found("ALTERNATE METHOD"),
    "ALTERNATE METHOD" if found("ALTERNATE METHOD") else "NOT FOUND", ""))

checks.append(row("TRADITIONAL DOWEL PIN ASSY",
    found("TRADITIONAL DOWEL PIN"),
    "TRADITIONAL DOWEL PIN ASSY" if found("TRADITIONAL DOWEL PIN") else "NOT FOUND", ""))

checks.append(row("CONFIDENTIALITY NOTICE",
    found("CONFIDENTIALITY NOTICE"),
    "PRESENT" if found("CONFIDENTIALITY NOTICE") else "NOT FOUND", ""))

results["notes"] = {"pass": sum(checks), "total": len(checks)}

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  DETAILED ACCURACY SUMMARY — BEV820-BM-01")
print(SEP)

total_pass  = 0
total_checks = 0

categories = [
    ("Identity",    "identity"),
    ("Threads",     "threads"),
    ("Holes",       "holes"),
    ("Dowel Pin",   "dowel"),
    ("Radii",       "radii"),
    ("Chamfers",    "chamfers"),
    ("Dimensions",  "dimensions"),
    ("Tolerances",  "tolerances"),
    ("Notes",       "notes"),
]

for label, key in categories:
    p = results[key]["pass"]
    t = results[key]["total"]
    pct = p / t * 100 if t else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  {label:15s}  {bar}  {p:2d}/{t:2d}  ({pct:.0f}%)")
    total_pass   += p
    total_checks += t

overall_pct = total_pass / total_checks * 100

print(f"\n{SEP2}")
print(f"  OVERALL OCR ACCURACY  :  {total_pass}/{total_checks}  ({overall_pct:.1f}%)")

# Parsed JSON completeness
expected_keys = [
    "part_number","revision","material","surface_finish","units",
    "surface_area","metric_threads","unc_threads","pcd_features",
    "slot_features","hole_patterns","holes","angles","radii",
    "chamfers","fit_tolerances","tolerance_groups","dimensions",
    "depths","classified_dimensions"
]
filled = sum(1 for k in expected_keys
             if parsed.get(k) not in [None, [], {}])
print(f"  PARSED JSON COMPLETENESS:  {filled}/{len(expected_keys)}  ({filled/len(expected_keys)*100:.0f}%)")
print(f"\n  OCR text quality:")
garbled   = len(re.findall(r'[A-Za-z]{1,3}[^A-Za-z0-9\s.,/\-°×xØ$]{2,}', raw))
clean_num = len(re.findall(r'\b\d+(?:\.\d+)?\b', raw))
print(f"    Total characters    : {len(raw)}")
print(f"    Total words         : {len(raw.split())}")
print(f"    Clean numeric tokens: {clean_num}")
print(f"    Garbled tokens      : {garbled}  ({garbled/len(raw.split())*100:.1f}% of words)")
print(f"    Ø symbols detected  : {raw.count('Ø')}")
print(SEP)
print("  DONE")
print(SEP)

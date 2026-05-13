"""
Phase 1 Feature Extractor
=========================
Parses engineering OCR text into structured JSON features.

NOTE: This script is a ONE-TIME setup tool.
      The parsed JSONs are already generated and saved in:
        features/samples/sample_01_N001/1-02-1065-N001_parsed.json
        features/samples/sample_02_BEV820/blueprint_page_parsed.json

      DO NOT re-run this unless you are adding a NEW drawing/sample.
      The feature_extractor.py pipeline reads these JSONs directly.

Usage (only when adding a new sample):
  python features/scripts/parse_engineering_text.py <ocr_txt> <output_json>

Handles: part number, revision, material, surface finish,
threads (metric + UNC), holes, hole patterns, PCD, slots,
radii, chamfers, tolerances, fit tolerances, depths,
angles, dimensions (classified), and surface area.
"""

import re
import json
import sys
import os


# =====================================================
# CONFIGURATION
# =====================================================

DEFAULT_OCR_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "samples", "sample_01_N001", "1-02-1065-N001_ocr.txt"
)

DEFAULT_OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "samples", "sample_01_N001", "1-02-1065-N001_parsed.json"
)


# =====================================================
# OCR NORMALIZATION
# =====================================================

OCR_REPLACEMENTS = {
    "O/": "Ø",
    "o/": "Ø",
    "0/": "Ø",
    "¢": "Ø",
    "ø": "Ø",
    "⌀": "Ø",
    "Ø ": "Ø",
    # Common OCR digit/letter confusions
    "RO.": "R0.",
    "RO ": "R0.",
    "l6": "16",
    "l5": "15",
    "l2": "12",
    "l1": "11",
    "l0": "10",
    "O.0": "0.0",
    "O.5": "0.5",
    "TH15": "THIS",
    "ELECTROPOL15H": "ELECTROPOLISH",
    "ELECTROLES5": "ELECTROLESS",
    # Slash-zero confusion in dimensions
    "Ø6Ø5": "065",
}


def normalize_ocr(text: str) -> str:
    """Apply OCR correction replacements and return cleaned text."""
    for bad, good in OCR_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# =====================================================
# HELPER: DEDUPLICATE LIST OF DICTS / PRIMITIVES
# =====================================================

def deduplicate(items):
    seen = set()
    result = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# =====================================================
# EXTRACTION FUNCTIONS
# =====================================================

def extract_part_number(text: str):
    """
    Matches patterns like:
      1-02-1065-N001
      BEV820-BM-01
    """
    patterns = [
        r'\b(\d{1,4}-\d{2}-\d{3,6}-[A-Z]\d{3,4})\b',
        r'\b(BEV[A-Z0-9\-]+)\b',
        r'\b([A-Z]{2,8}-[A-Z0-9]{2,10}-\d{2})\b',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1)
            # Reject phone-number-like matches
            if not re.match(r'^\d{3}-\d{4}$', candidate):
                return candidate
    return None


def extract_revision(text: str, filename: str = ""):
    """
    Matches REV A, REV: B, Rev B, ISSUE A, etc.
    Falls back to filename if OCR text doesn't contain a clean revision.
    """
    patterns = [
        r'\bREV(?:ISION)?[:\s.]*([A-Z])\b',
        r'\bISSUE[:\s]+([A-Z])\b',
        r'REV:\s*([A-Z])\b',
        # Title block row: letter alone on a line after ISSUE/REV header
        r'(?:REV|ISSUE)\s*\|\s*DATE.*?\n.*?([A-Z])\s*\|',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            candidate = m.group(1).upper()
            # Reject single letters that are clearly not revisions
            if candidate in ('I', 'O', 'Q', 'S', 'X', 'Z'):
                continue
            return candidate

    # Fallback: check filename for "Rev X" pattern
    if filename:
        m = re.search(r'\bRev\s*([A-Z])\b', os.path.basename(filename))
        if m:
            return m.group(1).upper()

    return None


def extract_material(text: str):
    """
    Looks for material callouts:
      AISI 316, SS316, EN1A, ALUMINUM, STEEL, etc.
    """
    patterns = [
        r'\b(AISI\s*\d{3,4}[A-Z]?)\b',
        r'\b(SS\s*\d{3,4})\b',
        r'\b(EN\d{1,3}[A-Z]?)\b',
        r'\b(STAINLESS\s+STEEL)\b',
        r'\b(ALUMINUM(?:\s+ALLOY)?)\b',
        r'\b(ALUMINIUM(?:\s+ALLOY)?)\b',
        r'\b(STEEL)\b',
        r'\b(TITANIUM)\b',
        r'\b(BRASS)\b',
        r'\b(COPPER)\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_surface_finish(text: str):
    """
    Extracts surface finish callouts:
      ELECTROLESS NICKEL, ELECTROPOLISH, ANODIZE, etc.
    Also extracts Ra values.
    """
    finish = None
    finish_patterns = [
        r'(ELECTROLESS\s+NICKEL(?:\s+Fe/NiP\([^)]+\)\s*\d+\s*[uµ]m)?)',
        r'(ELECTROPOLISH)',
        r'(HARD\s+ANODIZE)',
        r'(ANODIZE)',
        r'(ZINC\s+PLATE)',
        r'(ZINC)',
        r'(BLACK\s+OXIDE)',
        r'(PASSIVATE)',
        r'(NICKEL\s+PLATE)',
        r'(CHROME\s+PLATE)',
        r'(POWDER\s+COAT)',
    ]
    for pat in finish_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            finish = m.group(1).upper()
            break

    # Ra surface roughness
    ra_values = re.findall(
        r'(\d+(?:\.\d+)?)\s*Ra',
        text,
        re.IGNORECASE
    )
    ra_values = [float(v) for v in ra_values]

    return finish, deduplicate(ra_values)


def extract_units(text: str):
    """Detect inch vs mm drawings."""
    if re.search(r'\ball\s+dimensions\s+(?:in\s+)?mm\b', text, re.IGNORECASE):
        return "mm"
    if re.search(r'\ball\s+dimensions\s+(?:in\s+)?inch', text, re.IGNORECASE):
        return "inch"
    if re.search(r'\binch(?:es)?\b', text, re.IGNORECASE):
        return "inch"
    # Inch drawings typically have tolerances like +0.005 / ±0.010 (small decimals < 0.1)
    # and no large mm values. Check if most dimensions are < 5.0
    nums = re.findall(r'\b(\d+\.\d+)\b', text)
    if nums:
        vals = [float(n) for n in nums if float(n) < 500]
        if vals and sum(1 for v in vals if v < 5.0) / len(vals) > 0.85:
            return "inch"
    return "mm"  # default for engineering drawings


def extract_surface_area(text: str):
    """Extracts 'Surface area = 38726.48 square millimeters'."""
    m = re.search(
        r'[Ss]urface\s+area\s*=\s*(\d+(?:\.\d+)?)\s*square\s*(millimeters?|mm|inches?|in)',
        text
    )
    if m:
        return {
            "value": float(m.group(1)),
            "unit": "mm²" if "mm" in m.group(2).lower() or "milli" in m.group(2).lower() else "in²"
        }
    return None


def extract_metric_threads(text: str):
    """
    Matches:
      M4x0.7 - 6H DEPTH 12.0
      M6x1.0-6H
      M4x0.7 -6H
    """
    threads = []
    pattern = re.compile(
        r'(M\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)'
        r'(?:\s*[-–]\s*(\d+[A-Z]{1,2}))?'
        r'(?:\s+DEPTH\s+(\d+(?:\.\d+)?))?',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        size = m.group(1).upper()
        pitch = float(m.group(2))
        tol_class = m.group(3).upper() if m.group(3) else None
        depth = float(m.group(4)) if m.group(4) else None
        threads.append({
            "thread": f"{size}x{pitch}",
            "tolerance_class": tol_class,
            "depth": depth
        })
    return deduplicate(threads)


def extract_unc_threads(text: str):
    """
    Matches:
      1/4-20 UNC-2B
      10-32 UNC
      #10-32 UNC-2B
    """
    threads = []
    pattern = re.compile(
        r'(#?\d+(?:\/\d+)?-\d+\s*UNC(?:-\d+[A-Z])?)',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        threads.append({
            "thread": m.group(1).strip().upper()
        })
    return deduplicate(threads)


def extract_holes(text: str):
    """
    Matches:
      Ø 4.5 THRU
      Ø5.2 DEPTH 11.1
      3x Ø 4.5 THRU
      12x Ø 5.2 DEPTH 11.05
    """
    holes = []
    hole_patterns_list = []

    # Hole patterns: NxØD [THRU | DEPTH d]
    pattern_nx = re.compile(
        r'(\d+)\s*[xX×]\s*Ø\s*(\d+(?:\.\d+)?)'
        r'(?:\s+(THRU|THROUGH)|\s+DEPTH\s+(\d+(?:\.\d+)?))?',
        re.IGNORECASE
    )
    for m in pattern_nx.finditer(text):
        count = int(m.group(1))
        dia = round(float(m.group(2)), 4)
        through = bool(m.group(3))
        depth = float(m.group(4)) if m.group(4) else None
        hole_patterns_list.append({
            "count": count,
            "diameter": dia,
            "through": through,
            "depth": depth
        })

    # Simple holes: Ø D [THRU | DEPTH d]
    pattern_simple = re.compile(
        r'(?<!\d[xX×]\s)Ø\s*(\d+(?:\.\d+)?)'
        r'(?:\s+(THRU|THROUGH)|\s+DEPTH\s+(\d+(?:\.\d+)?))?',
        re.IGNORECASE
    )
    for m in pattern_simple.finditer(text):
        dia = round(float(m.group(1)), 4)
        if dia > 500:  # sanity check
            continue
        through = bool(m.group(2))
        depth = float(m.group(3)) if m.group(3) else None
        holes.append({
            "diameter": dia,
            "through": through,
            "depth": depth
        })

    return deduplicate(hole_patterns_list), deduplicate(holes)


def extract_pcd_features(text: str):
    """
    Matches:
      Ø96.0 PCD
      84.0 PCD
      $96.0 PCD
    Rejects OCR artifacts (values > 500 are likely noise).
    """
    pcd_list = []
    pattern = re.compile(
        r'[$Ø]?\s*(\d+(?:\.\d+)?)\s*PCD',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        val = round(float(m.group(1)), 4)
        if val > 500:  # reject OCR noise like 840.0 from $84.0
            continue
        pcd_list.append({"diameter": val})
    return deduplicate(pcd_list)


def extract_slot_features(text: str):
    """
    Matches:
      Ø10.0 SLOTS
      10.0 SLOTS
    """
    slots = []
    pattern = re.compile(
        r'Ø?\s*(\d+(?:\.\d+)?)\s+SLOTS?',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        slots.append({"slot_width": round(float(m.group(1)), 4)})
    return deduplicate(slots)


def extract_radii(text: str):
    """
    Matches:
      R3.0, R0.4, R0.2
    Rejects huge values (likely not radii).
    """
    radii = []
    pattern = re.compile(r'\bR(\d+(?:\.\d+)?)\b', re.IGNORECASE)
    for m in pattern.finditer(text):
        val = round(float(m.group(1)), 4)
        if val > 500:
            continue
        radii.append({"radius": val})
    return deduplicate(radii)


def extract_chamfers(text: str):
    """
    Matches:
      0.2 x 45°
      0.5 x 45
      1.0 X 90
    """
    chamfers = []
    pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*[xX×]\s*(45|90)°?',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        size = round(float(m.group(1)), 4)
        angle = int(m.group(2))
        chamfers.append({"size": size, "angle": angle})
    return deduplicate(chamfers)


def extract_fit_tolerances(text: str):
    """
    Matches:
      H8 +0.046 / 0.000   → fit_class=H8, upper=+0.046, lower=0.000
      H7
      g6
    """
    fits = []
    # Full fit with deviation values: H8 +0.046 / 0.000 or H8 +0.046 0.000
    pattern_full = re.compile(
        r'\b([A-Z][0-9]+)\s*\+(\d+\.\d+)\s*[/\\]?\s*(\d+\.\d+)',
        re.IGNORECASE
    )
    for m in pattern_full.finditer(text):
        fits.append({
            "fit_class": m.group(1).upper(),
            "tolerance": round(float(m.group(2)), 4)
        })

    # Standalone fit class (H7, g6, etc.) not already captured
    known_fits = {"H6", "H7", "H8", "H9", "H10", "H11",
                  "G6", "G7", "F7", "E8", "D9",
                  "h6", "h7", "h8", "g6", "f7", "e8"}
    captured = {f["fit_class"] for f in fits}
    pattern_simple = re.compile(r'\b([A-Z][0-9]+)\b')
    for m in pattern_simple.finditer(text):
        candidate = m.group(1)
        if candidate in known_fits and candidate.upper() not in captured:
            fits.append({"fit_class": candidate.upper(), "tolerance": None})
            captured.add(candidate.upper())

    return deduplicate(fits)


def extract_tolerance_groups(text: str):
    """
    Matches bilateral tolerances like:
      54.01 / 54.054
      0.3735 / 0.3745
    Rejects pairs where the spread is too large (> 2mm) to be a tolerance.
    """
    groups = []
    pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)'
    )
    for m in pattern.finditer(text):
        lower = round(float(m.group(1)), 4)
        upper = round(float(m.group(2)), 4)
        if lower >= upper:
            continue
        spread = upper - lower
        # Real tolerances are tight — reject anything wider than 2 units
        if spread > 2.0:
            continue
        nominal = round((lower + upper) / 2, 4)
        groups.append({
            "nominal": nominal,
            "lower_limit": lower,
            "upper_limit": upper
        })
    return deduplicate(groups)


def extract_depths(text: str):
    """
    Matches:
      DEPTH 12.0
      4.0 DEPTH 11.1
    Rejects values that are clearly OCR noise (> 200).
    """
    depths = []
    pattern = re.compile(
        r'(?:(\d+(?:\.\d+)?)\s+)?DEPTH\s+(\d+(?:\.\d+)?)',
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        depth_val = round(float(m.group(2)), 4)
        if depth_val > 200:
            continue
        depths.append(depth_val)
    return sorted(set(depths))


def extract_angles(text: str):
    """
    Matches standalone angle values like 13°, 42°, 15°.
    Excludes chamfer angles (45°, 90°) and projection angles.
    """
    angles = set()
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*°')
    for m in pattern.finditer(text):
        val = round(float(m.group(1)), 1)
        if val in (45.0, 90.0, 180.0, 360.0):
            continue
        if val > 360:
            continue
        angles.add(val)
    return sorted(angles)


def extract_dimensions(text: str):
    """
    Extracts all numeric dimension values.
    Filters OCR noise (values > 500 that aren't real dimensions).
    Classifies them into categories.
    """
    # Remove $ signs that OCR confuses with Ø (e.g. $80.0 → 80.0 already via Ø)
    clean = re.sub(r'\$(\d)', r'\1', text)

    raw = re.findall(r'\b(\d+\.\d+)\b', clean)
    seen = set()
    dims = []
    for d in raw:
        val = round(float(d), 4)
        if val in seen:
            continue
        # Reject obvious OCR noise: values > 500 that don't appear as Ø callouts
        if val > 500:
            continue
        seen.add(val)
        dims.append(val)
    dims.sort()

    # Classification thresholds (mm drawing)
    tiny = [d for d in dims if d < 2.0]
    small = [d for d in dims if 2.0 <= d < 11.0]
    medium = [d for d in dims if 11.0 <= d < 40.0]
    large = [d for d in dims if d >= 40.0]

    # Tolerances: very small values (< 0.1)
    tolerances = [d for d in dims if d < 0.1]

    # Outer diameters: large values that appear near Ø symbol
    outer_dia_pattern = re.compile(r'Ø\s*(\d+(?:\.\d+)?)')
    outer_dias = set()
    for m in outer_dia_pattern.finditer(text):
        val = round(float(m.group(1)), 4)
        if val >= 40.0:
            outer_dias.add(val)

    # PCD dimensions
    pcd_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*PCD', re.IGNORECASE)
    pcd_dims = set()
    for m in pcd_pattern.finditer(text):
        pcd_dims.add(round(float(m.group(1)), 4))

    classified = {
        "tiny_features": [d for d in tiny if d >= 0.1],
        "small_features": small,
        "medium_features": medium,
        "large_features": large,
        "tolerances": tolerances,
        "outer_diameters": sorted(outer_dias),
        "body_dimensions": sorted(
            d for d in large if d not in pcd_dims
        ),
        "feature_dimensions": small + medium,
        "pcd_dimensions": sorted(pcd_dims),
    }

    return dims, classified


# =====================================================
# MAIN PIPELINE
# =====================================================

def parse_engineering_text(ocr_text: str, source_file: str = "") -> dict:
    """
    Full Phase 1 feature extraction pipeline.
    Returns a structured dict of all extracted features.
    """
    text = normalize_ocr(ocr_text)
    upper = text.upper()

    # --- Core metadata ---
    part_number = extract_part_number(text)
    revision = extract_revision(text, source_file)
    material = extract_material(upper)
    surface_finish, ra_values = extract_surface_finish(upper)
    units = extract_units(upper)
    surface_area = extract_surface_area(text)

    # --- Geometry ---
    metric_threads = extract_metric_threads(upper)
    unc_threads = extract_unc_threads(upper)
    hole_patterns, holes = extract_holes(upper)
    pcd_features = extract_pcd_features(upper)
    slot_features = extract_slot_features(upper)
    radii = extract_radii(upper)
    chamfers = extract_chamfers(upper)

    # --- Tolerances ---
    fit_tolerances = extract_fit_tolerances(text)
    tolerance_groups = extract_tolerance_groups(text)

    # --- Dimensions ---
    depths = extract_depths(upper)
    angles = extract_angles(text)
    dimensions, classified_dimensions = extract_dimensions(text)

    features = {
        "part_number": part_number,
        "revision": revision,
        "material": material,
        "surface_finish": surface_finish,
        "ra_values": ra_values,
        "units": units,
        "surface_area": surface_area,
        "metric_threads": metric_threads,
        "unc_threads": unc_threads,
        "pcd_features": pcd_features,
        "slot_features": slot_features,
        "hole_patterns": hole_patterns,
        "holes": holes,
        "angles": angles,
        "radii": radii,
        "chamfers": chamfers,
        "fit_tolerances": fit_tolerances,
        "tolerance_groups": tolerance_groups,
        "dimensions": dimensions,
        "depths": depths,
        "classified_dimensions": classified_dimensions,
    }

    return features


# =====================================================
# ENTRY POINT
# =====================================================

def main():
    # Allow overriding input/output via CLI args
    ocr_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OCR_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE

    if not os.path.exists(ocr_file):
        print(f"[ERROR] OCR file not found: {ocr_file}")
        sys.exit(1)

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    print(f"\n[INFO] Parsing: {ocr_file}")

    features = parse_engineering_text(ocr_text, source_file=ocr_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=4)

    print(f"[OK]  Saved: {output_file}")
    print("\n--- SUMMARY ---")
    print(f"  Part Number    : {features['part_number']}")
    print(f"  Revision       : {features['revision']}")
    print(f"  Material       : {features['material']}")
    print(f"  Surface Finish : {features['surface_finish']}")
    print(f"  Units          : {features['units']}")
    print(f"  Surface Area   : {features['surface_area']}")
    print(f"  Metric Threads : {len(features['metric_threads'])}")
    print(f"  UNC Threads    : {len(features['unc_threads'])}")
    print(f"  Hole Patterns  : {len(features['hole_patterns'])}")
    print(f"  Holes          : {len(features['holes'])}")
    print(f"  PCD Features   : {len(features['pcd_features'])}")
    print(f"  Slot Features  : {len(features['slot_features'])}")
    print(f"  Radii          : {len(features['radii'])}")
    print(f"  Chamfers       : {len(features['chamfers'])}")
    print(f"  Fit Tolerances : {len(features['fit_tolerances'])}")
    print(f"  Tol Groups     : {len(features['tolerance_groups'])}")
    print(f"  Depths         : {features['depths']}")
    print(f"  Angles         : {len(features['angles'])}")
    print(f"  Dimensions     : {len(features['dimensions'])}")
    print("\n[DONE] Phase 1 feature extraction complete.")


if __name__ == "__main__":
    main()

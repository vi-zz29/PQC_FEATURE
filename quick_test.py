import sys
import cv2
import numpy as np
from pathlib import Path
from cad_image_alignment import align, match_best_template


def preprocess_cad(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load CAD image: {path}")
    inv = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(inv, (3, 3), 0)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges


def preprocess_real(img: np.ndarray) -> tuple:
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    real_masked = cv2.bitwise_and(img, img, mask=mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(real_masked, cv2.MORPH_GRADIENT, k3)
    _, internal = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
    internal = cv2.morphologyEx(
        internal,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    outer = cv2.Canny(mask, 50, 150)
    real_edges = cv2.bitwise_or(internal, outer)
    contours, _ = cv2.findContours(real_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(real_edges)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8:
            cv2.drawContours(clean, [cnt], -1, 255, 1)
    real_edges = clean
    return real_edges, mask, real_masked, gradient


def print_result(name: str, result, rank: int = None):
    prefix = (
        f"  {name}"
        if rank == 1
        else f"  {rank}. {name}"
        if rank
        else f"   {name}"
    )
    print(f"\n{prefix}")
    print(f"   Coverage:        {result.coverage:.4f}  ({result.coverage:.1%})")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Strategy:        {result.strategy}")
    if result.inlier_ratio:
        print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")


def main():
    print("=" * 70)
    print("Quick Alignment Test")
    print("=" * 70)

    multi_mode = False

    if len(sys.argv) >= 5 and len(sys.argv) % 2 == 0:
        real_path = sys.argv[1]
        pairs = sys.argv[2:]
        templates_input = [
            (pairs[i + 1], pairs[i])
            for i in range(0, len(pairs), 2)
        ]
        multi_mode = True

    elif len(sys.argv) == 3:
        cad_path, real_path = sys.argv[1], sys.argv[2]

    else:
        cad_candidates = ['cad.png', 'cad_front.png', 'cad_image.png', 'cad.jpg']
        real_candidates = ['real.png', 'real_photo.png', 'real_image.png', 'real.jpg', 'photo.jpg']
        cad_path = next((p for p in cad_candidates if Path(p).exists()), None)
        real_path = next((p for p in real_candidates if Path(p).exists()), None)
        if cad_path is None or real_path is None:
            print("\n[FAIL] Could not find images automatically.")
            print("\nSingle:  python quick_test.py cad.png real.png")
            print("Multi: python quick_test.py real.png cad1.png 'Object 1' cad2.png 'Object 2'")
            return

    print(f"\nLoading real image: {real_path}")
    real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    if real is None:
        print(f"[FAIL] Could not load: {real_path}")
        return

    print(f"   Shape: {real.shape}")
    print(f"\nPreprocessing real image...")
    real_edges, mask, real_masked, gradient = preprocess_real(real)
    print(f"   Real edges: {np.count_nonzero(real_edges)} pixels")

    cv2.imwrite('debug_mask.png', mask)
    cv2.imwrite('debug_masked_image.png', real_masked)
    cv2.imwrite('debug_gradient.png', gradient)
    cv2.imwrite('debug_final_edges.png', real_edges)

    SAMPLE_MAP = {
        ("cad2.png", "real2.png"):  "edges_N001_front.png",
        ("cad4.png", "real4.png"):  "edges_N001_rear.png",
        ("cad1.png", "real1.png"):  "edges_BEV820_front.png",
        ("cad3.png", "real3.png"):  "edges_BEV820_rear.png",
        ("cad2.png", "real2.jpeg"): "edges_N001_front.png",
        ("cad4.png", "real4.jpeg"): "edges_N001_rear.png",
        ("cad1.png", "real1.jpeg"): "edges_BEV820_front.png",
        ("cad3.png", "real3.jpeg"): "edges_BEV820_rear.png",
    }
    if not multi_mode:
        cad_base  = Path(cad_path).name
        real_base = Path(real_path).name
        named_edge = SAMPLE_MAP.get((cad_base, real_base))
        if named_edge:
            cv2.imwrite(named_edge, real_edges)
            print(f"   [OK] Named edge image saved: {named_edge}")

    if multi_mode:
        print(f"\nTemplates:")
        templates = []
        for name, cad_path in templates_input:
            try:
                cad_edges = preprocess_cad(cad_path)
                templates.append((name, cad_edges))
                print(f"   {name}: {cad_path} ({np.count_nonzero(cad_edges)} edge px)")
            except FileNotFoundError as e:
                print(f"   [FAIL] {e}")

        if not templates:
            print("[FAIL] No valid templates loaded.")
            return

        print(f"\nAligning {len(templates)} template(s)...")
        matches = match_best_template(templates, real_edges)

        print(f"\n{'=' * 70}")
        print("[OK] RESULTS — RANKED BY COVERAGE")
        print(f"{'=' * 70}")
        for m in matches:
            print_result(m.name, m.result, rank=m.rank)

        best = matches[0]
        print(f"\n{'=' * 70}")
        print("Identification:")
        if best.result.identified:
            print(f"   [OK] {best.name} identified (coverage {best.result.coverage:.1%})")
        else:
            print(f"   [FAIL] Unknown — best match '{best.name}' only {best.result.coverage:.1%} coverage")

        real_edges_vis = cv2.Canny(real_edges, 50, 150)
        overlay = np.zeros((*real_edges.shape, 3), dtype=np.uint8)
        overlay[:, :, 2] = best.result.aligned_image
        overlay[:, :, 1] = real_edges_vis
        cv2.imwrite('quick_overlay.png', overlay)
        cv2.imwrite('quick_aligned.png', best.result.aligned_image)
        print(f"\nSaved overlay for best match: quick_overlay.png")

    else:
        print(f"\nCAD image: {cad_path}")
        try:
            cad_edges = preprocess_cad(cad_path)
        except FileNotFoundError as e:
            print(f"[FAIL] {e}")
            return

        print(f"   CAD edges: {np.count_nonzero(cad_edges)} pixels")
        print(f"\nRunning alignment...")
        result = align(cad_edges, real_edges)

        print(f"\n{'=' * 70}")
        print("[OK] RESULTS")
        print(f"{'=' * 70}")
        print(f"\nMetrics:")
        print(f"   Strategy:        {result.strategy}")
        print(f"   Alignment Score: {result.alignment_score:.4f} (IoU)")
        print(f"   Coverage: {result.coverage:.4f} ({result.coverage:.1%})")
        print(f"   High Confidence: {result.high_confidence}")
        if result.inlier_ratio:
            print(f"   Inlier Ratio: {result.inlier_ratio:.2%}")

        print(f"\nIdentification:")
        if result.identified:
            print(f"   [OK] Object 1 identified (coverage {result.coverage:.1%})")
        else:
            print(f"   [FAIL] Unknown (coverage {result.coverage:.1%})")

        print(f"\nSaving results...")
        cv2.imwrite('quick_cad_edges.png', cad_edges)
        cv2.imwrite('quick_real_edges.png', real_edges)
        cv2.imwrite('quick_aligned.png', result.aligned_image)

        real_edges_vis = cv2.Canny(real_edges, 50, 150)
        overlay = np.zeros((*real_edges.shape, 3), dtype=np.uint8)
        overlay[:, :, 2] = result.aligned_image
        overlay[:, :, 1] = real_edges_vis
        cv2.imwrite('quick_overlay.png', overlay)

        print(f"   [OK] quick_cad_edges.png")
        print(f"   [OK] quick_real_edges.png")
        print(f"   [OK] quick_aligned.png")
        print(f"   [OK] quick_overlay.png (Red=CAD, Green=Real, Yellow=Match)")

    print(f"\n[OK] Done!")


if __name__ == "__main__":
    main()

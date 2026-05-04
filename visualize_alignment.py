"""
Better visualization of alignment results.
"""

import cv2
import numpy as np
from cad_image_alignment import align


def create_better_overlay(cad_edges, real_edges, result):
    """Create a better overlay visualization."""
    
    # Create RGB overlay
    h, w = cad_edges.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Red channel: CAD edges
    overlay[:, :, 2] = cad_edges
    
    # Green channel: Aligned real edges
    overlay[:, :, 1] = result.aligned_image
    
    # Where they overlap, it will be yellow (red + green)
    
    # Create a side-by-side comparison
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Panel 1: CAD edges (red)
    comparison[:, :w, 2] = cad_edges
    
    # Panel 2: Real edges before alignment (blue)
    # Resize real_edges to match CAD shape if needed
    if real_edges.shape != cad_edges.shape:
        real_edges_resized = cv2.resize(real_edges, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        real_edges_resized = real_edges
    comparison[:, w:2*w, 0] = real_edges_resized
    
    # Panel 3: Overlay (red + green = yellow where they match)
    comparison[:, 2*w:3*w] = overlay
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'CAD (Red)', (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, 'Real Before (Blue)', (w + 10, 30), font, 1, (255, 0, 0), 2)
    cv2.putText(comparison, 'Overlay (Yellow=Match)', (2*w + 10, 30), font, 1, (0, 255, 255), 2)
    
    # Add score info
    score_text = f'Score: {result.alignment_score:.4f} ({result.strategy})'
    cv2.putText(comparison, score_text, (2*w + 10, h - 20), font, 0.7, (255, 255, 255), 2)
    
    return overlay, comparison


def main():
    print("=" * 70)
    print("Better Alignment Visualization")
    print("=" * 70)
    
    # Load preprocessed edges
    cad_edges = cv2.imread('quick_cad_edges.png', cv2.IMREAD_GRAYSCALE)
    real_edges = cv2.imread('quick_real_edges.png', cv2.IMREAD_GRAYSCALE)
    
    if cad_edges is None or real_edges is None:
        print("\n❌ Could not load edge images. Run quick_test.py first!")
        return
    
    print(f"\n📥 Loaded edge maps:")
    print(f"   CAD:  {cad_edges.shape}, {np.count_nonzero(cad_edges)} edge pixels")
    print(f"   Real: {real_edges.shape}, {np.count_nonzero(real_edges)} edge pixels")
    
    # Run alignment
    print(f"\n⚙️  Running alignment...")
    result = align(cad_edges, real_edges)
    
    print(f"\n📊 Results:")
    print(f"   Strategy:        {result.strategy}")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Low Confidence:  {result.low_confidence}")
    if result.inlier_ratio:
        print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    overlay, comparison = create_better_overlay(cad_edges, real_edges, result)
    
    # Save results
    cv2.imwrite('better_overlay.png', overlay)
    cv2.imwrite('better_comparison.png', comparison)
    
    print(f"\n💾 Saved:")
    print(f"   ✓ better_overlay.png - Overlay only (Red=CAD, Green=Aligned, Yellow=Match)")
    print(f"   ✓ better_comparison.png - Side-by-side comparison")
    
    # Calculate overlap percentage
    cad_pixels = np.count_nonzero(cad_edges)
    aligned_pixels = np.count_nonzero(result.aligned_image)
    overlap_pixels = np.count_nonzero(np.logical_and(cad_edges > 0, result.aligned_image > 0))
    
    if cad_pixels > 0:
        overlap_pct = (overlap_pixels / cad_pixels) * 100
        print(f"\n📈 Overlap Analysis:")
        print(f"   CAD edge pixels:     {cad_pixels}")
        print(f"   Aligned edge pixels: {aligned_pixels}")
        print(f"   Overlapping pixels:  {overlap_pixels}")
        print(f"   Overlap percentage:  {overlap_pct:.1f}% of CAD edges")
    
    print(f"\n✅ Done! Check the images.")


if __name__ == "__main__":
    main()

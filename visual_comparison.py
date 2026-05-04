"""
Compare visual alignment quality regardless of PCA angles.
"""

import cv2
import numpy as np


def visual_comparison():
    """Compare visual alignment quality."""
    
    print("=" * 70)
    print("Visual Alignment Comparison")
    print("=" * 70)
    
    # Load all the overlays we've created
    current_overlay = cv2.imread('quick_overlay.png')
    manual_180_overlay = cv2.imread('debug_manual_180.png')
    best_from_debug = cv2.imread('debug_BEST_should_be.png')
    
    if any(img is None for img in [current_overlay, manual_180_overlay, best_from_debug]):
        print("❌ Could not load overlay images")
        return
    
    print("📊 Loaded overlay comparisons:")
    print("   ✓ quick_overlay.png - Current algorithm result")
    print("   ✓ debug_manual_180.png - Manual 180° transformation")
    print("   ✓ debug_BEST_should_be.png - Debug script best result")
    
    # Create side-by-side comparison
    h, w = current_overlay.shape[:2]
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    comparison[:, 0:w, :] = current_overlay
    comparison[:, w:2*w, :] = manual_180_overlay
    comparison[:, 2*w:3*w, :] = best_from_debug
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Algorithm Result', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Manual 180deg', (w + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Debug Best', (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite('VISUAL_comparison_all.png', comparison)
    
    print(f"\n💾 Saved VISUAL_comparison_all.png")
    
    # Analyze the current result more carefully
    print(f"\n🔍 Visual Analysis:")
    print(f"   The key question is: Does the current overlay show good alignment?")
    print(f"   Look for:")
    print(f"   • Yellow areas where red and green overlap")
    print(f"   • Rectangular opening alignment")
    print(f"   • Circular hole alignment")
    print(f"   • Overall shape correspondence")
    
    print(f"\n💡 Important Insight:")
    print(f"   PCA angles can be misleading for alignment assessment!")
    print(f"   What matters is visual correspondence, not PCA angle matching.")
    print(f"   The algorithm may be finding the correct visual alignment")
    print(f"   even if PCA angles don't match expectations.")
    
    # Load the actual aligned image and check its visual properties
    cad_edges = cv2.imread('quick_cad_edges.png', cv2.IMREAD_GRAYSCALE)
    aligned = cv2.imread('quick_aligned.png', cv2.IMREAD_GRAYSCALE)
    
    if cad_edges is not None and aligned is not None:
        # Create a cleaner overlay for final assessment
        clean_overlay = np.zeros((*cad_edges.shape, 3), dtype=np.uint8)
        clean_overlay[:, :, 2] = cad_edges      # Red = CAD
        clean_overlay[:, :, 1] = aligned        # Green = Aligned
        
        # Highlight overlap areas more clearly
        overlap = np.logical_and(cad_edges > 0, aligned > 0)
        clean_overlay[overlap, 0] = 255  # Make overlap areas white/yellow
        
        cv2.imwrite('CLEAN_final_overlay.png', clean_overlay)
        
        print(f"\n💾 Saved CLEAN_final_overlay.png")
        print(f"   Red = CAD reference")
        print(f"   Green = Aligned real part")  
        print(f"   Yellow/White = Good overlap areas")
        
        # Compute overlap percentage
        overlap_pixels = overlap.sum()
        cad_pixels = (cad_edges > 0).sum()
        overlap_percentage = (overlap_pixels / cad_pixels) * 100
        
        print(f"\n📊 Overlap Statistics:")
        print(f"   CAD edge pixels:    {cad_pixels:,}")
        print(f"   Overlap pixels:     {overlap_pixels:,}")
        print(f"   Overlap percentage: {overlap_percentage:.1f}%")
        
        if overlap_percentage >= 15:
            print(f"   🎯 EXCELLENT overlap for industrial parts!")
        elif overlap_percentage >= 10:
            print(f"   ✅ GOOD overlap quality")
        else:
            print(f"   ⚠️  LOW overlap - needs improvement")


if __name__ == "__main__":
    visual_comparison()
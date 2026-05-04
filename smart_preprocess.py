"""
Smart preprocessing script - tries multiple strategies and shows you the results.
Run this first to find the best preprocessing parameters.
"""

import cv2
import numpy as np
from pathlib import Path


def show_preprocessing_options(cad_path, real_path):
    """Show different preprocessing options side by side."""
    
    print("=" * 70)
    print("Smart Preprocessing - Finding Best Parameters")
    print("=" * 70)
    
    # Load images
    print(f"\n📥 Loading images...")
    cad = cv2.imread(cad_path, cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    
    if cad is None or real is None:
        print("❌ Could not load images")
        return
    
    print(f"   CAD shape:  {cad.shape}")
    print(f"   Real shape: {real.shape}")
    
    # Save original grayscale
    cv2.imwrite('step0_cad_original.png', cad)
    cv2.imwrite('step0_real_original.png', real)
    
    # ========== CAD PREPROCESSING ==========
    print(f"\n🔧 CAD Preprocessing...")
    _, cad_thresh = cv2.threshold(cad, 127, 255, cv2.THRESH_BINARY)
    cad_edges = cv2.Canny(cad_thresh, 50, 150)
    cv2.imwrite('step1_cad_edges.png', cad_edges)
    print(f"   ✓ CAD edges: {np.count_nonzero(cad_edges)} pixels")
    
    # ========== REAL PREPROCESSING - MULTIPLE STRATEGIES ==========
    print(f"\n🔧 Real Image Preprocessing (trying multiple strategies)...")
    
    # Strategy 1: Simple Canny with different thresholds
    print(f"\n   Strategy 1: Direct Canny (various thresholds)")
    real_s1_low = cv2.Canny(real, 20, 60)
    real_s1_med = cv2.Canny(real, 50, 150)
    real_s1_high = cv2.Canny(real, 100, 200)
    cv2.imwrite('strategy1a_canny_low_20_60.png', real_s1_low)
    cv2.imwrite('strategy1b_canny_med_50_150.png', real_s1_med)
    cv2.imwrite('strategy1c_canny_high_100_200.png', real_s1_high)
    print(f"      Low (20,60):    {np.count_nonzero(real_s1_low)} pixels")
    print(f"      Med (50,150):   {np.count_nonzero(real_s1_med)} pixels")
    print(f"      High (100,200): {np.count_nonzero(real_s1_high)} pixels")
    
    # Strategy 2: Background removal + Canny
    print(f"\n   Strategy 2: Background Removal + Canny")
    # Try Otsu thresholding
    _, real_otsu = cv2.threshold(real, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('strategy2a_otsu_mask.png', real_otsu)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    real_otsu_clean = cv2.morphologyEx(real_otsu, cv2.MORPH_CLOSE, kernel)
    real_otsu_clean = cv2.morphologyEx(real_otsu_clean, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('strategy2b_otsu_cleaned.png', real_otsu_clean)
    
    # Apply mask and detect edges
    real_masked = cv2.bitwise_and(real, real, mask=real_otsu_clean)
    cv2.imwrite('strategy2c_masked_image.png', real_masked)
    real_s2_edges = cv2.Canny(real_masked, 30, 100)
    cv2.imwrite('strategy2d_edges_from_masked.png', real_s2_edges)
    print(f"      Edges from masked: {np.count_nonzero(real_s2_edges)} pixels")
    
    # Strategy 3: Bilateral filter + Canny
    print(f"\n   Strategy 3: Bilateral Filter + Canny")
    real_bilateral = cv2.bilateralFilter(real, 9, 75, 75)
    cv2.imwrite('strategy3a_bilateral_filtered.png', real_bilateral)
    real_s3_edges = cv2.Canny(real_bilateral, 30, 90)
    cv2.imwrite('strategy3b_edges_from_bilateral.png', real_s3_edges)
    print(f"      Edges from bilateral: {np.count_nonzero(real_s3_edges)} pixels")
    
    # Strategy 4: Adaptive threshold + Canny
    print(f"\n   Strategy 4: Adaptive Threshold + Canny")
    real_adaptive = cv2.adaptiveThreshold(
        real, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite('strategy4a_adaptive_thresh.png', real_adaptive)
    real_s4_edges = cv2.Canny(real_adaptive, 50, 150)
    cv2.imwrite('strategy4b_edges_from_adaptive.png', real_s4_edges)
    print(f"      Edges from adaptive: {np.count_nonzero(real_s4_edges)} pixels")
    
    # Strategy 5: Morphological gradient
    print(f"\n   Strategy 5: Morphological Gradient")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    real_gradient = cv2.morphologyEx(real, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite('strategy5a_morphological_gradient.png', real_gradient)
    _, real_gradient_thresh = cv2.threshold(real_gradient, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite('strategy5b_gradient_thresholded.png', real_gradient_thresh)
    print(f"      Gradient edges: {np.count_nonzero(real_gradient_thresh)} pixels")
    
    # Strategy 6: Sobel edges
    print(f"\n   Strategy 6: Sobel Edge Detection")
    sobelx = cv2.Sobel(real, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(real, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    cv2.imwrite('strategy6a_sobel_magnitude.png', sobel_mag)
    _, real_sobel = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite('strategy6b_sobel_thresholded.png', real_sobel)
    print(f"      Sobel edges: {np.count_nonzero(real_sobel)} pixels")
    
    print(f"\n" + "=" * 70)
    print("✅ DONE - Check the output images!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  step0_*.png           - Original grayscale images")
    print(f"  step1_cad_edges.png   - CAD edges (reference)")
    print(f"  strategy1*.png        - Direct Canny with different thresholds")
    print(f"  strategy2*.png        - Background removal approach")
    print(f"  strategy3*.png        - Bilateral filter approach")
    print(f"  strategy4*.png        - Adaptive threshold approach")
    print(f"  strategy5*.png        - Morphological gradient approach")
    print(f"  strategy6*.png        - Sobel edge detection")
    print(f"\n💡 Look at these images and tell me which strategy captures")
    print(f"   the part outline best (similar to the CAD edges).")
    print(f"\n   Then I'll update quick_test.py to use that strategy!")


if __name__ == "__main__":
    # Look for images
    cad_candidates = ['cad.png', 'cad_front.png', 'cad_image.png', 'cad.jpg']
    real_candidates = ['real.png', 'real_photo.png', 'real_image.png', 'real.jpg', 'photo.jpg']
    
    cad_path = None
    real_path = None
    
    for candidate in cad_candidates:
        if Path(candidate).exists():
            cad_path = candidate
            break
    
    for candidate in real_candidates:
        if Path(candidate).exists():
            real_path = candidate
            break
    
    if cad_path is None or real_path is None:
        print("\n❌ Could not find images.")
        print("\nPlease save your images as:")
        print("  - cad.png (or cad.jpg)")
        print("  - real.png (or real.jpg)")
    else:
        show_preprocessing_options(cad_path, real_path)

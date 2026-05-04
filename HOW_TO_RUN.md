# How to Run the CAD-Image Alignment Module

## 🚀 Quick Start (3 Steps)

### Step 1: Save Your Images
Save the two images you showed me:
- **CAD image** (the line drawing) → Save as `cad.png`
- **Real image** (the photo) → Save as `real.png`

Put them in this folder (same folder as this file).

### Step 2: Run the Quick Test
```bash
python quick_test.py
```

### Step 3: Check the Results
Look at these output files:
- `quick_overlay.png` - Shows alignment (Yellow = good match)
- `quick_aligned.png` - The aligned real image
- Console output - Shows the alignment score

---

## 📖 Detailed Instructions

### Option 1: Quick Test (Automatic)

**What it does:** Automatically finds your images and runs alignment

**Steps:**
1. Save your images as `cad.png` and `real.png`
2. Run:
   ```bash
   python quick_test.py
   ```
3. Check the output files

**Output files:**
- `quick_cad_edges.png` - Preprocessed CAD edges
- `quick_real_edges.png` - Preprocessed real edges
- `quick_aligned.png` - Aligned result
- `quick_overlay.png` - Visualization (Red=CAD, Green=Aligned, Yellow=Match)

---

### Option 2: Full Test (Manual)

**What it does:** Full preprocessing and detailed visualization

**Steps:**
1. Run with your image paths:
   ```bash
   python test_real_images.py cad.png real.png
   ```

**Output files:**
- `preprocessed_cad.png` - CAD edge map
- `preprocessed_real.png` - Real edge map
- `aligned_result.png` - Aligned image
- `alignment_result.png` - 4-panel visualization
- `transform_matrix.npy` - Transformation matrix (can be reused)

---

### Option 3: Python Script (Custom)

**What it does:** Use the module in your own code

**Example:**
```python
import cv2
from cad_image_alignment import align

# Load your preprocessed edge maps
cad_edges = cv2.imread('cad_edges.png', cv2.IMREAD_GRAYSCALE)
real_edges = cv2.imread('real_edges.png', cv2.IMREAD_GRAYSCALE)

# Run alignment
result = align(cad_edges, real_edges)

# Check results
print(f"Strategy: {result.strategy}")
print(f"Score: {result.alignment_score:.4f}")

# Save aligned image
cv2.imwrite('aligned.png', result.aligned_image)
```

---

## 🔧 Preprocessing Your Images

### For CAD Images (Line Drawings):
```python
import cv2

# Load image
img = cv2.imread('cad.png', cv2.IMREAD_GRAYSCALE)

# Threshold
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Edge detection
edges = cv2.Canny(thresh, 50, 150)

# Save
cv2.imwrite('cad_edges.png', edges)
```

### For Real Images (Photos):
```python
import cv2

# Load image
img = cv2.imread('real.png', cv2.IMREAD_GRAYSCALE)

# Remove background (for white backgrounds)
_, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# Reduce noise while preserving edges
filtered = cv2.bilateralFilter(mask, 9, 75, 75)

# Edge detection
edges = cv2.Canny(filtered, 50, 150)

# Clean up noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Save
cv2.imwrite('real_edges.png', edges)
```

---

## 📊 Understanding the Results

### Alignment Score
- **0.9 - 1.0**: Excellent alignment ✅
- **0.7 - 0.9**: Good alignment ✅
- **0.5 - 0.7**: Moderate alignment ⚠️
- **0.3 - 0.5**: Poor alignment ⚠️
- **0.0 - 0.3**: Very poor alignment ❌

### Strategy
- **"homography"**: Fine alignment succeeded (best quality) ✅
- **"affine_coarse_only"**: Fine alignment failed, used coarse only ⚠️
- **"identity"**: Both failed, no alignment applied ❌

### Low Confidence Flag
- **False**: Alignment is trustworthy ✅
- **True**: Alignment score < 0.30, needs review ⚠️

---

## 🐛 Troubleshooting

### Problem: "Could not load image"
**Solution:** Check that:
- Image file exists in the current folder
- File name is correct (case-sensitive on Linux/Mac)
- Image format is supported (.png, .jpg, .jpeg, .bmp)

### Problem: "No valid contour found"
**Solution:** Your edge map might be:
- Too empty (not enough edges)
- Too noisy (too many small contours)
- Try adjusting Canny thresholds: `cv2.Canny(img, 30, 100)` (lower) or `cv2.Canny(img, 100, 200)` (higher)

### Problem: Low alignment score
**Possible causes:**
1. **Images are very different** - Check if they're the same part
2. **Preprocessing issues** - Check `preprocessed_*.png` files
3. **Heavy noise** - Try stronger filtering
4. **Partial occlusion** - Part is cut off in one image

**Solutions:**
- Adjust preprocessing parameters
- Ensure both images show the complete part
- Check that background removal worked correctly

### Problem: Strategy is "affine_coarse_only"
**Meaning:** Fine alignment (ORB) failed, but coarse alignment worked

**Common causes:**
- Not enough features (simple shape)
- Too much noise (reflections)
- This is normal for simple shapes!

**Action:** Check the alignment score. If score > 0.6, it's probably fine.

---

## 🎯 Tips for Best Results

1. **Good lighting** - Reduce reflections and shadows
2. **Plain background** - White or green screen works best
3. **Complete part** - Don't cut off edges
4. **Consistent scale** - Keep camera distance similar
5. **Clean edges** - Good preprocessing is crucial

---

## 📞 Need Help?

If you're stuck, check:
1. The output images (`quick_overlay.png`, etc.)
2. The console output (shows what went wrong)
3. The preprocessed edge maps (should show clear part outline)

Common fixes:
- Adjust Canny thresholds (50, 150) → try (30, 100) or (100, 200)
- Adjust background threshold (200) → try (150) or (220)
- Check that images are grayscale (not RGB)

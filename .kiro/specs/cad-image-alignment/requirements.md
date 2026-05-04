# Requirements Document

## Introduction

This document defines requirements for the CAD-Image Alignment Module, a component of an industrial computer vision inspection system for mechanical parts. The module receives pre-processed edge maps of a CAD reference image and a real camera image, then computes and applies a geometric transformation so that the real image is spatially aligned with the CAD image. Accurate alignment is a prerequisite for downstream mismatch detection between the manufactured part and its design specification.

Both inputs are already edge maps produced by the existing preprocessing pipeline:
- **CAD edge map**: clean binary edges from a STEP-derived front-view render
- **Real edge map**: cleaned edges from a camera image after background removal, bilateral filtering, Canny detection, and morphological cleaning

The alignment must handle translation, scale, rotation, and optionally perspective distortion, under industrial accuracy requirements and in the presence of metallic surface reflections and noise.

---

## Glossary

- **Alignment_Module**: The software component defined by this document, responsible for computing and applying the geometric transformation between the real edge map and the CAD edge map.
- **CAD_Edge_Map**: A binary edge image derived from the CAD model's 2D front-view render, used as the reference (fixed) image.
- **Real_Edge_Map**: A binary edge image derived from the camera-captured image after preprocessing, used as the source (moving) image.
- **Aligned_Image**: The result of applying the computed transformation to the Real_Edge_Map so that it spatially corresponds to the CAD_Edge_Map.
- **Transformation_Matrix**: A 2D homography or affine matrix (3×3) encoding the geometric mapping from Real_Edge_Map coordinates to CAD_Edge_Map coordinates.
- **Alignment_Score**: A numeric value in [0.0, 1.0] quantifying the spatial overlap quality between the Aligned_Image and the CAD_Edge_Map, where 1.0 is perfect overlap.
- **Contour**: A connected curve of edge pixels representing the boundary of a shape region.
- **Primary_Contour**: The largest contour by area in an edge map, representing the outer boundary of the mechanical component.
- **Keypoint**: A salient image location detected by a feature detector (e.g., ORB or SIFT) used for feature-based matching.
- **Descriptor**: A numerical vector encoding the local appearance around a Keypoint, used for matching Keypoints across images.
- **Homography**: A projective transformation (8 degrees of freedom) that maps one planar image to another, accounting for translation, rotation, scale, and perspective.
- **Affine_Transform**: A transformation (6 degrees of freedom) that maps one image to another, accounting for translation, rotation, scale, and shear, but not perspective.
- **RANSAC**: Random Sample Consensus, a robust estimation algorithm used to reject outlier matches when computing a transformation.
- **Inlier_Ratio**: The fraction of feature matches classified as inliers by RANSAC relative to the total number of matches.
- **IoU**: Intersection over Union — the ratio of the pixel-wise intersection to the union of two binary edge maps, used as a component of the Alignment_Score.
- **Centroid**: The geometric center of mass of a contour or binary region.
- **Bounding_Box**: The axis-aligned rectangle of minimum area that encloses a contour.
- **Pipeline**: The end-to-end sequence of steps executed by the Alignment_Module: strategy selection → transformation estimation → transformation application → quality assessment.

---

## Requirements

### Requirement 1: Accept Preprocessed Edge Maps as Input

**User Story:** As an inspection system, I want the Alignment_Module to accept the preprocessed edge maps from the existing pipeline, so that no additional preprocessing is required before alignment.

#### Acceptance Criteria

1. THE Alignment_Module SHALL accept a CAD_Edge_Map as a 2D NumPy array of dtype `uint8` with pixel values in {0, 255}.
2. THE Alignment_Module SHALL accept a Real_Edge_Map as a 2D NumPy array of dtype `uint8` with pixel values in {0, 255}.
3. IF the CAD_Edge_Map or Real_Edge_Map has a shape with fewer than 2 dimensions or contains no non-zero pixels, THEN THE Alignment_Module SHALL raise a descriptive `ValueError` identifying which input is invalid.
4. IF the CAD_Edge_Map and Real_Edge_Map have different spatial resolutions, THEN THE Alignment_Module SHALL resize the Real_Edge_Map to match the CAD_Edge_Map resolution before processing.

---

### Requirement 2: Contour-Based Coarse Alignment

**User Story:** As an inspection engineer, I want the system to perform a fast contour-based coarse alignment using the Primary_Contour, so that the real image is roughly positioned before fine alignment.

#### Acceptance Criteria

1. WHEN coarse alignment is executed, THE Alignment_Module SHALL extract the Primary_Contour from both the CAD_Edge_Map and the Real_Edge_Map.
2. WHEN the Primary_Contour is extracted, THE Alignment_Module SHALL compute the Centroid and the axis-aligned Bounding_Box of each Primary_Contour.
3. WHEN the Centroids and Bounding_Boxes are computed, THE Alignment_Module SHALL compute a scale factor as the ratio of the CAD Primary_Contour Bounding_Box diagonal to the Real Primary_Contour Bounding_Box diagonal.
4. WHEN the scale factor is computed, THE Alignment_Module SHALL compute a translation vector that maps the Real_Edge_Map Centroid to the CAD_Edge_Map Centroid after scaling.
5. WHEN the Primary_Contour is extracted, THE Alignment_Module SHALL estimate the principal orientation angle of the Primary_Contour using PCA on the contour point distribution or the minimum area bounding rectangle, and include a rotation component in the coarse Affine_Transform.
6. THE coarse rotation estimation SHALL handle arbitrary orientations in the range [0°, 360°).
7. WHEN the coarse Affine_Transform is computed, THE Alignment_Module SHALL apply it to the Real_Edge_Map to produce a coarsely aligned intermediate image.
8. IF no contour is found in the Real_Edge_Map, THEN THE Alignment_Module SHALL log a warning and proceed to feature-based alignment without a coarse alignment step.

---

### Requirement 3: Feature-Based Fine Alignment

**User Story:** As an inspection engineer, I want the system to refine alignment using feature matching, so that sub-pixel accuracy is achieved for reliable mismatch detection.

#### Acceptance Criteria

1. WHEN fine alignment is executed, THE Alignment_Module SHALL detect Keypoints and compute Descriptors on both the coarsely aligned Real_Edge_Map (or original if coarse alignment was skipped) and the CAD_Edge_Map using ORB.
2. WHEN Keypoints are detected, THE Alignment_Module SHALL match Descriptors using a brute-force Hamming-distance matcher with cross-check enabled.
3. WHEN matches are obtained, THE Alignment_Module SHALL apply RANSAC to estimate a Homography from the matched Keypoint pairs, using a reprojection error threshold of 3.0 pixels.
4. WHEN RANSAC completes, THE Alignment_Module SHALL compute the Inlier_Ratio from the RANSAC result.
5. IF the number of matched Keypoint pairs before RANSAC is fewer than 8, THEN THE Alignment_Module SHALL fall back to contour-based alignment only and log a warning indicating insufficient keypoints.
6. IF the Inlier_Ratio is below 0.25, THEN THE Alignment_Module SHALL discard the Homography, fall back to the coarse Affine_Transform, and log a warning indicating low inlier ratio.
7. WHEN a valid Homography is computed, THE Alignment_Module SHALL apply it to the Real_Edge_Map to produce the Aligned_Image.

---

### Requirement 4: Transformation Output

**User Story:** As a downstream mismatch detection module, I want the alignment result to include both the aligned image and the transformation matrix, so that I can use either for further analysis.

#### Acceptance Criteria

1. THE Alignment_Module SHALL return the Aligned_Image as a 2D NumPy array of dtype `uint8` with the same spatial dimensions as the CAD_Edge_Map.
2. THE Alignment_Module SHALL return the Transformation_Matrix as a NumPy array of shape (3, 3) and dtype `float64`.
3. THE Alignment_Module SHALL return the Alignment_Score as a Python `float` in the range [0.0, 1.0].
4. THE Alignment_Module SHALL return a strategy label as a Python `str` indicating which alignment path was taken, with one of the values: `"homography"`, `"affine_coarse_only"`, or `"identity"`.

---

### Requirement 5: Alignment Quality Assessment

**User Story:** As an inspection engineer, I want the alignment quality to be measured and reported, so that I can detect unreliable alignments before mismatch detection proceeds.

#### Acceptance Criteria

1. WHEN the Aligned_Image is produced, THE Alignment_Module SHALL compute the Alignment_Score as the IoU between the Aligned_Image and the CAD_Edge_Map, both dilated by a 3×3 structuring element to allow for sub-pixel tolerance.
2. WHEN the Alignment_Score is computed, THE Alignment_Module SHALL include it in the returned result.
3. IF the Alignment_Score is below 0.30, THEN THE Alignment_Module SHALL set a `low_confidence` flag to `True` in the returned result and log a warning with the score value.
4. WHILE the `low_confidence` flag is `True`, THE Alignment_Module SHALL still return the best available Aligned_Image rather than raising an exception, so that the caller can decide how to handle low-confidence results.

---

### Requirement 6: Robustness to Metallic Reflections and Noise

**User Story:** As an inspection engineer working with metallic parts, I want the alignment to remain stable despite reflections and noise in the real image, so that inspection results are reliable across varying lighting conditions.

#### Acceptance Criteria

1. WHEN feature matching is performed, THE Alignment_Module SHALL use RANSAC to suppress the effect of spurious matches caused by reflections or noise artifacts.
2. WHEN the Primary_Contour is selected, THE Alignment_Module SHALL select the contour with the largest area, ignoring contours with area below 1% of the total image area, to suppress small noise contours.
3. WHEN the Homography is estimated, THE Alignment_Module SHALL enforce that the computed scale component of the Homography is within the range [0.5, 2.0]; IF the scale is outside this range, THEN THE Alignment_Module SHALL reject the Homography and fall back to the coarse Affine_Transform.
4. WHEN the Homography is validated, THE Alignment_Module SHALL NOT constrain the rotation component of the Homography, since parts may arrive at arbitrary orientations in the range [0°, 360°).

---

### Requirement 7: Transformation Invertibility and Composition

**User Story:** As a downstream module, I want the transformation to be mathematically consistent, so that I can apply it, invert it, or compose it with other transforms without introducing additional error.

#### Acceptance Criteria

1. THE Transformation_Matrix SHALL be a non-singular matrix (determinant ≠ 0) for all valid alignment results.
2. FOR ALL valid Transformation_Matrix values produced by the Alignment_Module, applying the matrix and then its inverse to the Real_Edge_Map SHALL produce an image whose non-zero pixel positions match the original Real_Edge_Map non-zero pixel positions within a tolerance of 2 pixels (round-trip property).
3. FOR ALL valid Transformation_Matrix values, composing two successive transformations SHALL produce a result equivalent to applying each transformation independently in sequence (composition consistency property).

---

### Requirement 8: Performance

**User Story:** As an industrial inspection operator, I want alignment to complete within a bounded time, so that the inspection throughput meets production line requirements.

#### Acceptance Criteria

1. WHEN alignment is executed on images up to 2048×2048 pixels, THE Alignment_Module SHALL complete the full Pipeline within 2.0 seconds on a standard CPU (no GPU required).
2. WHEN alignment is executed on images up to 1024×1024 pixels, THE Alignment_Module SHALL complete the full Pipeline within 0.5 seconds on a standard CPU.

---

### Requirement 9: Logging and Diagnostics

**User Story:** As a developer or inspection engineer, I want the alignment module to emit structured log messages, so that I can diagnose alignment failures without modifying source code.

#### Acceptance Criteria

1. THE Alignment_Module SHALL emit log messages using Python's standard `logging` module at the `DEBUG`, `WARNING`, and `ERROR` levels.
2. WHEN a fallback strategy is activated, THE Alignment_Module SHALL emit a `WARNING`-level log message stating the reason for the fallback and the strategy selected.
3. WHEN alignment completes successfully, THE Alignment_Module SHALL emit a `DEBUG`-level log message containing the strategy label, Alignment_Score, and Inlier_Ratio (if applicable).
4. IF an unrecoverable error occurs (e.g., both alignment strategies fail), THEN THE Alignment_Module SHALL emit an `ERROR`-level log message and raise a descriptive exception.

---

### Requirement 10: Edge Map Round-Trip Consistency (Parser/Serializer Analogy)

**User Story:** As a developer, I want the transformation application to be consistent with the transformation matrix, so that saving and reloading the matrix produces identical alignment results.

#### Acceptance Criteria

1. THE Alignment_Module SHALL provide a `apply_transform(edge_map, matrix)` function that applies a given Transformation_Matrix to an edge map.
2. FOR ALL valid Transformation_Matrix values, serializing the matrix to a NumPy `.npy` file and deserializing it SHALL produce a matrix that, when applied via `apply_transform`, yields a pixel-wise identical Aligned_Image (round-trip property).
3. FOR ALL valid edge maps and Transformation_Matrix values, `apply_transform(edge_map, matrix)` SHALL return an array of the same dtype and spatial dimensions as the CAD_Edge_Map.

---

### Requirement 11: Arbitrary Orientation Handling

**User Story:** As an inspection engineer, I want the system to correctly align parts regardless of their physical orientation on the inspection surface, so that I do not need to manually orient parts before inspection.

#### Acceptance Criteria

1. THE Alignment_Module SHALL handle input Real_Edge_Maps where the part is rotated by any angle in [0°, 360°) relative to the CAD_Edge_Map.
2. WHEN coarse alignment is executed, THE Alignment_Module SHALL produce a rotation-corrected intermediate image before feature-based fine alignment is applied.
3. IF the coarse rotation estimate results in a flipped orientation (180° ambiguity from PCA), THEN THE Alignment_Module SHALL attempt both orientations and select the one with the higher Alignment_Score.

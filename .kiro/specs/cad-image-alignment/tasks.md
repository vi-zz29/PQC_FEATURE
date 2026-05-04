# Implementation Plan: CAD-Image Alignment Module

## Overview

This plan implements a two-stage alignment pipeline (coarse contour-based + fine feature-based) with a robust fallback chain. The module accepts preprocessed binary edge maps and produces a geometric transformation, aligned image, and quality metrics. Implementation uses Python with OpenCV, NumPy, and Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure and module skeleton
  - Create `cad_image_alignment/` package directory with `__init__.py`
  - Create `cad_image_alignment/alignment.py` for core module
  - Create `cad_image_alignment/constants.py` for algorithm parameters
  - Set up Python logging configuration in module `__init__.py`
  - Create `tests/` directory with `__init__.py`
  - Create `requirements.txt` with dependencies: `numpy`, `opencv-python`, `hypothesis`, `pytest`
  - _Requirements: 9.1_

- [x] 2. Implement data models and input validation
  - [x] 2.1 Create AlignmentResult dataclass in `alignment.py`
    - Define dataclass with 6 fields: `aligned_image`, `transform_matrix`, `alignment_score`, `strategy`, `low_confidence`, `inlier_ratio`
    - Add type hints matching design specification
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.3_
  
  - [x] 2.2 Create ContourDescriptor dataclass (internal)
    - Define dataclass with fields: `contour`, `centroid`, `bbox`, `bbox_diagonal`, `pca_angle_deg`
    - _Requirements: 2.1, 2.2_
  
  - [x] 2.3 Implement `_validate_inputs()` function
    - Check dtype is uint8 for both inputs
    - Check ndim == 2 for both inputs
    - Check at least one non-zero pixel exists in each input
    - Raise descriptive ValueError identifying which input is invalid
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 2.4 Implement resolution mismatch handling in `_validate_inputs()`
    - Detect shape mismatch between CAD and real edge maps
    - Resize real_edge_map using cv2.INTER_NEAREST interpolation
    - Emit DEBUG-level log message when resizing occurs
    - _Requirements: 1.4, 9.1_
  
  - [ ]* 2.5 Write property test for input validation (Property 1)
    - **Property 1: Input Validation Rejects Invalid Inputs**
    - **Validates: Requirements 1.1, 1.2, 1.3**
    - Generate random arrays with varying dtypes, shapes, and zero-fill patterns
    - Assert ValueError raised for all invalid inputs
    - Run 200 iterations
  
  - [ ]* 2.6 Write unit tests for input validation edge cases
    - Test invalid dtype (float32, int32)
    - Test empty edge maps (all zeros)
    - Test resolution mismatch triggers resize
    - _Requirements: 1.3, 1.4_

- [x] 3. Implement contour extraction and coarse alignment
  - [x] 3.1 Implement `_extract_primary_contour()` function
    - Use cv2.findContours with RETR_EXTERNAL and CHAIN_APPROX_SIMPLE
    - Filter contours by minimum area (1% of image area)
    - Select contour with largest area
    - Return None if no valid contour found, log WARNING
    - Compute centroid using cv2.moments
    - Compute bounding box using cv2.boundingRect
    - Compute bbox diagonal length
    - Return ContourDescriptor or None
    - _Requirements: 2.1, 2.2, 6.2_
  
  - [x] 3.2 Implement PCA-based rotation estimation in `_compute_pca_angle()`
    - Reshape contour points to (N, 2) array
    - Compute mean and center points
    - Compute covariance matrix
    - Extract principal eigenvector using np.linalg.eigh
    - Compute angle using np.arctan2
    - Normalize angle to [0°, 360°) range
    - _Requirements: 2.5, 2.6_
  
  - [x] 3.3 Implement affine matrix construction in `_build_affine_matrix()`
    - Create scale matrix S (scale about real centroid)
    - Create rotation matrix R (rotate about scaled real centroid)
    - Create translation matrix T (translate to CAD centroid)
    - Compose as T @ R @ S to produce 3×3 affine matrix
    - Ensure bottom row is [0, 0, 1]
    - _Requirements: 2.3, 2.4_
  
  - [x] 3.4 Implement 180° ambiguity resolution in `_resolve_180_ambiguity()`
    - Generate two candidate angles: pca_angle and pca_angle + 180°
    - Build affine matrix for each candidate
    - Apply each transform using apply_transform()
    - Compute dilated IoU score for each result
    - Select candidate with higher score
    - _Requirements: 2.5, 11.3_
  
  - [x] 3.5 Implement `_compute_coarse_transform()` orchestration function
    - Extract primary contours from both CAD and real edge maps
    - Handle case where no contour found in real map (return None, log WARNING)
    - Compute scale factor from bbox diagonals
    - Compute PCA angle for both contours
    - Call `_resolve_180_ambiguity()` to get final coarse matrix
    - Return 3×3 affine matrix
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.8_
  
  - [ ]* 3.6 Write property test for primary contour selection (Property 8)
    - **Property 8: Primary Contour Selection Invariant**
    - **Validates: Requirements 6.2**
    - Generate random binary images with synthetic contours of known sizes
    - Assert largest contour above 1% threshold is selected
    - Assert None returned when no contour meets threshold
    - Run 100 iterations
  
  - [ ]* 3.7 Write unit tests for coarse alignment
    - Test PCA angle computation on known shapes (rectangle, L-shape)
    - Test 180° ambiguity resolution with flipped shapes
    - Test fallback when no contour found
    - _Requirements: 2.5, 2.8, 11.3_

- [x] 4. Checkpoint - Ensure coarse alignment tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement fine alignment with ORB and RANSAC
  - [x] 5.1 Implement ORB keypoint detection and matching in `_detect_and_match_features()`
    - Create cv2.ORB_create with parameters from constants (1000 features, 1.2 scale, 8 levels)
    - Detect keypoints and compute descriptors on both images
    - Create BFMatcher with NORM_HAMMING and crossCheck=True
    - Match descriptors and sort by distance
    - Return keypoints, descriptors, and matches
    - _Requirements: 3.1, 3.2_
  
  - [x] 5.2 Implement RANSAC homography estimation in `_estimate_homography()`
    - Check if match count >= MIN_MATCH_COUNT (8), return None if insufficient
    - Extract source and destination point arrays from matches
    - Call cv2.findHomography with RANSAC and 3.0 pixel threshold
    - Compute inlier_ratio from RANSAC mask
    - Return homography matrix, inlier_ratio, or None if failed
    - _Requirements: 3.3, 3.4, 3.5_
  
  - [x] 5.3 Implement homography scale validation in `_validate_homography()`
    - Extract upper-left 2×2 submatrix
    - Compute scale as sqrt(abs(det(H[:2, :2])))
    - Return True if scale in [0.5, 2.0], False otherwise
    - _Requirements: 6.3_
  
  - [x] 5.4 Implement `_compute_fine_transform()` orchestration function
    - Call `_detect_and_match_features()` on coarsely aligned image and CAD map
    - Call `_estimate_homography()` to get homography and inlier_ratio
    - Check match count fallback (< 8 matches → return None, log WARNING)
    - Check inlier ratio fallback (< 0.25 → return None, log WARNING)
    - Call `_validate_homography()` for scale check
    - If scale invalid, return None and log WARNING with scale value
    - Compose homography with coarse transform: H_total = H_fine @ M_coarse
    - Return H_total and inlier_ratio
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.3_
  
  - [ ]* 5.5 Write property test for homography scale validation (Property 6)
    - **Property 6: Homography Scale Validation Correctness**
    - **Validates: Requirements 6.3**
    - Generate random 3×3 matrices with known scale values (in-range and out-of-range)
    - Assert `_validate_homography()` returns True iff scale in [0.5, 2.0]
    - Run 200 iterations
  
  - [ ]* 5.6 Write unit tests for fine alignment
    - Test insufficient keypoints fallback (< 8 matches)
    - Test low inlier ratio fallback (< 0.25)
    - Test scale validation rejection
    - _Requirements: 3.5, 3.6, 6.3_

- [x] 6. Implement transformation application and quality assessment
  - [x] 6.1 Implement `apply_transform()` public function
    - Accept edge_map (uint8, 2D), matrix (float64, 3×3), optional output_shape
    - Use cv2.warpPerspective with INTER_LINEAR and BORDER_CONSTANT
    - Default output_shape to input edge_map shape if not provided
    - Return uint8 array with specified output shape
    - _Requirements: 10.1, 10.3, 4.1_
  
  - [x] 6.2 Implement dilated IoU computation in `_compute_alignment_score()`
    - Create 3×3 rectangular structuring element
    - Dilate both aligned_image and cad_edge_map
    - Compute intersection as logical_and of dilated masks
    - Compute union as logical_or of dilated masks
    - Return intersection / union (or 0.0 if union is 0)
    - _Requirements: 5.1, 5.2_
  
  - [ ]* 6.3 Write property test for apply_transform invariants (Property 2)
    - **Property 2: apply_transform Output Shape and Dtype Invariant**
    - **Validates: Requirements 10.3, 4.1**
    - Generate random uint8 2D arrays and random 3×3 float64 matrices
    - Assert output has same shape and dtype as input
    - Run 100 iterations
  
  - [ ]* 6.4 Write property test for dilated IoU bounds (Property 7)
    - **Property 7: Dilated IoU Score Bounds**
    - **Validates: Requirements 5.1, 4.3**
    - Generate random pairs of same-shape binary uint8 arrays
    - Assert score is in [0.0, 1.0]
    - Run 100 iterations
  
  - [ ]* 6.5 Write unit tests for transformation application
    - Test identity transform produces unchanged image
    - Test output shape matches CAD edge map shape
    - _Requirements: 10.3, 4.1_

- [x] 7. Implement main align() function with fallback chain
  - [x] 7.1 Implement fallback chain logic in `align()`
    - Call `_validate_inputs()` and handle resolution mismatch
    - Call `_compute_coarse_transform()` to get M_coarse
    - If M_coarse is None (no contour), set strategy="identity", use np.eye(3)
    - If M_coarse exists, apply it to get coarsely_aligned image
    - Call `_compute_fine_transform()` with coarsely_aligned and CAD map
    - If fine transform succeeds, set strategy="homography", use H_total
    - If fine transform fails, set strategy="affine_coarse_only", use M_coarse
    - If both fail, set strategy="identity", use np.eye(3)
    - _Requirements: 3.5, 3.6, 6.3, 2.8_
  
  - [x] 7.2 Implement final transformation application and scoring in `align()`
    - Apply final transformation matrix to original real_edge_map
    - Call `_compute_alignment_score()` to get alignment_score
    - Set low_confidence = True if alignment_score < 0.30
    - If low_confidence, log WARNING with score value
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 7.3 Implement logging for all decision points in `align()`
    - Log DEBUG on successful alignment with strategy, score, inlier_ratio
    - Log WARNING on each fallback with reason and selected strategy
    - Log ERROR if unrecoverable error occurs
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 7.4 Construct and return AlignmentResult in `align()`
    - Populate all 6 fields: aligned_image, transform_matrix, alignment_score, strategy, low_confidence, inlier_ratio
    - Ensure result is always returned even when low_confidence=True
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.3, 5.4_
  
  - [ ]* 7.5 Write property test for AlignmentResult structural invariants (Property 9)
    - **Property 9: AlignmentResult Structural Invariants**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 5.3, 5.4**
    - Generate random valid synthetic edge map pairs
    - Assert strategy is one of {"homography", "affine_coarse_only", "identity"}
    - Assert transform_matrix shape is (3, 3) and dtype is float64
    - Assert alignment_score is in [0.0, 1.0]
    - Assert aligned_image has same shape as cad_edge_map and dtype uint8
    - Assert low_confidence is True iff alignment_score < 0.30
    - Assert result is always returned (no exception)
    - Run 100 iterations
  
  - [ ]* 7.6 Write unit tests for fallback chain
    - Test identity fallback when no contour found
    - Test affine_coarse_only fallback when fine alignment fails
    - Test homography strategy when fine alignment succeeds
    - Test low_confidence flag set correctly
    - Test result returned even with low_confidence=True
    - _Requirements: 2.8, 3.5, 3.6, 5.3, 5.4_

- [x] 8. Checkpoint - Ensure main alignment pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement transformation invertibility and composition properties
  - [ ]* 9.1 Write property test for transformation invertibility (Property 4)
    - **Property 4: Transformation Invertibility**
    - **Validates: Requirements 7.1, 7.2**
    - Generate matrices from align() output on random synthetic edge map pairs
    - Compute matrix inverse using np.linalg.inv
    - Apply matrix then inverse to real_edge_map
    - Assert non-zero pixel positions match original within 2 pixels tolerance
    - Run 100 iterations
  
  - [ ]* 9.2 Write property test for transformation composition (Property 5)
    - **Property 5: Transformation Composition Consistency**
    - **Validates: Requirements 7.3**
    - Generate pairs of random non-singular 3×3 matrices
    - Compute composed matrix T2 @ T1
    - Apply composed matrix to edge map
    - Apply T1 then T2 sequentially to same edge map
    - Assert pixel-wise identical results
    - Run 100 iterations
  
  - [ ]* 9.3 Write property test for serialization round-trip (Property 3)
    - **Property 3: Serialization Round-Trip**
    - **Validates: Requirements 10.2**
    - Generate random non-singular 3×3 float64 matrices
    - Save matrix to .npy file using np.save
    - Load matrix from .npy file using np.load
    - Apply both original and loaded matrices to synthetic edge map
    - Assert pixel-wise identical results
    - Run 100 iterations

- [ ] 10. Implement arbitrary orientation handling
  - [ ]* 10.1 Write property test for arbitrary orientation handling (Property 10)
    - **Property 10: Arbitrary Orientation Handling**
    - **Validates: Requirements 11.1, 2.6**
    - Generate CAD edge map with known shape
    - Rotate by random angle in [0°, 360°) using cv2.warpAffine
    - Call align() with CAD and rotated real edge map
    - Assert no exception raised
    - Assert strategy ≠ "identity" (some alignment was computed)
    - Run 100 iterations
  
  - [ ]* 10.2 Write unit test for 180° rotation case
    - Create synthetic bracket-like shape
    - Rotate by exactly 180°
    - Call align() and verify correct alignment
    - _Requirements: 11.3_

- [x] 11. Create synthetic test fixtures
  - [x] 11.1 Implement bracket-like shape generator in `tests/fixtures.py`
    - Draw rounded rectangle with square cutout and three circles
    - Apply Canny edge detection to produce binary edge map
    - Return uint8 edge map
    - _Requirements: Testing Strategy_
  
  - [x] 11.2 Implement rotated variant generator in `tests/fixtures.py`
    - Accept edge map and rotation angle
    - Apply cv2.warpAffine with known rotation
    - Return rotated edge map and ground-truth transformation matrix
    - _Requirements: Testing Strategy_
  
  - [x] 11.3 Implement noisy variant generator in `tests/fixtures.py`
    - Accept edge map and noise level
    - Add salt-and-pepper noise to simulate metallic reflections
    - Return noisy edge map
    - _Requirements: 6.1, Testing Strategy_

- [ ] 12. Implement integration tests with synthetic data
  - [ ]* 12.1 Write integration test with bracket-like shape at multiple orientations
    - Generate bracket fixture
    - Test alignment at 0°, 45°, 90°, 135°, 180°, 270°
    - Assert alignment_score > 0.70 for all orientations
    - Assert strategy is "homography" or "affine_coarse_only"
    - _Requirements: 11.1, 2.6_
  
  - [ ]* 12.2 Write integration test with noisy input
    - Generate bracket fixture with salt-and-pepper noise
    - Test alignment with noisy real edge map
    - Assert alignment completes without exception
    - Assert RANSAC suppresses spurious matches
    - _Requirements: 6.1, 6.2_
  
  - [ ]* 12.3 Write integration test for logging output
    - Capture log messages during alignment
    - Verify WARNING emitted on fallback
    - Verify DEBUG emitted on success with strategy and score
    - _Requirements: 9.2, 9.3_

- [ ] 13. Implement performance tests
  - [ ]* 13.1 Write performance test for 2048×2048 images
    - Generate 2048×2048 synthetic edge map pair
    - Measure execution time using time.perf_counter
    - Assert completion within 2.0 seconds
    - Mark with @pytest.mark.slow
    - _Requirements: 8.1_
  
  - [ ]* 13.2 Write performance test for 1024×1024 images
    - Generate 1024×1024 synthetic edge map pair
    - Measure execution time using time.perf_counter
    - Assert completion within 0.5 seconds
    - Mark with @pytest.mark.slow
    - _Requirements: 8.2_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Run full test suite including property tests and performance tests
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- Checkpoints ensure incremental validation at logical breaks
- All 10 correctness properties from the design are covered by property tests
- Synthetic test fixtures eliminate dependency on real camera images

# Design Document: CAD-Image Alignment Module

## Overview

The CAD-Image Alignment Module computes a geometric transformation that maps a real camera-derived edge map onto a CAD reference edge map, enabling downstream mismatch detection between a manufactured part and its design specification.

The module operates on pre-processed binary edge maps (uint8, values in {0, 255}) and produces an aligned image, a 3×3 transformation matrix, a numeric quality score, and a strategy label. It must handle arbitrary part orientations (full 360°), metallic surface noise, and scale variation, all within tight latency budgets on CPU hardware.

### Design Goals

- **Correctness**: Produce geometrically accurate alignments across all valid orientations and scales.
- **Robustness**: Degrade gracefully under noise, reflections, and poor feature coverage via a well-defined fallback chain.
- **Transparency**: Emit structured log messages at every decision point so failures are diagnosable without code changes.
- **Composability**: Return a serializable 3×3 matrix that downstream modules can invert, compose, or persist.

---

## Architecture

The module is structured as a two-stage pipeline with a deterministic fallback chain.

```
┌─────────────────────────────────────────────────────────────────┐
│                     AlignmentModule.align()                     │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  Input       │    │  Stage 1: Coarse Alignment           │  │
│  │  Validation  │───▶│  - Extract Primary_Contour           │  │
│  │              │    │  - PCA rotation estimation (0-360°)  │  │
│  └──────────────┘    │  - 180° ambiguity resolution         │  │
│                      │  - Scale + translation from bbox     │  │
│                      │  - Produce coarse Affine_Transform   │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │ coarsely aligned image    │
│                      ┌──────────────▼───────────────────────┐  │
│                      │  Stage 2: Fine Alignment (ORB)        │  │
│                      │  - ORB keypoint detection             │  │
│                      │  - Brute-force Hamming matching       │  │
│                      │  - RANSAC homography estimation       │  │
│                      │  - Scale validation [0.5, 2.0]        │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │                           │
│                      ┌──────────────▼───────────────────────┐  │
│                      │  Fallback Chain                       │  │
│                      │  homography → affine_coarse_only      │  │
│                      │            → identity                 │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │                           │
│                      ┌──────────────▼───────────────────────┐  │
│                      │  Quality Assessment                   │  │
│                      │  - Dilated IoU scoring                │  │
│                      │  - low_confidence flag                │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │                           │
│                      ┌──────────────▼───────────────────────┐  │
│                      │  AlignmentResult (dataclass)          │  │
│                      └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Fallback Chain

```
Attempt homography (fine alignment)
  ├─ < 8 keypoint matches          → affine_coarse_only
  ├─ inlier_ratio < 0.25           → affine_coarse_only
  ├─ scale outside [0.5, 2.0]      → affine_coarse_only
  └─ no contour in real image      → identity (skip coarse too)
```

---

## Components and Interfaces

### Public API

```python
def align(
    cad_edge_map: np.ndarray,   # uint8, shape (H, W), values in {0, 255}
    real_edge_map: np.ndarray,  # uint8, shape (H, W), values in {0, 255}
) -> AlignmentResult:
    """
    Compute the geometric transformation that maps real_edge_map onto
    cad_edge_map and return the aligned image plus metadata.

    Raises:
        ValueError: if either input is invalid (wrong dtype, shape, or empty).
    """
```

```python
def apply_transform(
    edge_map: np.ndarray,       # uint8, shape (H, W)
    matrix: np.ndarray,         # float64, shape (3, 3)
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Apply a 3×3 homography matrix to edge_map using cv2.warpPerspective.
    Returns a uint8 array with the same spatial dimensions as edge_map
    (or output_shape if provided).

    This function is the single point of truth for transformation application,
    ensuring that saving/loading the matrix and re-calling apply_transform
    produces pixel-identical results.
    """
```

### AlignmentResult Dataclass

```python
@dataclass
class AlignmentResult:
    aligned_image: np.ndarray       # uint8, shape (H, W) — same dims as CAD
    transform_matrix: np.ndarray    # float64, shape (3, 3)
    alignment_score: float          # float in [0.0, 1.0]
    strategy: str                   # "homography" | "affine_coarse_only" | "identity"
    low_confidence: bool            # True if alignment_score < 0.30
    inlier_ratio: float | None      # RANSAC inlier ratio, None if not computed
```

### Internal Components

| Component | Responsibility |
|---|---|
| `_validate_inputs` | dtype/shape/empty checks; resize if resolutions differ |
| `_extract_primary_contour` | find largest contour above 1% area threshold |
| `_compute_coarse_transform` | PCA rotation + scale + translation → 3×3 affine matrix |
| `_resolve_180_ambiguity` | score both 0° and 180° orientations, pick higher IoU |
| `_compute_fine_transform` | ORB detection, BF Hamming matching, RANSAC homography |
| `_validate_homography` | scale check [0.5, 2.0]; returns bool |
| `_compute_alignment_score` | dilated IoU between aligned image and CAD edge map |
| `apply_transform` | public utility; wraps `cv2.warpPerspective` |

---

## Data Models

### Input Constraints

| Field | Type | Constraint |
|---|---|---|
| `cad_edge_map` | `np.ndarray` | dtype=uint8, ndim=2, has non-zero pixels |
| `real_edge_map` | `np.ndarray` | dtype=uint8, ndim=2, has non-zero pixels |

If `real_edge_map.shape != cad_edge_map.shape`, the real map is resized with `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` before any processing.

### Transformation Matrix

A 3×3 `float64` NumPy array representing a projective (homography) transform in homogeneous coordinates:

```
| r00  r01  tx |
| r10  r11  ty |
|  p0   p1   1 |
```

For affine transforms (coarse stage), the bottom row is `[0, 0, 1]`.

The identity matrix `np.eye(3, dtype=np.float64)` is the fallback when all alignment strategies fail.

### Contour Descriptor (internal)

```python
@dataclass
class ContourDescriptor:
    contour: np.ndarray         # OpenCV contour array, shape (N, 1, 2)
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]   # x, y, w, h
    bbox_diagonal: float
    pca_angle_deg: float        # principal axis angle in [0°, 360°)
```

### Algorithm Parameters (module-level constants)

```python
# Coarse alignment
MIN_CONTOUR_AREA_FRACTION = 0.01   # ignore contours < 1% of image area

# Fine alignment
ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
MIN_MATCH_COUNT = 8
RANSAC_REPROJ_THRESHOLD = 3.0      # pixels
MIN_INLIER_RATIO = 0.25

# Homography validation
SCALE_MIN = 0.5
SCALE_MAX = 2.0

# Quality assessment
DILATION_KERNEL_SIZE = 3           # 3×3 structuring element
LOW_CONFIDENCE_THRESHOLD = 0.30
```

---

## Algorithms

### Stage 1: Coarse Alignment

#### 1a. Primary Contour Extraction

```
contours = cv2.findContours(edge_map, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
min_area = image_area * MIN_CONTOUR_AREA_FRACTION
valid = [c for c in contours if cv2.contourArea(c) >= min_area]
primary = max(valid, key=cv2.contourArea)
```

If `valid` is empty, log a WARNING and return `None`.

#### 1b. PCA Rotation Estimation

The contour points are treated as a 2D point cloud. PCA finds the principal axis:

```
pts = contour.reshape(-1, 2).astype(float)
mean = pts.mean(axis=0)
centered = pts - mean
cov = centered.T @ centered
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# eigenvector corresponding to largest eigenvalue = principal axis
principal = eigenvectors[:, np.argmax(eigenvalues)]
angle_rad = np.arctan2(principal[1], principal[0])
angle_deg = np.degrees(angle_rad)  # in (-180°, 180°]
```

`np.arctan2` returns values in `(-180°, 180°]`. We normalize to `[0°, 360°)` by adding 360° if negative.

#### 1c. 180° Ambiguity Resolution

PCA cannot distinguish between a direction and its opposite. Both candidate rotations are scored:

```
for candidate_angle in [pca_angle_deg, pca_angle_deg + 180.0]:
    M = build_affine(scale, candidate_angle, translation)
    warped = apply_transform(real_edge_map, M, output_shape=cad_shape)
    score = dilated_iou(warped, cad_edge_map)
    candidates.append((score, M))
best_M = max(candidates, key=lambda x: x[0])[1]
```

#### 1d. Coarse Affine Matrix Construction

```
# Step 1: scale about real centroid
S = scale_matrix(scale_factor, cx_real, cy_real)

# Step 2: rotate about real centroid (after scaling)
R = rotation_matrix(-pca_angle_deg, cx_real * scale_factor, cy_real * scale_factor)

# Step 3: translate centroid to CAD centroid
T = translation_matrix(cx_cad - cx_real_scaled, cy_cad - cy_real_scaled)

M_coarse = T @ R @ S   # 3×3 affine, bottom row = [0, 0, 1]
```

### Stage 2: Fine Alignment (ORB + RANSAC)

```
orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, scaleFactor=ORB_SCALE_FACTOR,
                     nlevels=ORB_N_LEVELS)
kp1, des1 = orb.detectAndCompute(coarsely_aligned, None)
kp2, des2 = orb.detectAndCompute(cad_edge_map, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

if len(matches) < MIN_MATCH_COUNT:
    → fallback to affine_coarse_only

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

H, mask = cv2.findHomography(src_pts, dst_pts,
                              cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
inlier_ratio = mask.sum() / len(mask)

if inlier_ratio < MIN_INLIER_RATIO:
    → fallback to affine_coarse_only

if not _validate_homography(H):
    → fallback to affine_coarse_only

# Compose with coarse transform: H_total = H_fine @ M_coarse
H_total = H @ M_coarse   # both are 3×3
```

### Homography Scale Validation

The scale component is extracted from the homography's upper-left 2×2 submatrix:

```
scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
return SCALE_MIN <= scale <= SCALE_MAX
```

### Quality Assessment: Dilated IoU

```
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_aligned = cv2.dilate(aligned_image, kernel)
dilated_cad     = cv2.dilate(cad_edge_map,  kernel)

intersection = np.logical_and(dilated_aligned > 0, dilated_cad > 0).sum()
union        = np.logical_or( dilated_aligned > 0, dilated_cad > 0).sum()

score = intersection / union if union > 0 else 0.0
```

---

## Data Flow

```
cad_edge_map ──────────────────────────────────────────────────────┐
                                                                    │
real_edge_map ──▶ _validate_inputs ──▶ (resize if needed)          │
                        │                                           │
                        ▼                                           │
              _extract_primary_contour(real)                        │
              _extract_primary_contour(cad) ◀──────────────────────┘
                        │
                        ▼
              _compute_coarse_transform
              (PCA angle, scale, translation)
                        │
                        ▼
              _resolve_180_ambiguity
              → M_coarse (3×3 affine)
                        │
                        ▼
              apply_transform(real, M_coarse)
              → coarsely_aligned
                        │
                        ▼
              _compute_fine_transform(coarsely_aligned, cad)
              → H_fine (3×3 homography) or None
                        │
                        ▼
              Fallback chain resolution
              → H_total (3×3), strategy label
                        │
                        ▼
              apply_transform(real, H_total)
              → aligned_image
                        │
                        ▼
              _compute_alignment_score(aligned_image, cad)
              → alignment_score, low_confidence
                        │
                        ▼
              AlignmentResult(aligned_image, H_total, score,
                              strategy, low_confidence, inlier_ratio)
```

---

## Error Handling

| Condition | Behavior |
|---|---|
| Input ndim < 2 or wrong dtype | Raise `ValueError` with field name |
| Input has no non-zero pixels | Raise `ValueError` with field name |
| Resolution mismatch | Resize real map silently (DEBUG log) |
| No valid contour in real map | Log WARNING; skip coarse stage; attempt fine alignment on original |
| < 8 keypoint matches | Log WARNING; strategy = `affine_coarse_only` |
| Inlier ratio < 0.25 | Log WARNING with ratio; strategy = `affine_coarse_only` |
| Homography scale out of [0.5, 2.0] | Log WARNING with scale value; strategy = `affine_coarse_only` |
| Alignment score < 0.30 | Log WARNING with score; set `low_confidence = True`; still return result |
| Both coarse and fine fail | Log ERROR; strategy = `identity`; return identity transform |
| Unrecoverable exception | Log ERROR; re-raise with descriptive message |

The module never silently swallows errors that indicate a programming mistake (e.g., wrong array shapes passed to OpenCV). Only expected operational failures (no contours, too few matches) are handled gracefully.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Input Validation Rejects Invalid Inputs

*For any* array that is not a 2D uint8 array with at least one non-zero pixel, passing it as either the `cad_edge_map` or `real_edge_map` argument to `align()` SHALL raise a `ValueError` with a message identifying which input is invalid.

**Validates: Requirements 1.1, 1.2, 1.3**

---

### Property 2: apply_transform Output Shape and Dtype Invariant

*For any* valid uint8 2D edge map and any valid 3×3 float64 transformation matrix, `apply_transform(edge_map, matrix)` SHALL return a uint8 array with the same spatial dimensions as the input edge map.

**Validates: Requirements 10.3, 4.1**

---

### Property 3: Serialization Round-Trip

*For any* valid 3×3 float64 transformation matrix produced by `align()`, serializing it to a NumPy `.npy` file, deserializing it, and applying it via `apply_transform` SHALL produce a pixel-wise identical result to applying the original matrix directly.

**Validates: Requirements 10.2**

---

### Property 4: Transformation Invertibility

*For any* valid Transformation_Matrix produced by `align()`, applying the matrix and then its inverse to a Real_Edge_Map SHALL produce an image whose non-zero pixel positions match the original Real_Edge_Map non-zero pixel positions within a tolerance of 2 pixels.

**Validates: Requirements 7.1, 7.2**

---

### Property 5: Transformation Composition Consistency

*For any* two valid 3×3 transformation matrices T1 and T2, applying the composed matrix `T2 @ T1` to an edge map SHALL produce a pixel-wise identical result to applying T1 first and then T2 sequentially.

**Validates: Requirements 7.3**

---

### Property 6: Homography Scale Validation Correctness

*For any* 3×3 matrix, `_validate_homography` SHALL return `True` if and only if the scale component — computed as `sqrt(abs(det(H[:2, :2])))` — is within the closed interval [0.5, 2.0].

**Validates: Requirements 6.3**

---

### Property 7: Dilated IoU Score Bounds

*For any* pair of binary uint8 edge maps of the same spatial dimensions, `_compute_alignment_score` SHALL return a float value in [0.0, 1.0].

**Validates: Requirements 5.1, 4.3**

---

### Property 8: Primary Contour Selection Invariant

*For any* binary edge map containing at least one contour with area ≥ 1% of the total image area, `_extract_primary_contour` SHALL return the contour with the largest area among all contours that meet the 1% threshold, and SHALL return `None` if no contour meets the threshold.

**Validates: Requirements 6.2**

---

### Property 9: AlignmentResult Structural Invariants

*For any* valid pair of input edge maps, the `AlignmentResult` returned by `align()` SHALL satisfy all of the following simultaneously:
- `strategy` is one of `{"homography", "affine_coarse_only", "identity"}`
- `transform_matrix` has shape `(3, 3)` and dtype `float64`
- `alignment_score` is a float in `[0.0, 1.0]`
- `aligned_image` has the same spatial dimensions as `cad_edge_map` and dtype `uint8`
- `low_confidence` is `True` if and only if `alignment_score < 0.30`
- A result is always returned (no exception raised) even when `low_confidence` is `True`

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 5.3, 5.4**

---

### Property 10: Arbitrary Orientation Handling

*For any* valid CAD edge map and any Real_Edge_Map produced by rotating that CAD edge map by an angle drawn uniformly from [0°, 360°), `align()` SHALL complete without raising an exception and SHALL return a result with `strategy` ≠ `"identity"` (i.e., some alignment was computed).

**Validates: Requirements 11.1, 2.6**

---

## Testing Strategy

### Dual Testing Approach

Both unit/example-based tests and property-based tests are used. Unit tests cover specific scenarios and integration points; property tests verify universal invariants across randomized inputs.

### Property-Based Testing Library

Use **Hypothesis** (`hypothesis` package) with `hypothesis.extra.numpy` for array generation. Each property test runs a minimum of 100 iterations (configured via `@settings(max_examples=100)`).

Each property test is tagged with a comment in the format:
```
# Feature: cad-image-alignment, Property N: <property_text>
```

### Unit / Example-Based Tests

| Test | What it verifies |
|---|---|
| `test_invalid_input_dtype` | ValueError on non-uint8 input (Req 1.3) |
| `test_invalid_input_empty` | ValueError on all-zero edge map (Req 1.3) |
| `test_resolution_mismatch_resize` | Real map resized to CAD shape before processing (Req 1.4) |
| `test_coarse_only_fallback_few_keypoints` | strategy = affine_coarse_only when < 8 matches (Req 3.5) |
| `test_coarse_only_fallback_low_inlier` | strategy = affine_coarse_only when inlier_ratio < 0.25 (Req 3.6) |
| `test_identity_fallback_no_contour` | strategy = identity when no contour found (Req 2.8) |
| `test_low_confidence_flag` | low_confidence=True when score < 0.30 (Req 5.3) |
| `test_low_confidence_still_returns` | result returned even when low_confidence=True (Req 5.4) |
| `test_180_degree_rotation` | correct alignment of a part rotated 180° (Req 11.3) |
| `test_logging_fallback_warning` | WARNING emitted on fallback (Req 9.2) |
| `test_logging_success_debug` | DEBUG emitted on success with score and strategy (Req 9.3) |
| `test_strategy_label_values` | strategy is one of the three valid strings (Req 4.4) |

### Property-Based Tests

| Property | Hypothesis Strategy | Iterations |
|---|---|---|
| P1: Input validation rejects invalid inputs | Random arrays with varying dtypes, shapes, and zero-fill | 200 |
| P2: apply_transform shape/dtype invariant | Random uint8 2D arrays + random 3×3 float64 matrices | 100 |
| P3: Serialization round-trip | Random non-singular 3×3 float64 matrices + synthetic edge maps | 100 |
| P4: Transformation invertibility | Matrices from `align()` output on random synthetic edge map pairs | 100 |
| P5: Composition consistency | Pairs of random non-singular 3×3 matrices + synthetic edge maps | 100 |
| P6: Scale validation correctness | Random 3×3 matrices with known scale values (in-range and out-of-range) | 200 |
| P7: Dilated IoU bounds | Random pairs of same-shape binary uint8 arrays | 100 |
| P8: Primary contour area filter | Random binary images with synthetic contours of known sizes | 100 |
| P9: AlignmentResult structural invariants | Random valid synthetic edge map pairs | 100 |
| P10: Arbitrary orientation handling | CAD edge map + rotated variants at random angles in [0°, 360°) | 100 |

### Performance Tests

| Test | Condition | Target |
|---|---|---|
| `test_perf_2048` | 2048×2048 synthetic edge maps | < 2.0 s |
| `test_perf_1024` | 1024×1024 synthetic edge maps | < 0.5 s |

Performance tests use `time.perf_counter` and are marked with `@pytest.mark.slow` to allow exclusion from fast CI runs.

### Test Data

Synthetic test fixtures are generated programmatically:
- **Bracket-like shape**: draw a rounded rectangle with a square cutout and three circles onto a blank canvas, then apply Canny to produce an edge map.
- **Rotated variants**: apply `cv2.warpAffine` with known rotation angles (0°, 45°, 90°, 135°, 180°, 270°) to produce ground-truth pairs.
- **Noisy variants**: add random salt-and-pepper noise to simulate metallic reflections.

No real camera images are required for unit or property tests.

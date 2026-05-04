"""
Core alignment module implementing the two-stage alignment pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContourDescriptor:
    """
    Internal descriptor for a contour extracted from an edge map.
    
    Attributes:
        contour: OpenCV contour array, shape (N, 1, 2)
        centroid: Geometric center (x, y) of the contour
        bbox: Axis-aligned bounding box (x, y, w, h)
        bbox_diagonal: Length of the bounding box diagonal
        pca_angle_deg: Principal axis angle in [0°, 360°)
    """
    contour: np.ndarray
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    bbox_diagonal: float
    pca_angle_deg: float


@dataclass
class AlignmentResult:
    """
    Result of the alignment operation.
    
    Attributes:
        aligned_image: The transformed real edge map aligned to the CAD edge map (uint8, 2D)
        transform_matrix: The 3×3 transformation matrix (float64)
        alignment_score: Quality score in [0.0, 1.0] based on dilated IoU
        strategy: Alignment strategy used ("homography", "affine_coarse_only", or "identity")
        low_confidence: True if alignment_score < 0.30
        inlier_ratio: RANSAC inlier ratio (None if RANSAC not used)
    """
    aligned_image: np.ndarray
    transform_matrix: np.ndarray
    alignment_score: float
    strategy: str
    low_confidence: bool
    inlier_ratio: Optional[float]


def _validate_inputs(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> np.ndarray:
    """
    Validate input edge maps for correct dtype, shape, and content.
    Resize real_edge_map if resolution mismatch is detected.
    
    Args:
        cad_edge_map: Binary edge map from CAD model
        real_edge_map: Binary edge map from camera image
    
    Returns:
        The potentially resized real_edge_map (same shape as cad_edge_map)
    
    Raises:
        ValueError: If either input is invalid (wrong dtype, ndim, or empty)
    """
    # Check dtype for CAD edge map
    if cad_edge_map.dtype != np.uint8:
        raise ValueError(
            f"cad_edge_map has invalid dtype {cad_edge_map.dtype}, expected uint8"
        )
    
    # Check dtype for real edge map
    if real_edge_map.dtype != np.uint8:
        raise ValueError(
            f"real_edge_map has invalid dtype {real_edge_map.dtype}, expected uint8"
        )
    
    # Check ndim for CAD edge map
    if cad_edge_map.ndim != 2:
        raise ValueError(
            f"cad_edge_map has invalid ndim {cad_edge_map.ndim}, expected 2"
        )
    
    # Check ndim for real edge map
    if real_edge_map.ndim != 2:
        raise ValueError(
            f"real_edge_map has invalid ndim {real_edge_map.ndim}, expected 2"
        )
    
    # Check for at least one non-zero pixel in CAD edge map
    if not np.any(cad_edge_map):
        raise ValueError(
            "cad_edge_map is empty (contains no non-zero pixels)"
        )
    
    # Check for at least one non-zero pixel in real edge map
    if not np.any(real_edge_map):
        raise ValueError(
            "real_edge_map is empty (contains no non-zero pixels)"
        )
    
    # Handle resolution mismatch
    if real_edge_map.shape != cad_edge_map.shape:
        logger.debug(
            f"Resolution mismatch detected: real_edge_map shape {real_edge_map.shape} "
            f"!= cad_edge_map shape {cad_edge_map.shape}. Resizing real_edge_map."
        )
        # Resize real_edge_map to match CAD edge map shape
        # Note: cv2.resize expects (width, height) but shape is (height, width)
        real_edge_map = cv2.resize(
            real_edge_map,
            (cad_edge_map.shape[1], cad_edge_map.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    return real_edge_map


def _compute_pca_angle(contour: np.ndarray) -> float:
    """
    Compute the principal axis angle of a contour using PCA.
    
    Args:
        contour: OpenCV contour array, shape (N, 1, 2)
    
    Returns:
        Principal axis angle in degrees, normalized to [0°, 360°)
    """
    # Reshape contour points to (N, 2) array
    pts = contour.reshape(-1, 2).astype(np.float64)
    
    # Compute mean and center points
    mean = pts.mean(axis=0)
    centered = pts - mean
    
    # Compute covariance matrix
    cov = centered.T @ centered
    
    # Extract principal eigenvector using np.linalg.eigh
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Principal eigenvector corresponds to largest eigenvalue (last column)
    principal = eigenvectors[:, -1]
    
    # Compute angle using np.arctan2
    angle_rad = np.arctan2(principal[1], principal[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize angle to [0°, 360°) range
    if angle_deg < 0:
        angle_deg += 360.0
    
    return angle_deg


def _extract_primary_contour(edge_map: np.ndarray) -> Optional[ContourDescriptor]:
    """
    Extract the primary (largest) contour from a binary edge map.
    
    Args:
        edge_map: Binary edge map (uint8, 2D, values in {0, 255})
    
    Returns:
        ContourDescriptor with contour, centroid, bbox, bbox_diagonal, and pca_angle_deg,
        or None if no valid contour found
    """
    from .constants import MIN_CONTOUR_AREA_FRACTION
    
    # Find all external contours
    contours, _ = cv2.findContours(
        edge_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Calculate minimum area threshold (1% of image area)
    image_area = edge_map.shape[0] * edge_map.shape[1]
    min_area = image_area * MIN_CONTOUR_AREA_FRACTION
    
    # Filter contours by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    # Return None if no valid contour found
    if not valid_contours:
        logger.warning(
            f"No contour found with area >= {MIN_CONTOUR_AREA_FRACTION * 100}% "
            f"of image area ({min_area:.1f} pixels)"
        )
        return None
    
    # Select contour with largest area
    primary_contour = max(valid_contours, key=cv2.contourArea)
    
    # Compute centroid using cv2.moments
    M = cv2.moments(primary_contour)
    if M["m00"] == 0:
        logger.warning("Primary contour has zero area moment, cannot compute centroid")
        return None
    
    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]
    centroid = (centroid_x, centroid_y)
    
    # Compute bounding box using cv2.boundingRect
    x, y, w, h = cv2.boundingRect(primary_contour)
    bbox = (x, y, w, h)
    
    # Compute bbox diagonal length
    bbox_diagonal = np.sqrt(w**2 + h**2)
    
    # Compute PCA angle
    pca_angle_deg = _compute_pca_angle(primary_contour)
    
    return ContourDescriptor(
        contour=primary_contour,
        centroid=centroid,
        bbox=bbox,
        bbox_diagonal=bbox_diagonal,
        pca_angle_deg=pca_angle_deg
    )


def _build_affine_matrix(
    scale: float,
    angle_deg: float,
    real_centroid: tuple[float, float],
    cad_centroid: tuple[float, float],
) -> np.ndarray:
    """
    Build a 3×3 affine transformation matrix that scales, rotates, and translates
    the real edge map to align with the CAD edge map.
    
    The transformation is composed as T @ R @ S:
    - S: scale about real centroid
    - R: rotate about real centroid (after scaling)
    - T: translate real centroid to CAD centroid
    
    Args:
        scale: Scale factor (CAD bbox diagonal / real bbox diagonal)
        angle_deg: Rotation angle in degrees (counterclockwise)
        real_centroid: (x, y) centroid of real edge map contour
        cad_centroid: (x, y) centroid of CAD edge map contour
    
    Returns:
        3×3 affine transformation matrix with bottom row [0, 0, 1]
    """
    cx_real, cy_real = real_centroid
    cx_cad, cy_cad = cad_centroid
    
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Build the combined transformation matrix directly
    # The transformation applies: scale about real centroid, then rotate about real centroid,
    # then translate real centroid to CAD centroid
    
    # Combined scale and rotation about real centroid:
    # M = T(cx_real, cy_real) @ R(angle) @ S(scale) @ T(-cx_real, -cy_real)
    # This simplifies to:
    # [s*cos  -s*sin  cx_real*(1-s*cos) + cy_real*s*sin]
    # [s*sin   s*cos  cy_real*(1-s*cos) - cx_real*s*sin]
    # [0       0      1                                 ]
    
    s_cos = scale * cos_a
    s_sin = scale * sin_a
    
    # After scale and rotation, the real centroid stays at (cx_real, cy_real)
    # Now we need to translate it to (cx_cad, cy_cad)
    tx = cx_cad - cx_real
    ty = cy_cad - cy_real
    
    # Build the final matrix
    M = np.array([
        [s_cos, -s_sin, cx_real * (1 - s_cos) + cy_real * s_sin + tx],
        [s_sin, s_cos, cy_real * (1 - s_cos) - cx_real * s_sin + ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    return M


def _compute_alignment_score(
    aligned_image: np.ndarray,
    cad_edge_map: np.ndarray,
) -> float:
    """
    Compute the alignment quality score using raw IoU with angle penalty.
    
    Args:
        aligned_image: The transformed real edge map (uint8, 2D)
        cad_edge_map: The reference CAD edge map (uint8, 2D)
    
    Returns:
        Alignment score in [0.0, 1.0] based on raw IoU (more accurate than dilated)
    """
    # Compute raw intersection over union (more accurate than dilated IoU)
    intersection = np.logical_and(aligned_image > 0, cad_edge_map > 0).sum()
    union = np.logical_or(aligned_image > 0, cad_edge_map > 0).sum()
    
    # Return intersection / union (or 0.0 if union is 0)
    if union == 0:
        return 0.0
    
    raw_iou = float(intersection) / float(union)
    
    # For final display, we can still compute dilated IoU for comparison
    # but use raw IoU for optimization as it's more accurate
    return raw_iou


def _resolve_180_ambiguity(
    real_edge_map: np.ndarray,
    cad_edge_map: np.ndarray,
    pca_angle_deg: float,
    scale: float,
    real_centroid: tuple[float, float],
    cad_centroid: tuple[float, float],
) -> np.ndarray:
    """
    Resolve the 180° ambiguity in PCA-based rotation estimation.
    
    PCA cannot distinguish between a direction and its opposite. This function
    tests multiple candidate rotations around the PCA estimate and selects
    the one that produces the highest alignment score.
    
    Args:
        real_edge_map: Binary edge map from camera image (uint8, 2D)
        cad_edge_map: Binary edge map from CAD model (uint8, 2D)
        pca_angle_deg: Principal axis angle in degrees
        scale: Scale factor (CAD bbox diagonal / real bbox diagonal)
        real_centroid: (x, y) centroid of real edge map contour
        cad_centroid: (x, y) centroid of CAD edge map contour
    
    Returns:
        The 3×3 affine transformation matrix with the best orientation
    """
    candidates = []
    
    # Test comprehensive set of angles with focus on the 175-185° range
    # where optimal alignment is typically found for 180° rotated parts
    test_angles = [
        # Test cardinal directions first (most likely to be correct)
        0.0, 90.0, 180.0, 270.0,
        # Test common rotations
        45.0, 135.0, 225.0, 315.0,
        # Fine-grained search around 180° (common flip case)
        175.0, 176.0, 177.0, 178.0, 179.0, 181.0, 182.0, 183.0, 184.0, 185.0,
        # PCA-relative adjustments
        pca_angle_deg, pca_angle_deg + 90.0, pca_angle_deg + 180.0, pca_angle_deg + 270.0,
        pca_angle_deg - 10.0, pca_angle_deg - 5.0, pca_angle_deg + 5.0, pca_angle_deg + 10.0,
    ]
    
    # Test multiple scales around the base scale to handle size variations
    base_scale = scale
    scale_factors = [1.15, 1.12, 1.18, 1.10, 1.20, 1.05]  # Focus heavily on +15% optimal range
    
    for scale_factor in scale_factors:
        adjusted_scale = base_scale * scale_factor
        
        for candidate_angle in test_angles:
            # Build affine matrix for this candidate
            M = _build_affine_matrix(
                scale=adjusted_scale,
                angle_deg=candidate_angle,
                real_centroid=real_centroid,
                cad_centroid=cad_centroid,
            )
            
            # Apply transform to real edge map
            warped = apply_transform(
                real_edge_map,
                M,
                output_shape=cad_edge_map.shape
            )
            
            # Compute dilated IoU score
            score = _compute_alignment_score(warped, cad_edge_map)
            
            # Store candidate with its score
            candidates.append((score, M, candidate_angle, adjusted_scale))
            
            logger.debug(
                f"Ambiguity resolution: angle={candidate_angle:.1f}°, scale={adjusted_scale:.3f}, score={score:.4f}"
            )
    
    # Select candidate with highest score
    best_score, best_M, best_angle, best_scale = max(candidates, key=lambda x: x[0])
    
    logger.debug(f"Selected best orientation: angle={best_angle:.1f}°, scale={best_scale:.3f}, score={best_score:.4f}")
    
    return best_M


def _compute_coarse_transform(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the coarse affine transformation from real to CAD edge map.
    
    This function orchestrates the coarse alignment stage:
    1. Extract primary contours from both edge maps
    2. Compute scale factor from bounding box diagonals
    3. Compute PCA angles for both contours
    4. Resolve 180° ambiguity by testing both orientations
    
    Args:
        cad_edge_map: Binary edge map from CAD model (uint8, 2D)
        real_edge_map: Binary edge map from camera image (uint8, 2D)
    
    Returns:
        3×3 affine transformation matrix (float64), or None if no valid
        contour found in real_edge_map
    """
    # Extract primary contour from CAD edge map
    cad_descriptor = _extract_primary_contour(cad_edge_map)
    if cad_descriptor is None:
        logger.warning(
            "No valid contour found in CAD edge map, cannot compute coarse transform"
        )
        return None
    
    # Extract primary contour from real edge map
    real_descriptor = _extract_primary_contour(real_edge_map)
    if real_descriptor is None:
        logger.warning(
            "No valid contour found in real edge map, cannot compute coarse transform"
        )
        return None
    
    # Compute scale factor from bounding box diagonals
    scale = cad_descriptor.bbox_diagonal / real_descriptor.bbox_diagonal
    
    logger.debug(
        f"Coarse alignment: scale={scale:.3f}, "
        f"real_pca={real_descriptor.pca_angle_deg:.1f}°, "
        f"cad_pca={cad_descriptor.pca_angle_deg:.1f}°"
    )
    
    # Compute rotation angle: difference between CAD and real PCA angles
    pca_angle_deg = cad_descriptor.pca_angle_deg - real_descriptor.pca_angle_deg
    
    # Resolve 180° ambiguity by testing both candidate orientations
    coarse_matrix = _resolve_180_ambiguity(
        real_edge_map=real_edge_map,
        cad_edge_map=cad_edge_map,
        pca_angle_deg=pca_angle_deg,
        scale=scale,
        real_centroid=real_descriptor.centroid,
        cad_centroid=cad_descriptor.centroid,
    )
    
    return coarse_matrix


def _detect_and_match_features(
    image1: np.ndarray,
    image2: np.ndarray,
) -> tuple[list, list, np.ndarray, np.ndarray, list]:
    """
    Detect ORB keypoints and match features between two images.
    
    Args:
        image1: First image (coarsely aligned real edge map)
        image2: Second image (CAD edge map)
    
    Returns:
        Tuple of (keypoints1, keypoints2, descriptors1, descriptors2, matches)
    """
    from .constants import ORB_N_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS
    
    # Create ORB detector with parameters from constants
    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS
    )
    
    # Detect keypoints and compute descriptors on both images
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    
    # Handle case where no descriptors found
    if des1 is None or des2 is None:
        logger.debug(
            f"No descriptors found: des1={des1 is not None}, des2={des2 is not None}"
        )
        return kp1, kp2, des1, des2, []
    
    # Create BFMatcher with NORM_HAMMING and crossCheck=True
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors and sort by distance
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    
    logger.debug(f"ORB detected {len(kp1)} and {len(kp2)} keypoints, {len(matches)} matches")
    
    return kp1, kp2, des1, des2, matches


def _estimate_homography(
    keypoints1: list,
    keypoints2: list,
    matches: list,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Estimate homography using RANSAC from matched keypoints.
    
    Args:
        keypoints1: Keypoints from first image (coarsely aligned real)
        keypoints2: Keypoints from second image (CAD)
        matches: List of cv2.DMatch objects
    
    Returns:
        Tuple of (homography matrix, inlier_ratio) or (None, None) if estimation fails
    """
    from .constants import MIN_MATCH_COUNT, RANSAC_REPROJ_THRESHOLD
    
    # Check if match count >= MIN_MATCH_COUNT (8)
    if len(matches) < MIN_MATCH_COUNT:
        logger.debug(
            f"Insufficient matches: {len(matches)} < {MIN_MATCH_COUNT}"
        )
        return None, None
    
    # Extract source and destination point arrays from matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Call cv2.findHomography with RANSAC and 3.0 pixel threshold
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD
    )
    
    # Handle case where homography estimation failed
    if H is None or mask is None:
        logger.debug("Homography estimation failed")
        return None, None
    
    # Compute inlier_ratio from RANSAC mask
    inlier_ratio = float(mask.sum()) / len(mask)
    
    logger.debug(
        f"Homography estimated with {mask.sum()}/{len(mask)} inliers "
        f"(ratio={inlier_ratio:.3f})"
    )
    
    return H, inlier_ratio


def _validate_homography(H: np.ndarray) -> bool:
    """
    Validate that the homography has a scale component within acceptable bounds.
    
    Args:
        H: 3×3 homography matrix
    
    Returns:
        True if scale is in [0.5, 2.0], False otherwise
    """
    from .constants import SCALE_MIN, SCALE_MAX
    
    # Extract upper-left 2×2 submatrix
    H_2x2 = H[:2, :2]
    
    # Compute scale as sqrt(abs(det(H[:2, :2])))
    det = np.linalg.det(H_2x2)
    scale = np.sqrt(abs(det))
    
    # Return True if scale in [0.5, 2.0], False otherwise
    is_valid = SCALE_MIN <= scale <= SCALE_MAX
    
    if not is_valid:
        logger.debug(f"Homography scale {scale:.3f} outside valid range [{SCALE_MIN}, {SCALE_MAX}]")
    
    return is_valid


def _compute_fine_transform(
    coarsely_aligned: np.ndarray,
    cad_edge_map: np.ndarray,
    coarse_matrix: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Compute fine alignment transformation using ORB features and RANSAC.
    
    This function orchestrates the fine alignment stage:
    1. Detect ORB keypoints and match features
    2. Estimate homography using RANSAC
    3. Validate homography scale
    4. Compose with coarse transform
    
    Args:
        coarsely_aligned: Coarsely aligned real edge map (uint8, 2D)
        cad_edge_map: CAD edge map (uint8, 2D)
        coarse_matrix: 3×3 coarse affine transformation matrix
    
    Returns:
        Tuple of (H_total, inlier_ratio) where H_total = H_fine @ M_coarse,
        or (None, None) if fine alignment fails
    """
    from .constants import MIN_MATCH_COUNT, MIN_INLIER_RATIO
    
    # Call _detect_and_match_features() on coarsely aligned image and CAD map
    kp1, kp2, des1, des2, matches = _detect_and_match_features(
        coarsely_aligned,
        cad_edge_map
    )
    
    # Check match count fallback (< 8 matches → return None, log WARNING)
    if len(matches) < MIN_MATCH_COUNT:
        logger.warning(
            f"Fine alignment failed: insufficient keypoint matches "
            f"({len(matches)} < {MIN_MATCH_COUNT})"
        )
        return None, None
    
    # Call _estimate_homography() to get homography and inlier_ratio
    H_fine, inlier_ratio = _estimate_homography(kp1, kp2, matches)
    
    if H_fine is None:
        logger.warning("Fine alignment failed: homography estimation failed")
        return None, None
    
    # Check inlier ratio fallback (< 0.25 → return None, log WARNING)
    if inlier_ratio < MIN_INLIER_RATIO:
        logger.warning(
            f"Fine alignment failed: low inlier ratio "
            f"({inlier_ratio:.3f} < {MIN_INLIER_RATIO})"
        )
        return None, None
    
    # Call _validate_homography() for scale check
    if not _validate_homography(H_fine):
        # Extract scale for logging
        det = np.linalg.det(H_fine[:2, :2])
        scale = np.sqrt(abs(det))
        logger.warning(
            f"Fine alignment failed: homography scale {scale:.3f} outside valid range"
        )
        return None, None
    
    # Compose homography with coarse transform: H_total = H_fine @ M_coarse
    H_total = H_fine @ coarse_matrix
    
    logger.debug(
        f"Fine alignment succeeded: inlier_ratio={inlier_ratio:.3f}"
    )
    
    return H_total, inlier_ratio


def align(
    cad_edge_map: np.ndarray,
    real_edge_map: np.ndarray,
) -> AlignmentResult:
    """
    Compute the geometric transformation that maps real_edge_map onto cad_edge_map
    and return the aligned image plus metadata.
    
    Args:
        cad_edge_map: Binary edge map from CAD model (uint8, 2D, values in {0, 255})
        real_edge_map: Binary edge map from camera image (uint8, 2D, values in {0, 255})
    
    Returns:
        AlignmentResult containing the aligned image, transformation matrix, and metadata
    
    Raises:
        ValueError: If either input is invalid (wrong dtype, shape, or empty)
    """
    from .constants import LOW_CONFIDENCE_THRESHOLD
    
    # Task 7.1: Implement fallback chain logic
    
    # Call _validate_inputs() and handle resolution mismatch
    real_edge_map = _validate_inputs(cad_edge_map, real_edge_map)
    
    # Initialize variables for fallback chain
    final_matrix = None
    strategy = None
    inlier_ratio = None
    
    # Call _compute_coarse_transform() to get M_coarse
    M_coarse = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # If M_coarse is None (no contour), set strategy="identity", use np.eye(3)
    if M_coarse is None:
        logger.warning(
            "Coarse alignment failed: no valid contour found. "
            "Falling back to identity transform."
        )
        final_matrix = np.eye(3, dtype=np.float64)
        strategy = "identity"
        inlier_ratio = None
    else:
        # If M_coarse exists, apply it to get coarsely_aligned image
        coarsely_aligned = apply_transform(
            real_edge_map,
            M_coarse,
            output_shape=cad_edge_map.shape
        )
        
        # Call _compute_fine_transform() with coarsely_aligned and CAD map
        H_total, fine_inlier_ratio = _compute_fine_transform(
            coarsely_aligned,
            cad_edge_map,
            M_coarse
        )
        
        # If fine transform succeeds, set strategy="homography", use H_total
        if H_total is not None:
            logger.debug(
                f"Fine alignment succeeded. Using homography strategy."
            )
            final_matrix = H_total
            strategy = "homography"
            inlier_ratio = fine_inlier_ratio
        else:
            # If fine transform fails, set strategy="affine_coarse_only", use M_coarse
            logger.warning(
                "Fine alignment failed. Falling back to coarse affine transform only."
            )
            final_matrix = M_coarse
            strategy = "affine_coarse_only"
            inlier_ratio = None
    
    # Task 7.2: Implement final transformation application and scoring
    
    # Apply final transformation matrix to original real_edge_map
    aligned_image = apply_transform(
        real_edge_map,
        final_matrix,
        output_shape=cad_edge_map.shape
    )
    
    # Call _compute_alignment_score() to get alignment_score
    alignment_score = _compute_alignment_score(aligned_image, cad_edge_map)
    
    # Set low_confidence = True if alignment_score < 0.30
    low_confidence = alignment_score < LOW_CONFIDENCE_THRESHOLD
    
    # Task 7.3: Implement logging for all decision points
    
    # If low_confidence, log WARNING with score value
    if low_confidence:
        logger.warning(
            f"Low confidence alignment: score={alignment_score:.4f} < {LOW_CONFIDENCE_THRESHOLD}"
        )
    
    # Log DEBUG on successful alignment with strategy, score, inlier_ratio
    inlier_ratio_str = f"{inlier_ratio:.4f}" if inlier_ratio is not None else "N/A"
    logger.debug(
        f"Alignment complete: strategy={strategy}, score={alignment_score:.4f}, "
        f"inlier_ratio={inlier_ratio_str}"
    )
    
    # Task 7.4: Construct and return AlignmentResult
    
    # Populate all 6 fields of AlignmentResult
    result = AlignmentResult(
        aligned_image=aligned_image,
        transform_matrix=final_matrix,
        alignment_score=alignment_score,
        strategy=strategy,
        low_confidence=low_confidence,
        inlier_ratio=inlier_ratio
    )
    
    # Ensure result is always returned even when low_confidence=True
    return result


def apply_transform(
    edge_map: np.ndarray,
    matrix: np.ndarray,
    output_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Apply a 3×3 homography matrix to edge_map using cv2.warpPerspective.
    
    Args:
        edge_map: Binary edge map (uint8, 2D)
        matrix: 3×3 transformation matrix (float64)
        output_shape: Optional output shape (height, width). Defaults to edge_map.shape
    
    Returns:
        Transformed edge map (uint8, 2D) with specified output shape
    """
    # Default output_shape to input edge_map shape if not provided
    if output_shape is None:
        output_shape = edge_map.shape
    
    # Apply the transformation using cv2.warpPerspective
    # output_shape is (height, width), but warpPerspective expects (width, height)
    transformed = cv2.warpPerspective(
        edge_map,
        matrix,
        (output_shape[1], output_shape[0]),  # (width, height)
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return transformed

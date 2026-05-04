"""
Synthetic test fixtures for CAD-Image Alignment Module.

This module provides generators for creating synthetic edge maps used in testing:
- Bracket-like shapes with known geometry
- Rotated variants with ground-truth transformation matrices
- Noisy variants simulating metallic reflections
"""

import numpy as np
import cv2
from typing import Tuple


def generate_bracket_shape(
    image_size: Tuple[int, int] = (512, 512),
    outer_rect_size: Tuple[int, int] = (300, 200),
    cutout_size: int = 60,
    circle_radius: int = 15,
) -> np.ndarray:
    """
    Generate a bracket-like shape with rounded rectangle, square cutout, and three circles.
    
    This creates a synthetic edge map resembling a mechanical bracket:
    - Rounded rectangle as the outer boundary
    - Square cutout on one side
    - Three circles arranged in a pattern
    
    The shape is drawn on a blank canvas and Canny edge detection is applied
    to produce a binary edge map suitable for alignment testing.
    
    Args:
        image_size: Output image dimensions (height, width)
        outer_rect_size: Size of the outer rounded rectangle (width, height)
        cutout_size: Size of the square cutout
        circle_radius: Radius of the three circles
    
    Returns:
        Binary edge map (uint8, 2D, values in {0, 255})
    
    **Validates: Requirements Testing Strategy, 6.1**
    """
    height, width = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate center position for the shape
    center_x = width // 2
    center_y = height // 2
    
    # Draw outer rounded rectangle
    rect_width, rect_height = outer_rect_size
    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
    corner_radius = 20
    
    # Draw rounded rectangle using cv2.rectangle with thickness=-1 (filled)
    # We'll draw it filled first, then extract edges
    temp_canvas = canvas.copy()
    cv2.rectangle(
        temp_canvas,
        top_left,
        bottom_right,
        255,
        thickness=-1
    )
    
    # Draw rounded corners by adding circles at corners
    # Top-left corner
    cv2.circle(
        temp_canvas,
        (top_left[0] + corner_radius, top_left[1] + corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    # Top-right corner
    cv2.circle(
        temp_canvas,
        (bottom_right[0] - corner_radius, top_left[1] + corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    # Bottom-left corner
    cv2.circle(
        temp_canvas,
        (top_left[0] + corner_radius, bottom_right[1] - corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    # Bottom-right corner
    cv2.circle(
        temp_canvas,
        (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    
    # Draw square cutout on the right side (subtract from the shape)
    cutout_x = center_x + rect_width // 2 - cutout_size // 2
    cutout_y = center_y - cutout_size // 2
    cv2.rectangle(
        temp_canvas,
        (cutout_x, cutout_y),
        (cutout_x + cutout_size, cutout_y + cutout_size),
        0,  # Black to create cutout
        thickness=-1
    )
    
    # Draw three circles in a triangular pattern
    # Circle 1: top-left
    circle1_pos = (center_x - rect_width // 4, center_y - rect_height // 4)
    cv2.circle(temp_canvas, circle1_pos, circle_radius, 0, thickness=-1)
    
    # Circle 2: bottom-left
    circle2_pos = (center_x - rect_width // 4, center_y + rect_height // 4)
    cv2.circle(temp_canvas, circle2_pos, circle_radius, 0, thickness=-1)
    
    # Circle 3: center
    circle3_pos = (center_x, center_y)
    cv2.circle(temp_canvas, circle3_pos, circle_radius, 0, thickness=-1)
    
    # Apply Canny edge detection to produce binary edge map
    # First, blur slightly to reduce noise in edge detection
    blurred = cv2.GaussianBlur(temp_canvas, (5, 5), 0)
    
    # Apply Canny edge detection
    edge_map = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    return edge_map


def generate_rotated_variant(
    edge_map: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a rotated variant of an edge map with known ground-truth transformation.
    
    This function applies a rotation transformation to an edge map and returns both
    the rotated image and the ground-truth transformation matrix. The rotation is
    performed about the image center.
    
    Args:
        edge_map: Input binary edge map (uint8, 2D)
        angle_deg: Rotation angle in degrees (counterclockwise)
    
    Returns:
        Tuple of (rotated_edge_map, transform_matrix) where:
        - rotated_edge_map: Rotated binary edge map (uint8, 2D)
        - transform_matrix: 3×3 ground-truth transformation matrix (float64)
    
    **Validates: Requirements Testing Strategy**
    """
    height, width = edge_map.shape
    center = (width / 2.0, height / 2.0)
    
    # Get rotation matrix using cv2.getRotationMatrix2D
    # This returns a 2×3 affine matrix
    rotation_matrix_2x3 = cv2.getRotationMatrix2D(
        center=center,
        angle=angle_deg,
        scale=1.0
    )
    
    # Convert 2×3 matrix to 3×3 homogeneous matrix
    transform_matrix = np.eye(3, dtype=np.float64)
    transform_matrix[:2, :] = rotation_matrix_2x3
    
    # Apply rotation using cv2.warpAffine
    rotated_edge_map = cv2.warpAffine(
        edge_map,
        rotation_matrix_2x3,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return rotated_edge_map, transform_matrix


def generate_noisy_variant(
    edge_map: np.ndarray,
    noise_level: float = 0.02,
) -> np.ndarray:
    """
    Generate a noisy variant of an edge map simulating metallic reflections.
    
    This function adds salt-and-pepper noise to an edge map to simulate the effect
    of metallic surface reflections and other imaging artifacts. The noise consists
    of randomly placed white pixels (salt) and black pixels (pepper).
    
    Args:
        edge_map: Input binary edge map (uint8, 2D)
        noise_level: Fraction of pixels to corrupt with noise (default: 0.02 = 2%)
    
    Returns:
        Noisy binary edge map (uint8, 2D)
    
    **Validates: Requirements 6.1, Testing Strategy**
    """
    # Create a copy to avoid modifying the original
    noisy_edge_map = edge_map.copy()
    
    # Calculate total number of pixels
    total_pixels = edge_map.shape[0] * edge_map.shape[1]
    
    # Calculate number of pixels to corrupt
    num_noise_pixels = int(total_pixels * noise_level)
    
    # Add salt noise (white pixels)
    # Randomly select pixel coordinates
    salt_coords_y = np.random.randint(0, edge_map.shape[0], num_noise_pixels // 2)
    salt_coords_x = np.random.randint(0, edge_map.shape[1], num_noise_pixels // 2)
    noisy_edge_map[salt_coords_y, salt_coords_x] = 255
    
    # Add pepper noise (black pixels)
    pepper_coords_y = np.random.randint(0, edge_map.shape[0], num_noise_pixels // 2)
    pepper_coords_x = np.random.randint(0, edge_map.shape[1], num_noise_pixels // 2)
    noisy_edge_map[pepper_coords_y, pepper_coords_x] = 0
    
    return noisy_edge_map

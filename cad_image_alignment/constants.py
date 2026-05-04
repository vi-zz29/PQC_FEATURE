"""
Algorithm parameters and constants for the alignment module.
"""

# Coarse alignment parameters
MIN_CONTOUR_AREA_FRACTION = 0.01  # Ignore contours < 1% of image area

# Fine alignment - ORB feature detection parameters
ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Fine alignment - matching and RANSAC parameters
MIN_MATCH_COUNT = 8
RANSAC_REPROJ_THRESHOLD = 3.0  # pixels
MIN_INLIER_RATIO = 0.25

# Homography validation parameters
SCALE_MIN = 0.5
SCALE_MAX = 2.0

# Quality assessment parameters
DILATION_KERNEL_SIZE = 3  # 3×3 structuring element for dilated IoU
LOW_CONFIDENCE_THRESHOLD = 0.05  # 5% - realistic threshold for raw IoU (more accurate than dilated)

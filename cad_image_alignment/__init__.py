"""
CAD-Image Alignment Module

This module computes geometric transformations that map real camera-derived edge maps
onto CAD reference edge maps, enabling downstream mismatch detection between manufactured
parts and their design specifications.
"""

import logging

# Configure module-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Public API
from cad_image_alignment.alignment import align, apply_transform, AlignmentResult

__all__ = ["align", "apply_transform", "AlignmentResult"]
__version__ = "0.1.0"

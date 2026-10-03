######################################################################
# SYNTHETIC SCRIBBLE GENERATION - Shape Utilities
######################################################################
"""
This module provides utilities for generating synthetic shapes (circles,
rectangles, triangles) as binary masks for training data augmentation.
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import random

# Pip/Third-party installs
import cv2
import numpy as np

# Application/Local module imports
import src.configConstants as configConstants

######################################################################
# SHAPE GENERATION FUNCTIONS
######################################################################

def generateShapes(imageSize: tuple[int, ...], nSamples: int, shape: str) -> list[np.ndarray]:
    """
    Randomly generate a list of shapes (circles, rectangles, triangles) as binary masks.
    
    Parameters
    ----------
    imageSize : tuple[int, ...]
        Shape of the output mask (height, width)
    nSamples : int
        Number of shapes to generate
    shape : str
        Type of shape to generate: 'circle', 'rectangle', 'triangle', or 'any'
    
    Returns
    -------
    list[np.ndarray]
        List of binary masks containing the generated shapes
    """
    # Step 1 - Handle multiple samples batch generation
    if nSamples > 1:
        # Step 1.1 - Determine shape distribution
        if shape == configConstants.SHAPE_VALUE_ANY:
            # Distribute equally: 1/3 circles, 1/3 triangles, remainder rectangles
            nCircles = nTriangles = nSamples // 3
            nRectangles = nSamples - nCircles * 2
        else:
            # Generate only specified shape type
            nCircles = nTriangles = nRectangles = 0
            if shape == "circle":
                nCircles = nSamples
            elif shape == "rectangle":
                nRectangles = nSamples
            elif shape == "triangle":
                nTriangles = nSamples
        
        # Step 1.2 - Generate all shapes
        circles = [generateCircle(imageSize) for _ in range(nCircles)]
        rectangles = [generateRectangle(imageSize) for _ in range(nRectangles)]
        triangles = [generateTriangle(imageSize) for _ in range(nTriangles)]

        return circles + rectangles + triangles
    
    # Step 2 - Handle single sample generation
    else:
        if shape == configConstants.SHAPE_VALUE_ANY:
            shape = random.choice(["circle", "rectangle", "triangle"])
        if shape == "circle":
            return [generateCircle(imageSize)]
        elif shape == "rectangle":
            return [generateRectangle(imageSize)]
        elif shape == "triangle":
            return [generateTriangle(imageSize)]
        else:
            raise ValueError(f"Invalid shape {shape}")


def generateCircle(imageSize: tuple[int, ...]) -> np.ndarray:
    """
    Generate a random circle as a binary mask.
    
    Parameters
    ----------
    imageSize : tuple[int, ...]
        Shape of the output mask (height, width)
    
    Returns
    -------
    np.ndarray, shape=(height, width), dtype=uint8
        Binary mask with a filled circle (255 inside, 0 outside)
        
    Notes
    -----
    Circle radius ranges from 1/4 to 1/2 of the minimum image dimension.
    """
    # Step 0 - Initialize empty mask
    mask = np.zeros(imageSize, dtype=np.uint8)

    # Step 1 - Generate random circle parameters
    radius = random.randint(min(imageSize) // 4, min(imageSize) // 2)
    center = (
        random.randint(radius, imageSize[0] - radius),
        random.randint(radius, imageSize[1] - radius),
    )
    
    # Step 2 - Draw filled circle
    cv2.circle(mask, center=center, radius=radius, color=(255,), thickness=-1)
    return mask


def generateRectangle(imageSize: tuple[int, ...]) -> np.ndarray:
    """
    Generate a random rectangle as a binary mask.
    
    Parameters
    ----------
    imageSize : tuple[int, ...]
        Shape of the output mask (height, width)
    
    Returns
    -------
    np.ndarray
        Binary mask with a filled rectangle
    """
    mask = np.zeros(imageSize, dtype=np.uint8)

    width = random.randint(imageSize[0] // 3, imageSize[0] // 2)
    height = random.randint(imageSize[1] // 3, imageSize[1] // 2)

    x1 = random.randint(0, imageSize[0] - width)
    y1 = random.randint(0, imageSize[1] - height)

    x2 = x1 + width
    y2 = y1 + height

    cv2.rectangle(mask, pt1=(y1, x1), pt2=(y2, x2), color=(255,), thickness=-1)
    return mask


def generateTriangle(imageSize: tuple[int, ...]) -> np.ndarray:
    """
    Generate a random triangle as a binary mask.
    
    Parameters
    ----------
    imageSize : tuple[int, ...]
        Shape of the output mask (height, width)
    
    Returns
    -------
    np.ndarray
        Binary mask with a filled triangle
    """
    mask = np.zeros(imageSize, dtype=np.uint8)

    # Step 1 - Calculate bounds for triangle size
    minSize = min(imageSize[0], imageSize[1]) // 3
    maxSize = min(imageSize[0], imageSize[1])

    # Step 2 - Generate three random points for the triangle
    # Note: the base is always axis-aligned here; orientation diversity comes from the
    # rotation augmentation applied by ScribbleDataset._applyTransforms().
    # First point - base left
    x1 = random.randint(0, imageSize[0] - minSize)
    y1 = random.randint(0, imageSize[1] - minSize)

    # Second point - base right (ensuring minimum width)
    x2 = x1 + random.randint(minSize, maxSize)
    x2 = min(x2, imageSize[0] - 1)
    y2 = y1

    # Third point - apex (ensuring minimum height)
    x3 = (x1 + x2) // 2
    height = random.randint(minSize, maxSize)
    y3 = (
        max(y1 - height, 0)
        if y1 > imageSize[1] // 2
        else min(y1 + height, imageSize[1] - 1)
    )
    y3 = max(y3, minSize)

    # Step 3 - Create triangle points array
    pts = np.array([[y1, x1], [y2, x2], [y3, x3]], np.int32)

    # Step 4 - Draw filled triangle
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

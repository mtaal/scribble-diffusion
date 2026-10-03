######################################################################
# SCRIBBLE GENERATION - Morphological and Contour-Based Methods
######################################################################
"""
This module provides utilities for generating synthetic scribbles on binary masks.

Functions:
    getErodedMasks(): Creates eroded versions of masks for scribble generation
    getMaskForLabel(): Extracts mask for a specific label
    getContourScribbleForBinarySlice(): Generates contour-based scribbles
    getMorphologyScribbleForBinarySlice(): Generates medial/skeleton-based scribbles
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import random
import traceback

# Pip/Third-party installs
import torch
import numpy as np
import skimage
import skimage.morphology
import skimage.measure
import matplotlib.pyplot as plt
import voxynth

# Application/Local module imports
import src.configConstants as configConstants

######################################################################
# HELPER FUNCTIONS - MASK PROCESSING
######################################################################

def getErodedMasks(mask: np.ndarray, getContour: bool = False):
    """
    Create eroded versions of a mask for synthetic scribble generation.
    
    Parameters
    ----------
    mask : np.ndarray, shape=(H, W)
        Binary mask (error region from GT and prediction)
    getContour : bool, default=False
        If True, return contour between erosion levels
        If False, return randomly selected eroded mask
    
    Returns
    -------
    np.ndarray or None
        Eroded mask or contour mask for scribble generation
        
    Notes
    -----
    Uses 4 erosion steps to avoid scribbles being too deep inside the mask.
    Used by getContourScribbleForBinarySlice() and getMorphologyScribbleForBinarySlice().
    """
    # Step 0 - Initialize return variables
    maskErodeContour, maskErodeN = None, None
    try:
        # Step 1 - Apply multiple erosion operations
        maskErode1, maskErode2, maskErode3, maskErode4 = None, None, None, None
        try:
            maskErode1 = skimage.morphology.erosion(mask.astype(np.uint8))
            maskErode2 = skimage.morphology.erosion(maskErode1)
            maskErode3 = skimage.morphology.erosion(maskErode2)
            maskErode4 = skimage.morphology.erosion(maskErode3)
        except:
            pass

        # Step 2 - Select random eroded mask with non-zero pixels
        if not getContour:
            maskChoices = [mask]
            if maskErode1 is not None: 
                if np.sum(maskErode1) > 0: maskChoices.append(maskErode1)
            if maskErode2 is not None: 
                if np.sum(maskErode2) > 0: maskChoices.append(maskErode2)
            if maskErode3 is not None:
                if np.sum(maskErode3) > 0: maskChoices.append(maskErode3)
            if maskErode4 is not None:
                if np.sum(maskErode4) > 0: maskChoices.append(maskErode4)
            if len(maskChoices) > 0:
                maskErodeN = random.choice(maskChoices).astype(np.uint8)
                if np.sum(maskErodeN) == 0: 
                    maskErodeN = mask

        # Step 3 - Compute contour between erosion levels
        else:
            if maskErode1 is not None and maskErode2 is not None and maskErode3 is not None and maskErode4 is not None:
                if np.sum(maskErode1) > 0 and np.sum(maskErode2) > 0 and np.sum(maskErode3) > 0 and np.sum(maskErode4) > 0:
                    maskErodeContour = [maskErode1 - maskErode2, maskErode2 - maskErode3, maskErode3 - maskErode4][np.random.choice([0,1,2])]
            elif maskErode1 is not None and maskErode2 is not None and maskErode3 is not None:
                if np.sum(maskErode1) > 0 and np.sum(maskErode2) > 0 and np.sum(maskErode3) > 0:
                    maskErodeContour = [maskErode1 - maskErode2, maskErode2 - maskErode3][np.random.choice([0,1])]
            elif maskErode1 is not None and maskErode2 is not None:
                if np.sum(maskErode1) > 0 and np.sum(maskErode2) > 0:
                    maskErodeContour = maskErode1 - maskErode2
            elif maskErode1 is not None:
                if np.sum(maskErode1) > 0:
                    maskErodeContour = mask - maskErode1
            
            if maskErodeContour is not None:
                if np.sum(maskErodeContour) > 0:
                    maskErodeContour = maskErodeContour.astype(np.uint8)
                else:
                    maskErodeContour = None

    except:
        traceback.print_exc()

    # Step 4 - Return appropriate result
    if not getContour:
        return maskErodeN
    else:
        return maskErodeContour

def getMaskForLabel(arrayMask: np.ndarray, label: int):
    """
    Extract binary mask for a specific label from a labeled array.
    
    Parameters
    ----------
    arrayMask : np.ndarray, shape=(H, W) or (H, W, D)
        Labeled array containing integer labels
    label : int
        Label value to extract
    
    Returns
    -------
    np.ndarray
        Binary mask where pixels matching label are 1, others are 0
    """
    arrayMaskOfLabel = None
    try:
        # Step 1 - Create binary mask for specified label
        arrayMaskOfLabel = np.zeros(arrayMask.shape)
        arrayMaskOfLabel[arrayMask == label] = 1
    except:
        traceback.print_exc()

    return arrayMaskOfLabel

######################################################################
# SCRIBBLE GENERATION FUNCTIONS
######################################################################

def getContourScribbleForBinarySlice(mask: np.ndarray, show: bool = False):
    """
    Generate a contour-based scribble along the boundary of a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray, shape=(H, W)
        Binary mask containing 1s and 0s
    show : bool, default=False
        If True, display visualization of scribble generation steps
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (contourScribbleMask, contourScribbleMaskFinal)
        - contourScribbleMask: Unwarped contour scribble
        - contourScribbleMaskFinal: Warped and skeletonized contour scribble
        
    Notes
    -----
    Applies Perlin noise masking and optional spatial warping to create
    realistic interactive scribbles.
    """
    # Step 0 - Initialize return variables
    contourScribbleMask = None
    contourScribbleMaskFinal = None

    try:
        # Step 1 - Get eroded contour edge
        maskErodeEdge = getErodedMasks(mask, getContour=True)
        if maskErodeEdge is None:
            return None, None

        # Step 2 - Generate Perlin noise mask for variation
        noiseMask = (
            voxynth.noise.perlin(
                shape=[maskErodeEdge.shape[0], maskErodeEdge.shape[1]],
                smoothing=np.random.uniform(2, 14),
                magnitude=1,
            )
            .cpu()
            .numpy()
            > 0
        )

        # Step 3 - Apply noise mask to edge to create interaction
        contourScribbleMask = noiseMask * maskErodeEdge

        # Step 4 - Retry with new noise if empty
        if np.sum(contourScribbleMask) == 0:
            noiseMask2 = (
                voxynth.noise.perlin(
                    shape=[maskErodeEdge.shape[0], maskErodeEdge.shape[1]],
                    smoothing=np.random.uniform(2, 14),
                    magnitude=1,
                )
                .cpu()
                .numpy()
                > 0
            )
            contourScribbleMask = noiseMask2 * maskErodeEdge
            if np.sum(contourScribbleMask) == 0:
                contourScribbleMask = maskErodeEdge

        contourScribbleMask = np.round(contourScribbleMask).astype(np.uint8)

        # Step 5 - Apply spatial warping for realism
        if random.random() < 0.2:
            contourScribbleMaskFinal = contourScribbleMask
        else:
            deformationField = voxynth.transform.random_transform(
                shape=mask.shape,
                affine_probability=0,
                warp_probability=1,
                warp_integrations=0,
                warp_smoothing_range=[10, 16],
                warp_magnitude_range=[1, 6],
                isdisp=False,
            )
            contourScribbleMaskWarped = (
                voxynth.transform.spatial_transform(
                    torch.from_numpy(contourScribbleMask).unsqueeze(0),
                    trf=deformationField,
                    isdisp=False,
                )
                .cpu()
                .numpy()[0]
            )

            # Step 5.1 - Skeletonize the warped scribble
            contourScribbleMaskWarpedSkel = skimage.morphology.skeletonize(
                contourScribbleMaskWarped > 0
            )
            contourScribbleMaskFinal = np.round(
                contourScribbleMaskWarpedSkel
            ).astype(np.uint8)

        # Step 6 - Visualization for debugging
        if show:
            f, axarr = plt.subplots(1, 5, figsize=(15, 3))
            axarr[0].imshow(mask)
            axarr[0].set_title("Original Mask")
            axarr[1].imshow(mask)
            axarr[1].imshow(maskErodeEdge, alpha=0.5)
            axarr[1].set_title("Eroded Edge")
            axarr[2].imshow(noiseMask)
            axarr[2].imshow(maskErodeEdge, alpha=0.5)
            axarr[2].set_title("Noise Mask")
            axarr[3].imshow(mask)
            axarr[3].imshow(contourScribbleMask, alpha=0.5)
            axarr[3].set_title("Contour Scribble")
            axarr[4].imshow(mask)
            axarr[4].imshow(contourScribbleMaskFinal, alpha=0.5)
            axarr[4].set_title("Final Warped")
            plt.show()

    except:
        traceback.print_exc()

    return contourScribbleMask, contourScribbleMaskFinal

def getMorphologyScribbleForBinarySlice(
    mask: np.ndarray, scribbleType: str, valMode: bool = False, verbose: bool = False
):
    """
    Generate morphology-based scribbles using medial axis or skeletonization.
    
    Parameters
    ----------
    mask : np.ndarray, shape=(H, W)
        Binary mask containing 1s and 0s
    scribbleType : str
        Type of morphological scribble: 'medial' or 'skeleton'
    valMode : bool, default=False
        If True, skip random breakage for consistent validation
    verbose : bool, default=False
        Enable verbose logging
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (binaryInteractionMaskSlice, binaryInteractionMaskSliceBroken)
        - binaryInteractionMaskSlice: Complete medial/skeleton scribble
        - binaryInteractionMaskSliceBroken: Randomly broken/sampled version
        
    Notes
    -----
    Reference: https://scikit-image.org/docs/0.24.x/auto_examples/edges/plot_skeleton.html
    In most cases, medial axis and skeletonize produce similar results.
    """
    binaryInteractionMaskSlice = None
    binaryInteractionMaskSliceBroken = None

    try:
        # Step 0 - Define helper function for random skeleton breaking
        def randomBreakSkeleton(skeleton: np.ndarray, numBreaks: int = 3):
            """
            Randomly remove pixels from skeleton to create breaks.
            
            Parameters
            ----------
            skeleton : np.ndarray
                Binary skeleton mask
            numBreaks : int, default=3
                Number of random breaks to introduce
                
            Returns
            -------
            np.ndarray
                Skeleton with random breaks
            """
            coords = np.column_stack(np.where(skeleton))
            for _ in range(numBreaks):
                y, x = random.choice(coords)
                skeleton[y, x] = 0
            return skeleton

        # Step 1 - Get eroded error mask
        errorMask = getErodedMasks(mask, getContour=False)
        if errorMask is None: 
            return None, None
        if np.sum(errorMask) < 1:
            errorMask = mask

        # Step 2 - Apply morphological operation (medial axis or skeletonize)
        if scribbleType == configConstants.SCRIBBLE_TYPE_MEDIAL:
            binaryInteractionMaskSlice = skimage.morphology.medial_axis(errorMask)
        elif scribbleType == configConstants.SCRIBBLE_TYPE_SKELETON:
            binaryInteractionMaskSlice = skimage.morphology.skeletonize(errorMask)
        else:
            print(f"   --- [WARNING][getMorphologyScribbleForBinarySlice()] Invalid scribbleType: {scribbleType}")
            return None, None

        if np.sum(binaryInteractionMaskSlice) == 0:
            return None, None

        # Step 3 - Apply random breakage/sampling for variation
        PERC_SCRIBBLE_BREAK_INTO_SECTIONS = 0.45  # Break skeleton into sections
        PERC_SCRIBBLE_BREAK_INTO_POINTS = 0.90    # Sample random points
        NUM_BREAKS = 3
        
        try:
            chanceVal = random.random()
            if valMode:
                chanceVal = 1.0  # No breakage for validation

            # Step 3.1 - Break skeleton and select random connected component
            if chanceVal < PERC_SCRIBBLE_BREAK_INTO_SECTIONS:
                binaryInteractionCopy: np.ndarray = binaryInteractionMaskSlice.copy()  # type: ignore
                skeletonizedMaskSliceBroken = randomBreakSkeleton(
                    binaryInteractionCopy, NUM_BREAKS
                )
                labelResult = skimage.measure.label(skeletonizedMaskSliceBroken, return_num=True)
                skeletonizedMaskSliceBrokenComponents: np.ndarray = labelResult[0]  # type: ignore
                skeletonizedMaskSliceBrokenComponentCount: int = labelResult[1]  # type: ignore
                
                if skeletonizedMaskSliceBrokenComponentCount == 0:
                    binaryInteractionMaskSliceBroken = binaryInteractionMaskSlice
                    if verbose:
                        print(f"--- [INFO][getMorphologyScribbleForBinarySlice()] No components found: "
                              f"mask_sum={np.sum(errorMask)}, scribble_sum={np.sum(binaryInteractionMaskSlice)}")
                else:
                    chosenComponent = random.randint(1, skeletonizedMaskSliceBrokenComponentCount)
                    binaryInteractionMaskSliceBroken = getMaskForLabel(
                        skeletonizedMaskSliceBrokenComponents, chosenComponent
                    )

            # Step 3.2 - Sample random percentage of skeleton points
            elif PERC_SCRIBBLE_BREAK_INTO_SECTIONS <= chanceVal < PERC_SCRIBBLE_BREAK_INTO_POINTS:
                binaryInteractionPoints = np.argwhere(binaryInteractionMaskSlice)
                np.random.shuffle(binaryInteractionPoints)
                keepRatio = np.random.uniform(0.3, 1)
                binaryInteractionPoints = binaryInteractionPoints[:int(keepRatio * len(binaryInteractionPoints))]
                binaryInteractionMaskSliceBroken = np.zeros_like(errorMask)
                for point in binaryInteractionPoints:
                    binaryInteractionMaskSliceBroken[point[0], point[1]] = 1

            # Step 3.3 - Keep complete medial axis/skeleton
            else:
                binaryInteractionMaskSliceBroken = binaryInteractionMaskSlice

        except:
            traceback.print_exc()
            print(f"   --- [WARNING][getMorphologyScribbleForBinarySlice(valMode={valMode})] Error breaking skeleton")

    except:
        traceback.print_exc()

    return binaryInteractionMaskSlice, binaryInteractionMaskSliceBroken

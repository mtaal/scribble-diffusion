######################################################################
# SCRIBBLE DATASET - Data Loading and Augmentation
######################################################################
"""
Dataset class for loading and generating scribble segmentation training data.

This module handles:
    - Real scribble data loading from disk
    - Synthetic scribble generation with Perlin noise augmentation
    - Fully synthetic shape-based data generation
    - Data transformations and augmentation
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import os
import random
import traceback
from pathlib import Path
from typing import List, Optional, Union

# Pip/Third-party installs
import cv2
import imageio
import tqdm
import scipy
import scipy.ndimage
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation
from scipy import ndimage as ndi
import voxynth as voxynth  # https://github.com/dalcalab/voxynth

import torch
import torchvision
import torchvision.transforms.functional as TF

# Application/Local module imports
import src.configConstants as configConstants
import src.utils.utils as utils
from src.data.scribbles import (
    getContourScribbleForBinarySlice,
    getMorphologyScribbleForBinarySlice,
)
from src.data.syntheticScribbles import generateShapes

######################################################################
# CONSTANTS
######################################################################

GT_COLOR = np.array([0, 255, 0], dtype=np.uint8)
PREDICTION_COLOR = np.array([255, 0, 0], dtype=np.uint8)
OVERLAP_COLOR = np.array([255, 255, 0], dtype=np.uint8)
SCRIBBLE_COLOR = np.array([0, 0, 255], dtype=np.uint8)

######################################################################
# GLOBALS - PATHS
######################################################################

DIR_SRC = Path(__file__).parent.parent.absolute()
DIR_ROOT = DIR_SRC.parent
DIR_DATA = DIR_ROOT.parent / "data"
DIR_TMP = DIR_ROOT / configConstants.FOLDERNAME_TMP
Path(DIR_TMP).mkdir(parents=True, exist_ok=True)

######################################################################
# HELPER FUNCTIONS - IMAGE I/O
######################################################################

def saveImage(image, path, name):
    """
    Save image to path with name.
    
    Parameters
    ----------
    image : np.ndarray
        Image to save
    path : str
        Directory path
    name : str
        Filename
    """
    if not os.path.exists(path):
        os.makedirs(path)
    imagePath = os.path.join(path, name)
    imageio.imwrite(imagePath, image)


def loadRandomImagesRepeated(repeat=32, width=128, height=128):
    """
    Load and repeat a random image for testing.
    
    Parameters
    ----------
    repeat : int
        Number of times to repeat
    width : int
        Target width
    height : int
        Target height
        
    Returns
    -------
    torch.Tensor
        Repeated image tensor
    """
    imagePath = DIR_SRC / "data" / "random_image.png"
    image = Image.open(imagePath).convert('L')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((width, height))
    ])
    
    tensorImage: torch.Tensor = transform(image)  # type: ignore
    return tensorImage.unsqueeze(0).repeat(repeat, 3, 1, 1)


######################################################################
# HELPER FUNCTIONS - SCRIBBLE PROCESSING
######################################################################

def determineScribbleClass(
    scribble: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray
) -> int:
    """
    Determine scribble class (foreground or background) based on majority overlap.
    
    Binary classification: each scribble is classified as EITHER foreground OR background.
    
    Parameters
    ----------
    scribble : np.ndarray
        Binary scribble mask (single channel, not split)
    gt : np.ndarray
        Ground truth mask
    prediction : np.ndarray
        Prediction mask
        
    Returns
    -------
    int
        0 = background (scribbles primarily on FP regions: Pred=1, GT=0)
        1 = foreground (scribbles primarily on GT=1 regions)
    """
    # Step 1 - Get scribble pixels
    scribblePixels = scribble > 0
    if not scribblePixels.any():
        return 0  # Default to background if no scribbles
    
    # Step 2 - Calculate overlaps
    # Foreground: scribbles on GT regions (including true positives and false negatives)
    scribbleOnGt = scribblePixels & (gt > 0)
    scribbleOnGtSum = np.sum(scribbleOnGt)

    # If there is no prediction (only ground truth), decide by GT majority within scribble pixels.
    # Foreground if most scribble pixels lie on GT, otherwise background.
    if not np.any(prediction > 0):
        scribbleOnNotGtSum = np.sum(scribblePixels & (gt == 0))
        return 1 if scribbleOnGtSum > scribbleOnNotGtSum else 0
    
    # Background: scribbles on false positive regions (Pred=1, GT=0)
    scribbleOnPredictionNotGtSum = scribblePixels & (prediction > 0) & (gt == 0)
    scribbleOnPredictionNotGtSum = np.sum(scribbleOnPredictionNotGtSum)

    # Step 3 - Binary classification based on majority
    # Classify as foreground if more scribbles are on GT than on FP
    if scribbleOnPredictionNotGtSum >= scribbleOnGtSum:
        return 0  # Background scribble
    else:
        return 1  # Foreground scribble


def getScribbleType(mask: np.ndarray) -> bool:
    """
    Determine scribble type based on overlap between foreground and background.
    
    Parameters
    ----------
    mask : np.ndarray, shape=(H, W, 3)
        RGB array with R=Pred, G=GT, B=Scribble
        
    Returns
    -------
    bool
        True if foreground scribble (more overlap with GT)
    """
    try:
        # Step 1 - Extract foreground, background, and scribble masks
        foregroundMask = mask[:, :, 1]
        scribble = mask[:, :, 2]
        backgroundMask = 1 - foregroundMask

        # Step 2 - Calculate intersection sums
        foregroundIntersectionSum = np.sum(foregroundMask & scribble)
        backgroundIntersectionSum = np.sum(backgroundMask & scribble)
        
        return bool(foregroundIntersectionSum > backgroundIntersectionSum)

    except:
        traceback.print_exc()
        raise


def scribbleGaussianize(scribbleMask, sampling=3, sigma=0.01):
    """
    Convert binary scribble mask to Gaussian distance map.
    
    Parameters
    ----------
    scribbleMask : np.ndarray, shape=(H, W)
        Binary scribble mask
    sampling : int, default=3
        Sampling rate for distance transform
    sigma : float, default=0.01
        Standard deviation of the Gaussian, as a fraction of the image diagonal
        (so sigma=0.01 on a 128x128 image is ~1.8 px)
        
    Returns
    -------
    np.ndarray
        Gaussian distance map
        
    Notes
    -----
    Uses Euclidean distance transform to compute distance of each pixel
    to the nearest scribble pixel, then applies Gaussian kernel. Distances are
    normalised by the image diagonal (a fixed reference) rather than by the
    per-image maximum distance, so the scribble width in the target does not
    depend on where the scribble happens to sit in the image.
    """
    gaussianDistanceMap = np.zeros_like(scribbleMask)

    try:
        # Step 0 - Validate input
        assert len(scribbleMask.shape) == 2, "Scribble mask must be 2D"
        if np.sum(scribbleMask) == 0:
            return gaussianDistanceMap
        if scribbleMask.dtype != bool:
            scribbleMask = scribbleMask.astype(bool)

        # Step 1 - Compute Euclidean distance map (in `sampling` units)
        euclideanDistanceMap: np.ndarray = scipy.ndimage.distance_transform_edt(  # type: ignore
            1 - scribbleMask, sampling=sampling
        )
        # Step 1.1 - Normalise by a fixed reference: the image diagonal in the same units
        referenceDistance = sampling * float(np.hypot(*scribbleMask.shape))
        normalisedDistance = euclideanDistanceMap / referenceDistance

        # Step 2 - Apply Gaussian kernel
        gaussianDistanceMap = np.exp(-normalisedDistance**2 / (2 * sigma**2)).astype(np.float32)

    except:
        traceback.print_exc()

    return gaussianDistanceMap


######################################################################
# HELPER FUNCTIONS - REAL DATA DISCOVERY AND SPLITTING
######################################################################

def collectRealImagePaths(pathsToRealData: List[Path]) -> List[str]:
    """
    Recursively collect real scribble images (``*.png``) from the given directories.

    Parameters
    ----------
    pathsToRealData : list[Path]
        Directories to scan

    Returns
    -------
    list[str]
        Sorted image paths (sorted so that a seeded split is reproducible)

    Raises
    ------
    ValueError
        If a path is missing, is not a directory, or no images were found
    """
    imagePaths: List[str] = []
    missingPaths, nonDirectoryPaths = [], []
    for pathToRealData in pathsToRealData:
        pathToRealData = Path(pathToRealData)
        if not pathToRealData.exists():
            missingPaths.append(str(pathToRealData))
        elif not pathToRealData.is_dir():
            nonDirectoryPaths.append(str(pathToRealData))
        else:
            imagePaths += [str(p) for p in pathToRealData.rglob('*.png')]

    if missingPaths or nonDirectoryPaths:
        errorParts = []
        if missingPaths:
            errorParts.append("missing data path(s): " + ", ".join(missingPaths))
        if nonDirectoryPaths:
            errorParts.append("non-directory path(s): " + ", ".join(nonDirectoryPaths))
        raise ValueError(
            "ScribbleDataset could not load real training data because " + "; ".join(errorParts)
        )
    if len(imagePaths) == 0:
        raise ValueError(
            "ScribbleDataset found no .png images under the configured real data path(s): "
            + ", ".join(str(p) for p in pathsToRealData)
        )
    return sorted(imagePaths)


def splitRealImagePaths(
    imagePaths: List[str], valFraction: float, seed: int
) -> tuple[List[str], List[str]]:
    """
    Deterministically split image paths into disjoint train / validation lists.

    Parameters
    ----------
    imagePaths : list[str]
        All real image paths
    valFraction : float
        Fraction held out for validation, in [0, 1). With 0 both lists contain
        every image (no hold-out; validation then measures training data).
    seed : int
        Seed for the shuffle, so the split is stable across runs

    Returns
    -------
    tuple[list[str], list[str]]
        (trainPaths, valPaths)
    """
    if valFraction <= 0:
        return list(imagePaths), list(imagePaths)
    shuffled = list(imagePaths)
    random.Random(seed).shuffle(shuffled)
    nVal = max(1, int(round(len(shuffled) * valFraction)))
    nVal = min(nVal, len(shuffled) - 1)  # always keep at least one training image
    return shuffled[nVal:], shuffled[:nVal]


######################################################################
# HELPER FUNCTIONS - BATCH COLLATION
######################################################################

def collateFn(
    batch: list[dict[str, Union[int, float, torch.Tensor]]],
) -> dict[str, Union[list, torch.Tensor]]:
    """
    Custom collate function to filter out samples with insufficient scribbles.
    
    Parameters
    ----------
    batch : list[dict]
        List of dataset samples
        
    Returns
    -------
    dict
        Collated batch dictionary
        
    Notes
    -----
    Filters out samples where scribble sum < configConstants.MIN_SUM_SCRIBBLE.
    Returns an empty dict when every sample was filtered; consumers must skip it.
    """
    batchDict: dict[str, Union[list, torch.Tensor]] = {}
    
    # Step 1 - Filter batch
    filteredBatch = [
        item for item in batch
        if isinstance(item[configConstants.KEY_SCRIBBLE], torch.Tensor) and np.sum(item[configConstants.KEY_SCRIBBLE].numpy()) >= configConstants.MIN_SUM_SCRIBBLE # type: ignore
    ]
    
    if len(filteredBatch) == 0:
        return batchDict

    # Step 2 - Stack tensors or collect lists
    for key in list(filteredBatch[0].keys()):
        if isinstance(filteredBatch[0][key], torch.Tensor):
            tensorList: list[torch.Tensor] = [item[key] for item in filteredBatch if isinstance(item[key], torch.Tensor)]  # type: ignore
            batchDict[key] = torch.stack(tensorList)
        else:
            batchDict[key] = [item[key] for item in filteredBatch]

    return batchDict


######################################################################
# MAIN DATASET CLASS
######################################################################

class ScribbleDataset(torch.utils.data.Dataset):
    """
    Dataset for scribble-based segmentation.
    
    Supports three data sources:
    1. Real scribbles from disk
    2. Synthetic scribbles with Perlin noise augmentation
    3. Fully synthetic shape-based scribbles
    """

    def __init__(
        self,
        imageSize: tuple[int, int],
        realProbability: float,
        pathsToRealData: List[Path],
        transforms: Optional[torchvision.transforms.Compose] = None,
        generateComponents: bool = False,
        valMode: bool = False,
        verbose: bool = False,
        shape: str = configConstants.SHAPE_VALUE_ANY,
        maxNumberOfRealImages: int = 20000,
        syntheticPerlinGeneratedProbability: float = 0.0,
        saveImagePath: Optional[str] = None,
        computeScribbleGauss: bool = True,
        realImagePaths: Optional[List[str]] = None,
    ):
        """
        Initialize ScribbleDataset.
        
        Parameters
        ----------
        imageSize : tuple[int, int]
            (height, width) of images
        realProbability : float
            Probability of sampling real vs synthetic data
        pathsToRealData : List[Path]
            Directories containing real scribble images
        transforms : torchvision.transforms.Compose, optional
            Data augmentation transforms
        generateComponents : bool, default=False
            Whether to generate connected components
        valMode : bool, default=False
            Validation mode flag
        verbose : bool, default=False
            Enable verbose logging
        shape : str, default='any'
            Shape type for synthetic data
        maxNumberOfRealImages : int, default=20000
            Maximum number of real images to load
        syntheticPerlinGeneratedProbability : float, default=0.0
            Probability of Perlin noise augmentation
        saveImagePath : str, optional
            Path to save debug images
        computeScribbleGauss : bool, default=True
            Whether to compute the Gaussian scribble distance map. This runs a
            full Euclidean distance transform per sample, so skip it when the
            consumer only needs the binary scribble (e.g. a training set with
            scribbleGauss=False).
        realImagePaths : list[str], optional
            Explicit list of real image files to use instead of scanning
            ``pathsToRealData``. Use this to hand a train/val split to two
            dataset instances (see ``collectRealImagePaths`` / ``splitRealImagePaths``).
        """
        # Step 0 - Store configuration
        self.computeScribbleGauss = computeScribbleGauss
        self.imageSize = imageSize
        self.realProbability = realProbability
        self.pathsToRealData = pathsToRealData
        self.transforms = transforms
        self.valMode = valMode
        self.verbose = verbose
        self.shape = shape
        self.saveImagePath = saveImagePath
        self.generateComponents = generateComponents

        # Step 1 - Load real data paths (explicit list wins over directory scan)
        if realImagePaths is not None:
            self.realDataImagePaths = list(realImagePaths)
        else:
            self.realDataImagePaths = collectRealImagePaths(self.pathsToRealData)

        # Step 2 - Limit and shuffle real data paths
        if len(self.realDataImagePaths) > maxNumberOfRealImages:
            self.realDataImagePaths = self.realDataImagePaths[:maxNumberOfRealImages]

        if len(self.realDataImagePaths) == 0:
            raise ValueError(
                "ScribbleDataset found no .png images under the configured real data path(s): "
                + ", ".join(str(p) for p in self.pathsToRealData)
            )

        nRealImages = len(self.realDataImagePaths)
        random.shuffle(self.realDataImagePaths)
        
        # Step 3 - Calculate number of synthetic samples needed
        # Balance dataset based on realProbability ratio
        self.nSynthetic = (
            int(nRealImages / realProbability) - nRealImages
            if realProbability > 0 else nRealImages
        )
        
        if syntheticPerlinGeneratedProbability > 0:
            self.nSyntheticPerlin = int(
                self.nSynthetic * syntheticPerlinGeneratedProbability
            )
            self.nSynthetic = int(
                self.nSynthetic * (1 - syntheticPerlinGeneratedProbability)
            )
        else:
            self.nSyntheticPerlin = 0
    
        if realProbability == 0:
            self.realDataImagePaths = []

        # Step 4 - Log transforms (parameters are sampled per-sample in _applyTransforms)
        print(f' - [ScribbleDataset][val={valMode}] Handling transforms: {self.transforms}')

    def __len__(self) -> int:
        """Return total dataset size."""
        return len(self.realDataImagePaths) + self.nSynthetic + self.nSyntheticPerlin

    def _scribbleGauss(self, scribbleBool: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian scribble distance map, or a zero map when disabled.

        The Euclidean distance transform inside scribbleGaussianize() is one of the
        most expensive per-sample operations, so it is skipped entirely (returning a
        zero map of matching shape) when the consumer does not need it.

        Parameters
        ----------
        scribbleBool : np.ndarray, shape=(H, W)
            Binary scribble mask.

        Returns
        -------
        np.ndarray, shape=(H, W)
            Gaussian distance map, or zeros when computeScribbleGauss is False.
        """
        if self.computeScribbleGauss:
            return scribbleGaussianize(scribbleBool)
        return np.zeros(scribbleBool.shape, dtype=np.float32)

    def _applyTransforms(self, gt, prediction, scribble, scribbleGauss, meta):
        """
        Apply data augmentation transforms.
        
        Parameters
        ----------
        gt, prediction, scribble, scribbleGauss : torch.Tensor
            Input tensors to transform
        meta : str
            Metadata for logging
            
        Returns
        -------
        tuple
            Transformed tensors and applied transforms dict
        """
        transformsApplied = {}

        # Random parameters are drawn ONCE per sample and the identical functional op is
        # applied to every tensor. Calling a Random* transform object on each tensor in
        # turn would re-roll its randomness and desynchronise gt / prediction / scribble.
        try:
            if self.transforms is not None:
                for transform in self.transforms.transforms:
                    try:
                        transformType = type(transform)
                        if transformType == torchvision.transforms.transforms.RandomRotation:
                            lowDeg, highDeg = transform.degrees
                            randomDegrees = random.uniform(float(lowDeg), float(highDeg))
                            transformsApplied[configConstants.KEY_ROTATE] = randomDegrees
                            op = lambda x: TF.rotate(x, randomDegrees, interpolation=transform.interpolation,
                                                     expand=transform.expand, center=transform.center, fill=0)
                        elif transformType == torchvision.transforms.transforms.RandomHorizontalFlip:
                            if random.random() >= transform.p:
                                continue
                            transformsApplied[configConstants.KEY_FLIP_HOR] = True
                            op = TF.hflip
                        elif transformType == torchvision.transforms.transforms.RandomVerticalFlip:
                            if random.random() >= transform.p:
                                continue
                            transformsApplied[configConstants.KEY_FLIP_VER] = True
                            op = TF.vflip
                        elif transformType == torchvision.transforms.transforms.RandomAffine:
                            imgSize = [int(gt.shape[-1]), int(gt.shape[-2])]
                            angle, translations, scale, shear = transform.get_params(
                                transform.degrees, transform.translate, transform.scale, transform.shear, imgSize
                            )
                            transformsApplied[configConstants.KEY_TRANS] = list(translations)
                            op = lambda x: TF.affine(x, angle=angle, translate=list(translations), scale=scale,
                                                     shear=list(shear), interpolation=transform.interpolation, fill=0)
                        else:
                            # Deterministic transforms (e.g. CenterCrop) can be applied directly
                            op = transform

                        gt, prediction, scribble, scribbleGauss = (
                            op(gt), op(prediction), op(scribble), op(scribbleGauss)
                        )

                    except Exception:
                        print(f' - [ERROR][{meta}] Error in transform: {transform}')
                        traceback.print_exc()

        except Exception:
            print(f' - [ERROR][{meta}] Error in _applyTransforms()')
            traceback.print_exc()
            
        return gt, prediction, scribble, scribbleGauss, transformsApplied

    def _getSyntheticBadoutput(self):
        """Return empty sample for failed synthetic generation."""
        return {
            configConstants.KEY_GT: torch.zeros(self.imageSize, dtype=torch.float32).unsqueeze(0),
            configConstants.KEY_PREDICTION: torch.zeros(self.imageSize, dtype=torch.float32).unsqueeze(0),
            configConstants.KEY_SCRIBBLE_TYPE: -1,
            configConstants.KEY_SCRIBBLE_CLASS: torch.tensor(0, dtype=torch.long),  # Default to background
            configConstants.KEY_SCRIBBLE: torch.zeros(self.imageSize, dtype=torch.float32).unsqueeze(0),
            configConstants.KEY_SCRIBBLE_GAUSS: torch.zeros(self.imageSize, dtype=torch.float32).unsqueeze(0),
            configConstants.KEY_META: {
                configConstants.KEY_SCRIBBLE_GENERATION_TYPE: '',
                configConstants.KEY_SCRIBBLE_PATH: '',
                configConstants.KEY_TRANSFORMS: {},
                configConstants.KEY_DATA_SOURCE: configConstants.DATA_SOURCE_SYNTHETIC_SHAPE,
            },
        }

    def _loadRealData(self, index: int) -> dict[str, Union[int, float, torch.Tensor, dict]]:
        """
        Load real scribble data from disk.
        
        Parameters
        ----------
        index : int
            Index in real data array
            
        Returns
        -------
        dict
            Sample dictionary with GT, prediction, scribble, etc.
        """
        # Step 1 - Load image
        imagePath = Path(self.realDataImagePaths[index])
        realImage = cv2.cvtColor(cv2.imread(str(imagePath)), cv2.COLOR_BGR2RGB)
        
        # Step 2 - Extract channels (R=Pred, G=GT, B=Scribble)
        tensorGt = torch.tensor(realImage[:, :, 1], dtype=torch.uint8).unsqueeze(0)
        tensorPred = torch.tensor(realImage[:, :, 0], dtype=torch.uint8).unsqueeze(0)
        tensorScribbleBinary = torch.tensor(realImage[:, :, 2], dtype=torch.uint8).unsqueeze(0)
        tensorScribbleGauss = torch.tensor(
            self._scribbleGauss(realImage[:, :, 2].astype(bool)),
            dtype=torch.float32
        ).unsqueeze(0)
        
        scribbleType = getScribbleType(realImage)
        scribbleMetaPathstub = '/'.join(imagePath.parts[-2:])
        
        # Step 2.1 - Determine scribble class (foreground vs background)
        scribbleClassValue = determineScribbleClass(
            realImage[:, :, 2],  # Scribble channel
            realImage[:, :, 1],  # GT channel
            realImage[:, :, 0]   # Prediction channel
        )

        # Step 3 - Apply transforms
        tensorGt, tensorPred, tensorScribbleBinary, tensorScribbleGauss, transformsApplied = \
            self._applyTransforms(
                tensorGt, tensorPred, tensorScribbleBinary,
                tensorScribbleGauss, scribbleMetaPathstub
            )

        # Step 4 - Return sample
        return {
            configConstants.KEY_GT: tensorGt.to(torch.bool).to(torch.float32),
            configConstants.KEY_PREDICTION: tensorPred.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_TYPE: int(scribbleType),
            configConstants.KEY_SCRIBBLE_CLASS: torch.tensor(scribbleClassValue, dtype=torch.long),
            configConstants.KEY_SCRIBBLE: tensorScribbleBinary.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_GAUSS: tensorScribbleGauss.to(torch.float32),
            configConstants.KEY_META: {
                configConstants.KEY_SCRIBBLE_GENERATION_TYPE: configConstants.SCRIBBLE_TYPE_HUMAN,
                configConstants.KEY_SCRIBBLE_PATH: '/'.join(imagePath.parts[-2:]),
                configConstants.KEY_TRANSFORMS: transformsApplied,
                configConstants.KEY_DATA_SOURCE: configConstants.DATA_SOURCE_REAL,
            },
        }

    def _loadPerlinSyntheticData(self, index: int) -> dict[str, Union[int, float, torch.Tensor, dict]]:
        """
        Load real data and apply Perlin noise augmentation to scribbles.
        
        Parameters
        ----------
        index : int
            Index in synthetic array
            
        Returns
        -------
        dict
            Sample dictionary with augmented scribbles
        """
        # Step 1 - Pick a real image
        useRealPathImage = index % len(self.realDataImagePaths)
        imagePath = Path(self.realDataImagePaths[useRealPathImage])
        realImage = cv2.cvtColor(cv2.imread(str(imagePath)), cv2.COLOR_BGR2RGB)
        
        tensorGt = torch.tensor(realImage[:, :, 1], dtype=torch.uint8).unsqueeze(0)
        tensorPred = torch.tensor(realImage[:, :, 0], dtype=torch.uint8).unsqueeze(0)
        scribbleBinary = realImage[:, :, 2]
                            
        # Step 2 - Generate Perlin noise mask
        noiseMask = (
            voxynth.noise.perlin(
                shape=[scribbleBinary.shape[0], scribbleBinary.shape[1]],
                smoothing=np.random.uniform(2, 14),
                magnitude=1,
            )
            .cpu()
            .numpy()
            > 0
        )

        # Step 3 - Apply noise mask to scribble
        scribbleBinaryMasked = noiseMask * scribbleBinary

        # Step 4 - Retry if empty
        if np.sum(scribbleBinaryMasked) == 0:
            noiseMask2 = (
                voxynth.noise.perlin(
                    shape=[scribbleBinary.shape[0], scribbleBinary.shape[1]],
                    smoothing=np.random.uniform(2, 14),
                    magnitude=1,
                )
                .cpu()
                .numpy()
                > 0
            )
            scribbleBinaryMasked = noiseMask2 * scribbleBinary
            if np.sum(scribbleBinaryMasked) == 0:
                scribbleBinaryMasked = scribbleBinary
            noiseMask = noiseMask2

        scribbleBinary = np.round(scribbleBinaryMasked).astype(np.uint8)

        # Step 5 - Save debug images if requested
        if self.saveImagePath is not None:
            saveImage(np.round(noiseMask * 255).astype(np.uint8), self.saveImagePath, f"{index}_noise.png")
            saveImage(np.round(scribbleBinary).astype(np.uint8), self.saveImagePath, f"{index}_binary.png")
            saveImage(np.round(scribbleBinary).astype(np.uint8), self.saveImagePath, f"{index}_binary_masked.png")

        # Step 6 - Convert to tensors
        tensorScribbleBinary = torch.tensor(scribbleBinaryMasked, dtype=torch.uint8).unsqueeze(0)
        tensorScribbleGauss = torch.tensor(
            self._scribbleGauss(scribbleBinaryMasked.astype(bool)),
            dtype=torch.float32
        ).unsqueeze(0)
        scribbleType = getScribbleType(realImage)
        scribbleMetaPathstub = '/'.join(imagePath.parts[-2:])
        
        # Step 6.1 - Determine scribble class
        scribbleClassValue = determineScribbleClass(
            scribbleBinaryMasked,
            tensorGt.squeeze().numpy(),
            tensorPred.squeeze().numpy()
        )

        # Step 7 - Apply transforms
        tensorGt, tensorPred, tensorScribbleBinary, tensorScribbleGauss, transformsApplied = \
            self._applyTransforms(
                tensorGt, tensorPred, tensorScribbleBinary,
                tensorScribbleGauss, scribbleMetaPathstub
            )

        # Step 8 - Return sample
        return {
            configConstants.KEY_GT: tensorGt.to(torch.bool).to(torch.float32),
            configConstants.KEY_PREDICTION: tensorPred.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_TYPE: int(scribbleType),
            configConstants.KEY_SCRIBBLE_CLASS: torch.tensor(scribbleClassValue, dtype=torch.long),
            configConstants.KEY_SCRIBBLE: tensorScribbleBinary.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_GAUSS: tensorScribbleGauss.to(torch.float32),
            configConstants.KEY_META: {
                configConstants.KEY_SCRIBBLE_GENERATION_TYPE: configConstants.SCRIBBLE_TYPE_HUMAN,
                configConstants.KEY_SCRIBBLE_PATH: '/'.join(imagePath.parts[-2:]),
                configConstants.KEY_TRANSFORMS: transformsApplied,
                configConstants.KEY_DATA_SOURCE: configConstants.DATA_SOURCE_SYNTHETIC_PERLIN,
            },
        }

    def _loadFullySyntheticData(self, index: int) -> dict[str, Union[int, float, torch.Tensor, dict]]:
        """
        Generate fully synthetic data from shapes.
        
        Parameters
        ----------
        index : int
            Index in fully synthetic array
            
        Returns
        -------
        dict
            Sample dictionary with synthetic shapes and scribbles
        """
        # Step 1 - Generate GT and prediction shapes
        gt = generateShapes(self.imageSize, 1, self.shape)[0]
        prediction = generateShapes(self.imageSize, 1, self.shape)[0]

        # Step 2 - Compute error regions (TP, FP, FN)
        tp = (gt > 0) & (prediction > 0)
        fp = (prediction > 0) & ~tp
        fn = (gt > 0) & ~tp

        # Step 3 - Randomly choose scribble type and error mask
        # scribbleType: 0=FP (background), 1=FN (foreground)
        scribbleType = random.choice([0, 1])
        errorMask = fn if scribbleType else fp

        # Step 4 - Generate scribble from error mask (falls back to the other region if empty)
        if random.random() > 0.5:
            scribble = getContourScribbleForBinarySlice(errorMask, False)[1]
            generationType = configConstants.SCRIBBLE_TYPE_CONTOUR
            if scribble is None:
                scribbleType = 1 - scribbleType
                errorMask = fn if scribbleType else fp
                scribble = getContourScribbleForBinarySlice(errorMask, False)[1]
        else:
            generationType = random.choice([
                configConstants.SCRIBBLE_TYPE_MEDIAL,
                configConstants.SCRIBBLE_TYPE_SKELETON
            ])
            scribble = getMorphologyScribbleForBinarySlice(
                errorMask, generationType, valMode=self.valMode
            )[1]
            if scribble is None:
                scribbleType = 1 - scribbleType
                errorMask = fn if scribbleType else fp
                scribble = getMorphologyScribbleForBinarySlice(
                    errorMask, generationType, valMode=self.valMode
                )[1]

        if scribble is None:
            if self.verbose:
                print(f" - [INFO] Skipping synthetic scribble generation for index {index}: {generationType}")
            return self._getSyntheticBadoutput()

        # The class label must describe the region the scribble was finally drawn on,
        # so derive it from scribbleType only AFTER the fallback above.
        scribbleClassValue = int(scribbleType)

        # Step 5 - Refine scribble: keep only the largest connected component.
        # Components are found on a dilated copy so that near-touching fragments count as one.
        dilatedScribble = binary_dilation(scribble)
        labelResult = ndi.label(dilatedScribble)
        components: np.ndarray = labelResult[0]  # type: ignore
        nComponents: int = labelResult[1]  # type: ignore
        scribbleFinal: np.ndarray = scribble.copy()  # type: ignore
        if nComponents > 1:
            values, counts = np.unique(components, return_counts=True)
            # values[0] is background (label 0); pick the largest FOREGROUND label
            componentInd = values[1:][np.argmax(counts[1:])]
            scribbleFinal[components != componentInd] = 0

        # Step 6 - Convert to tensors
        tensorGt = torch.tensor(gt, dtype=torch.uint8).unsqueeze(0)
        tensorPred = torch.tensor(prediction, dtype=torch.uint8).unsqueeze(0)
        tensorScribbleBinary = torch.tensor(scribbleFinal, dtype=torch.uint8).unsqueeze(0)
        tensorScribbleGauss = torch.tensor(
            self._scribbleGauss(scribbleFinal.astype(bool)),
            dtype=torch.float32
        ).unsqueeze(0)

        # Step 7 - Apply the same augmentation as the real/Perlin loaders
        tensorGt, tensorPred, tensorScribbleBinary, tensorScribbleGauss, transformsApplied = \
            self._applyTransforms(
                tensorGt, tensorPred, tensorScribbleBinary,
                tensorScribbleGauss, f"synthetic-{generationType}-{index}"
            )

        return {
            configConstants.KEY_GT: tensorGt.to(torch.bool).to(torch.float32),
            configConstants.KEY_PREDICTION: tensorPred.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_TYPE: int(scribbleType),
            configConstants.KEY_SCRIBBLE_CLASS: torch.tensor(scribbleClassValue, dtype=torch.long),
            configConstants.KEY_SCRIBBLE: tensorScribbleBinary.to(torch.bool).to(torch.float32),
            configConstants.KEY_SCRIBBLE_GAUSS: tensorScribbleGauss,
            configConstants.KEY_META: {
                configConstants.KEY_SCRIBBLE_GENERATION_TYPE: generationType,
                configConstants.KEY_SCRIBBLE_PATH: '',
                configConstants.KEY_TRANSFORMS: transformsApplied,
                configConstants.KEY_DATA_SOURCE: configConstants.DATA_SOURCE_SYNTHETIC_SHAPE,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Union[int, float, torch.Tensor, dict]]:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        index : int
            Sample index
            
        Returns
        -------
        dict
            Sample dictionary containing GT, prediction, scribble, etc.
        """
        # Step 1 - Real data
        if index < len(self.realDataImagePaths):
            return self._loadRealData(index)
        
        # Step 2 - Perlin-augmented synthetic data
        elif index < (len(self.realDataImagePaths) + self.nSyntheticPerlin):
            return self._loadPerlinSyntheticData(index)
        
        # Step 3 - Fully synthetic data
        else:
            index -= len(self.realDataImagePaths) + self.nSyntheticPerlin
            return self._loadFullySyntheticData(index)

######################################################################
# TESTING / DEBUGGING
######################################################################

def main():
    """Test dataset loading and visualization."""
    try:
        # Step 1 - Configure dataset
        imageSize = (128, 128)
        realProbability = 0.5
        syntheticPerlinGeneratedProbability = 0
        pathsToRealData = [
            DIR_DATA / "scribble-interactions-v1",
            DIR_DATA / "scribble-interactions-v2",
            DIR_DATA / "scribble-interactions-v3"
        ]

        transformsList = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(imageSize),
            torchvision.transforms.RandomHorizontalFlip(p=1.0),
            torchvision.transforms.RandomVerticalFlip(p=1.0),
            torchvision.transforms.RandomRotation(degrees=360),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
        ])
        batchSize = 16

        # Step 2 - Create dataset and dataloader
        dataset = ScribbleDataset(
            imageSize, realProbability, pathsToRealData,
            transforms=transformsList,
            valMode=False,
            verbose=True,
            generateComponents=True,
            saveImagePath=str(DIR_TMP / "med_noised"),
            syntheticPerlinGeneratedProbability=syntheticPerlinGeneratedProbability,
            shape=configConstants.SHAPE_VALUE_ANY
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, collate_fn=collateFn, shuffle=True
        )

        # Step 3 - Iterate and visualize
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for batch in dataloader:
                gt = batch[configConstants.KEY_GT]
                prediction = batch[configConstants.KEY_PREDICTION]
                scribbleType = batch[configConstants.KEY_SCRIBBLE_TYPE]
                scribble = batch[configConstants.KEY_SCRIBBLE]
                scribbleGauss = batch[configConstants.KEY_SCRIBBLE_GAUSS]
                meta = batch[configConstants.KEY_META]
                
                assert gt.shape == prediction.shape == scribble.shape == scribbleGauss.shape, \
                    f"Shapes do not match: gt={gt.shape}, pred={prediction.shape}, scribble={scribble.shape}, gauss={scribbleGauss.shape}"

                utils.plotScribbleDatasetBatch(gt, prediction, scribble, scribbleGauss, scribbleType, meta, DIR_TMP)
                pbar.update(len(gt))

    except:
        traceback.print_exc()


if __name__ == "__main__":
    main()

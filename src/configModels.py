######################################################################
# CONFIGURATION MODELS - Pydantic Schema Validation
######################################################################
"""
Pydantic models for validating configuration files used in training.

These models ensure type safety and provide validation for configuration
parameters loaded from JSON files.

Classes:
    ImageSize: Image dimensions configuration
    SchedulerParams: Scheduler hyperparameters
    DataParams: Dataset configuration
    ModelParams: Model architecture configuration
    TrainParams: Training hyperparameters
    Config: Complete configuration combining all sections
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
from typing import List, Literal, Optional

# Pip/Third-party installs
from pydantic import BaseModel, Field

######################################################################
# NESTED CONFIGURATION MODELS
######################################################################

class ImageSize(BaseModel):
    """Image dimensions configuration."""
    width: int = Field(default=128, gt=0, description="Image width in pixels")
    height: int = Field(default=128, gt=0, description="Image height in pixels")


class SchedulerParams(BaseModel):
    """Variance scheduler hyperparameters for diffusion process."""
    beta1: float = Field(default=1e-4, gt=0, description="Beta value at timestep 1")
    betaT: float = Field(default=0.2, gt=0, description="Beta value at timestep T")
    s: float = Field(default=0.008, ge=0, description="Smoothing parameter for cosine scheduler")
    maxBeta: float = Field(default=0.999, gt=0, le=1, description="Maximum beta value")


class ScribbleClassLossWeights(BaseModel):
    """Class-aware loss multipliers (target region depends on scribbleClass).

    The target (encourage) region is where a scribble of the given class lives
    in the data; the non-target (discourage) region is every other pixel:

    - Foreground scribble (class 1): target = FN ∪ TP (the whole GT foreground)
    - Background scribble (class 0): target = FP (the prediction's false positives)
    """
    targetRegionLossWeight: float = Field(
        default=1.0, gt=0,
        description="Weight on the class-dependent target region (FN∪TP for FG, FP for BG)"
    )
    nonTargetRegionLossWeight: float = Field(
        default=1.0, ge=0,
        description="Weight on all non-target regions (everything else), suppressed uniformly"
    )


######################################################################
# MAIN CONFIGURATION SECTIONS
######################################################################

class DataParams(BaseModel):
    """Dataset configuration parameters."""
    imageSize: ImageSize = Field(default_factory=ImageSize, description="Target image dimensions")
    pathsToRealData: List[str] = Field(
        default=["/data"],
        description="Paths to real scribble data directories"
    )
    realProbability: float = Field(default=0.5, ge=0, le=1, description="Probability of sampling real vs synthetic data")
    syntheticPerlinGeneratedProbability: float = Field(
        default=0.0, ge=0, le=1, 
        description="Probability of Perlin noise augmentation for synthetic data"
    )
    shape: str = Field(default="any", description="Filter by shape type (any, circle, rectangle, triangle)")
    valFraction: float = Field(
        default=0.2, ge=0, lt=1,
        description=(
            "Fraction of the real images held out for validation (deterministic split, seeded by "
            "trainParams.seed). 0 disables the hold-out, i.e. validation runs on training images."
        )
    )


class ModelParams(BaseModel):
    """Model architecture and initialization configuration."""
    conditional: bool = Field(default=True, description="Whether to use conditional diffusion")
    imageChannels: int = Field(
        default=1, gt=0,
        description="Channels of the scribble sample (UNet in/out channels and ControlNet sample input)"
    )
    nChannels: int = Field(
        default=64, gt=0,
        description="DEPRECATED - not used by duUnet4Blocks; width is set by blockOutChannels"
    )
    model: str = Field(default="duUnet4Blocks", description="Model architecture identifier")
    blockOutChannels: List[int] = Field(
        default=[128, 128, 256, 256],
        description=(
            "Channel count per encoder/decoder block (4 entries for duUnet4Blocks). "
            "Shared by UNet and ControlNet so the residual streams stay aligned. "
            "Each value must be divisible by normNumGroups."
        )
    )
    layersPerBlock: int = Field(
        default=2, gt=0, description="Number of ResNet layers per encoder/decoder block"
    )
    normNumGroups: int = Field(
        default=32, gt=0,
        description=(
            "GroupNorm groups for every ResNet/attention norm. Must divide every value in "
            "blockOutChannels. Keep ~4-16 channels-per-group; avoid groups == channels "
            "(degenerates to InstanceNorm)."
        )
    )


class TrainParams(BaseModel):
    """Training hyperparameters and configuration."""
    scribbleGauss: bool = Field(default=True, description="Use Gaussian scribbles instead of binary")
    numWorkers: int = Field(default=4, ge=0, description="Number of data loading workers")
    batchSize: int = Field(default=16, gt=0, description="Training batch size")
    gradientAccumulationSteps: int = Field(default=1, gt=0, description="Gradient accumulation steps")
    learningRate: float = Field(default=1e-4, gt=0, description="Initial learning rate")
    lrWarmupSteps: int = Field(default=0, ge=0, description="Learning rate warmup steps")
    logEvery: int = Field(default=50, gt=0, description="Logging frequency in iterations")
    logNImages: int = Field(default=2, ge=0, description="Number of images to log")
    loss: str = Field(default="l2", description="Loss function type (l1, l2)")
    mixedPrecision: str = Field(default="bf16", description="Mixed precision training mode")
    numEpochs: int = Field(default=300, gt=0, description="Total number of training epochs")
    savePath: str = Field(default="/workspace/results", description="Directory to save checkpoints and results")
    schedulerArgs: SchedulerParams = Field(default_factory=SchedulerParams, description="Variance scheduler configuration")
    schedulerType: str = Field(default="cosine", description="Scheduler type (linear, cosine)")
    steps: int = Field(default=50, gt=0, description="Number of diffusion timesteps")
    logWandb: bool = Field(default=True, description="Enable Weights & Biases logging")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    wandbProject: str = Field(default="diffusion-scribble-segmentation", description="W&B project name")
    wandbName: Optional[str] = Field(
        default=None,
        description="W&B run name (a timestamp is appended). When omitted the config file stem is used."
    )
    wandbNotes: str = Field(
        default="Run notes",
        description="W&B run notes/description"
    )
    clipNorm: float = Field(default=-1, description="Gradient clipping norm (-1 to disable)")
    optimizer: str = Field(default="AdamW", description="Optimizer type (Adam, AdamW)")
    evalScribbleClassOverride: Optional[int] = Field(
        default=None, 
        ge=0, 
        le=1,
        description="Override scribble class during evaluation (0=background, 1=foreground, None=use batch class)"
    )
    evalSaveEveryNImages: int = Field(
        default=1000,
        ge=0,
        description="Save inference plots every N evaluated images during validation (0 disables saving)"
    )
    evalSaveMode: Literal["stochastic", "deterministic"] = Field(
        default="stochastic",
        description="Evaluation image save mode: stochastic uses probability batchSize/N, deterministic saves on exact N-image boundaries"
    )
    valShuffle: bool = Field(
        default=True,
        description="Shuffle validation dataloader to mix real and synthetic samples across batches"
    )
    lossClassWeights: ScribbleClassLossWeights = Field(
        default_factory=ScribbleClassLossWeights,
        description="Class-aware target/non-target region loss weights"
    )


######################################################################
# COMPLETE CONFIGURATION MODEL
######################################################################

class Config(BaseModel):
    """Complete training configuration combining all sections."""
    dataParams: DataParams = Field(..., description="Dataset configuration")
    modelParams: ModelParams = Field(..., description="Model architecture configuration")
    trainParams: TrainParams = Field(..., description="Training hyperparameters")
    configPath: Optional[str] = Field(default=None, description="Path to config file (set at runtime)")

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields like configPath for runtime metadata

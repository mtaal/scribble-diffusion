######################################################################
# DIFFUSION PROCESS - DDPM Implementation
######################################################################
"""
This module implements a diffusion-based model for morphologically dependent
interactive scribble segmentation. It includes classes and functions for
variance scheduling, loss computation, and forward/backward diffusion processes.

Classes:
    Diffusion: Implements the diffusion process, including forward and backward
              diffusion, loss computation, and sampling.

Functions:
    computeVarianceSchedule(): Computes the variance schedule for the diffusion process.
    errorRegionFocusedLoss(): Implements a custom loss function that focuses on
                            error regions in the input image.

References:
    - DDPM Paper: https://arxiv.org/pdf/2006.11239
    - Improved DDPM Paper: https://arxiv.org/pdf/2102.09672
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import argparse
import json
from pathlib import Path
from typing import Optional

# Pip/Third-party installs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from torchinfo import summary

try:
    from skimage import feature as skfeature
except ImportError:  # pragma: no cover - optional dependency
    skfeature = None

# Application/Local module imports
import src.configConstants as configConstants
from src.models.duUnetBlocks4 import Diffusion4BlocksUNet
from src.models.schedule import computeVarianceSchedule  # re-exported for callers
from src.configModels import Config, TrainParams, DataParams, ModelParams

######################################################################
# CONSTANTS AND CONFIGURATION
######################################################################

LOSS_DEBUG_SAVE_PROBABILITY = 1.0 / 1000.0


def saveClassBasedLossDebugGrid(
    gt: torch.Tensor,
    prediction: torch.Tensor,
    generatedScribble: torch.Tensor,
    originalScribble: torch.Tensor,
    weightHeatmap: torch.Tensor,
    epoch: int,
    batchIndex: int,
    outputDir: Path,
    sampleIndex: int = 0,
    scribbleClass: int = -1,
) -> None:
    """Save a 5-panel debug grid for class-based loss visualization.

    Panels:
      1) GT (green)
      2) Prediction (red)
      3) GT + Prediction (combined)
        4) FP/FN + scribbles
            FP=red, FN=green, generated scribble=red, original scribble=blue
      5) Weight heat map (from loss function)
    """
    try:
        outputDir.mkdir(parents=True, exist_ok=True)

        gtSample = gt[sampleIndex].detach().float().cpu().squeeze().numpy()
        predSample = prediction[sampleIndex].detach().float().cpu().squeeze().numpy()
        generatedScribbleSample = (
            generatedScribble[sampleIndex].detach().float().cpu().squeeze().numpy()
        )
        originalScribbleSample = (
            originalScribble[sampleIndex].detach().float().cpu().squeeze().numpy()
        )
        weightSample = (
            weightHeatmap[sampleIndex].detach().float().cpu().squeeze().numpy()
        )

        gtMask = gtSample > 0.5
        predMask = predSample > 0.5
        fpMask = predMask & (~gtMask)
        fnMask = gtMask & (~predMask)

        gtRgb = np.zeros((*gtMask.shape, 3), dtype=np.float32)
        gtRgb[..., 1] = gtMask.astype(np.float32)

        predRgb = np.zeros((*predMask.shape, 3), dtype=np.float32)
        predRgb[..., 1] = predMask.astype(np.float32)  # Changed to green channel

        combinedRgb = np.zeros((*gtMask.shape, 3), dtype=np.float32)
        # Both GT and prediction in green channel (0.5 each)
        combinedRgb[..., 1] = (gtMask.astype(np.float32) * 0.5 + predMask.astype(np.float32) * 0.5)

        # Draw yellow contour on GT boundary when skimage is available.
        if skfeature is not None:
            gt_edges = skfeature.canny(gtMask.astype(float), sigma=1.0)
        else:
            gt_edges = np.zeros_like(gtMask, dtype=bool)
        combinedRgb[gt_edges, 0] = 1.0  # Red channel
        combinedRgb[gt_edges, 1] = 1.0  # Green channel (Red + Green = Yellow)

        fpFnRgb = np.zeros((*gtMask.shape, 3), dtype=np.float32)
        fpFnRgb[..., 0] = fpMask.astype(np.float32)
        fpFnRgb[..., 1] = fnMask.astype(np.float32)

        generatedScribbleMask = generatedScribbleSample > 0.5
        originalScribbleMask = originalScribbleSample > 0.5

        overlayRgb = np.zeros((*gtMask.shape, 3), dtype=np.float32)
        overlayRgb[..., 0] = fpMask.astype(np.float32)
        overlayRgb[..., 1] = fnMask.astype(np.float32)
        # Generated scribble in red
        overlayRgb[..., 0] = np.maximum(overlayRgb[..., 0], generatedScribbleMask.astype(np.float32))
        # Original scribble in blue
        overlayRgb[..., 2] = np.maximum(overlayRgb[..., 2], originalScribbleMask.astype(np.float32))

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        axes[0].imshow(gtRgb)
        axes[0].set_title("GT (Green)")
        axes[0].axis("off")

        axes[1].imshow(predRgb)
        axes[1].set_title("Pred (Green)")
        axes[1].axis("off")

        axes[2].imshow(combinedRgb)
        axes[2].set_title("GT + Pred (Both Green) + GT Contour (Yellow)")
        axes[2].axis("off")

        axes[3].imshow(overlayRgb)
        axes[3].set_title("FP/FN + Scribbles")
        axes[3].axis("off")

        legendHandles = [
            Patch(facecolor="red", edgecolor="red", label="Generated Scribble"),
            Patch(facecolor="blue", edgecolor="blue", label="Original Scribble"),
        ]
        axes[3].legend(
            handles=legendHandles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            fontsize=8,
            framealpha=0.8,
            ncol=2,
        )

        # Explicitly set vmin/vmax to avoid colorbar state confusion
        vmin, vmax = float(np.min(weightSample)), float(np.max(weightSample))
        heatmap = axes[4].imshow(weightSample, cmap=plt.get_cmap("Oranges"), vmin=vmin, vmax=vmax)
        axes[4].set_title("Weight Heat Map")
        axes[4].axis("off")
        fig.colorbar(heatmap, ax=axes[4], fraction=0.046, pad=0.04)

        className = "Foreground" if scribbleClass == 1 else "Background" if scribbleClass == 0 else "Unknown"
        fig.suptitle(
            f"Class-based Loss Debug | Epoch {epoch} | Batch {batchIndex} | Sample {sampleIndex} | Class: {className}",
            fontsize=12,
        )
        fig.tight_layout()

        saveX0HatHistogram(
            x0Hat=generatedScribble,
            epoch=epoch,
            batchIndex=batchIndex,
            outputDir=outputDir,
        )

        filePath = outputDir / f"loss_debug_epoch_{epoch:04d}_batch_{batchIndex:05d}.png"
        fig.savefig(filePath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plt.close('all')  # Ensure all figure state is cleared
    except Exception:
        print(" - [saveClassBasedLossDebugGrid()] Warning: failed to save debug grid")


def saveX0HatHistogram(
    x0Hat: torch.Tensor,
    epoch: int,
    batchIndex: int,
    outputDir: Path,
) -> None:
    """Save the x0_hat value distribution after scaling to RGB-style 0-255 values."""
    try:
        outputDir.mkdir(parents=True, exist_ok=True)

        x0HatValues = x0Hat.detach().float().cpu().numpy()
        x0HatValues = np.clip(x0HatValues, 0.0, 1.0)
        rgbValues = np.rint(x0HatValues * 255.0).astype(np.int64).ravel()
        rgbHistogram = np.bincount(rgbValues, minlength=256)

        fig, ax = plt.subplots(figsize=(16, 5))
        xValues = np.arange(256)
        ax.bar(xValues, rgbHistogram, width=1.0, color="red", edgecolor="red", linewidth=0.2)
        ax.set_xlim(0, 255)
        ax.set_xlabel("x0_hat scaled value")
        ax.set_ylabel("Pixel count")
        ax.set_title("x0_hat Histogram")
        ax.set_xticks(np.arange(0, 256, 32))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        filePath = outputDir / f"red_channel_histo_epoch_{epoch:04d}_batch_{batchIndex:05d}.png"
        fig.savefig(filePath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        print(" - [saveX0HatHistogram()] Warning: failed to save x0_hat histogram")


######################################################################
# MAIN CLASS
######################################################################

class Diffusion:
    """
    Implements the DDPM diffusion process for image generation.

    Terminology:
        - x0: initial image (of scribble)
        - x0_hat: prediction of initial image (of scribble)
        - xt: final image (after forward diffusion)
    """

    def __init__(
        self,
        trainParams: TrainParams,
        dataParams: DataParams,
        modelParams: ModelParams,
        device: str = "cpu",
    ):
        """
        Initialize the Diffusion model.

        Parameters
        ----------
        trainParams : TrainParams
            Training parameters (Pydantic model)
        dataParams : DataParams
            Data parameters (Pydantic model)
        modelParams : ModelParams
            Model parameters (Pydantic model)
        device : str, default="cpu"
            Device to use for computation
        """
        # Step 0 - Initialize basic attributes
        self.device = torch.device(device)
        self.steps = trainParams.steps
        self.num_epochs = trainParams.numEpochs

        # Step 0.1 - Initialize model
        model = modelParams.model
        if model == configConstants.MODEL_DU_UNET_4BLOCKS:
            self.decoder = Diffusion4BlocksUNet(
                **modelParams.model_dump(),
                sampleSize=(dataParams.imageSize.height, dataParams.imageSize.width),
            )
        else:
            raise ValueError(f"Model {model} is not supported.")

        if modelParams.conditional:
            self.controlnet = self.decoder.controlnet
        else:
            self.controlnet = None

        self.decoder.to(self.device) # type: ignore

        # Log model summary (dummy forward at the configured image size, on self.device)
        summaryShape = (1, modelParams.imageChannels, dataParams.imageSize.height, dataParams.imageSize.width)
        summaryConditions = None
        if modelParams.conditional:
            summaryConditions = [
                torch.randn(summaryShape[0], 1, *summaryShape[2:], device=self.device),
                torch.randn(summaryShape[0], 1, *summaryShape[2:], device=self.device),
            ]
        summary(
            self.decoder,
            input_data=torch.randn(*summaryShape, device=self.device),
            t=torch.tensor([5], device=self.device),
            conditions=summaryConditions,
            device=self.device,
        )

        # Step 0.2 - Initialize scheduler
        varScheduler = computeVarianceSchedule(
            trainParams.steps,
            trainParams.schedulerType,
            beta1=trainParams.schedulerArgs.beta1,
            betaT=trainParams.schedulerArgs.betaT,
            s=trainParams.schedulerArgs.s,
            maxBeta=trainParams.schedulerArgs.maxBeta,
        )
        self.alphaBar = varScheduler[0].to(device)
        self.beta = varScheduler[1].to(device)

        # Step 0.3 - Configure loss function
        if trainParams.loss.lower() == configConstants.LOSS_L1.lower():
            self.baseLoss = F.l1_loss
        elif trainParams.loss.lower() == configConstants.LOSS_L2.lower():
            self.baseLoss = F.mse_loss
        else:
            raise ValueError(f"Loss {trainParams.loss} is not supported.")

        # Step 0.4 - Set loss function
        print(" - [Diffusion] Loss function setup")

        # Step 0.6 - Class-aware loss weights (configurable per run)
        classWeights = trainParams.lossClassWeights
        self.targetRegionLossWeight = classWeights.targetRegionLossWeight
        self.nonTargetRegionLossWeight = classWeights.nonTargetRegionLossWeight
        print(
            f" - [Diffusion] Class-aware loss weights: "
            f"target={self.targetRegionLossWeight}, "
            f"nonTarget={self.nonTargetRegionLossWeight}"
        )

        self.lossDebugDir = Path(trainParams.savePath) / "loss_debug_grids"
        self._savedLossDebugEpochs: set[int] = set()

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: add noise to initial image.

        Parameters
        ----------
        x0 : torch.Tensor, shape=(B, C=1, H, W)
            Initial image (of scribble)
        t : torch.Tensor, shape=(B, 1, 1, 1)
            Time step

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (eps, xt) - true noise and noisy image

        Notes
        -----
        Refer to DDPM paper (https://arxiv.org/pdf/2006.11239) - Algorithm 1
        """
        # Step 1 - Randomly sample noise
        eps = torch.randn_like(x0)

        # Step 2 - Compute mu and sigma
        alphaBar = self.alphaBar[t]
        mu = torch.sqrt(alphaBar) * x0
        sigma = torch.sqrt(1 - alphaBar)

        return eps, mu + sigma * eps

    def backward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[list[torch.Tensor]] = None,
        scribbleClass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Backward diffusion process: predict noise at time step t.

        Parameters
        ----------
        xt : torch.Tensor, shape=(B, C=1, H, W)
            Noisy image at time step t
        t : torch.Tensor
            Time step
        conditions : list[torch.Tensor], optional
            Conditioning tensors for guided diffusion
        scribbleClass : torch.Tensor, optional
            Scribble class type indices (0=background, 1=foreground)

        Returns
        -------
        torch.Tensor
            Predicted noise
        """
        t = t.squeeze()
        noise = self.decoder(xt, t, conditions, scribbleClass=scribbleClass)
        return noise

    def compute_loss(
        self,
        x0: torch.Tensor,
        conditions: list[torch.Tensor],
        scribbleClass: torch.Tensor,
        imgIndex: int = 0,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Compute diffusion loss with region-aware weighting and Gaussian dilation.

        Uses only noise loss, weighted by target region (FN∪TP for FG, FP for BG)
        with Gaussian dilation for tolerance. Does not touch gradient state: the
        caller is responsible for ``optimizer.zero_grad()`` so that gradient
        accumulation across calls works.

        Parameters
        ----------
        x0 : torch.Tensor, shape=(B, C=1, H, W)
            Initial image (of scribble)
        conditions : list[torch.Tensor]
            Conditioning tensors [gt, prediction]
        scribbleClass : torch.Tensor
            Scribble class type indices (0=background, 1=foreground)
        imgIndex : int, default=0
            Image index for logging
        epoch : int, default=0
            Current epoch number

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        # Read conditions
        gt, prediction = conditions

        # Build confusion regions from binary GT/prediction masks
        gtMask = gt > 0.5
        predictionMask = prediction > 0.5

        maskTP = gtMask & predictionMask
        maskFP = (~gtMask) & predictionMask
        maskFN = gtMask & (~predictionMask)

        # Class-aware target region (where the scribble of this class lives in the data):
        #   Foreground (class 1): GT foreground = FN ∪ TP (real FG scribbles span the
        #                         whole object, not just the missed FN region).
        #   Background (class 0): FP only (Pred=1, GT=0) — the prediction's false-positive
        #                         error. TN is deliberately excluded so the model is not
        #                         encouraged to scribble across the correct background.
        fgIdx = (scribbleClass == 1).view(-1)
        bgIdx = ~fgIdx

        targetRegion = torch.zeros_like(gt, dtype=torch.float32)
        if fgIdx.any():
            targetRegion[fgIdx] = (maskFN | maskTP)[fgIdx].float()
        if bgIdx.any():
            targetRegion[bgIdx] = maskFP[bgIdx].float()

        # Dilate target region with Gaussian for tolerance
        targetDilated = self._get_dilated_region(targetRegion, sigma=2.0)

        # Non-target region: every pixel not in the (undilated) target region.
        nonTargetRegion = 1.0 - targetRegion

        # Combined weight map used only for debug visualisation — not for the loss.
        debugWeights = (
            targetDilated * self.targetRegionLossWeight
            + nonTargetRegion * self.nonTargetRegionLossWeight
        )

        # Randomly sample time step in [1, steps]: t=0 has alphaBar=1 (no noise), so the
        # true noise would be unrecoverable and the sample would only add gradient noise.
        t = torch.randint(1, self.steps + 1, (x0.shape[0], 1, 1, 1), device=x0.device)

        # Forward diffusion (add noise)
        noise_true, xt = self.forward(x0, t)

        # Backward diffusion (predict noise)
        noise_pred = self.backward(xt, t, conditions, scribbleClass=scribbleClass)

        # Compute per-pixel noise loss
        lossImage = self.baseLoss(noise_pred, noise_true, reduction="none")

        if torch.isnan(debugWeights).any():
            print(" - [compute_loss()] Warning: NaN in debugWeights")

        # Region-normalised loss: mean loss per region is computed independently so the
        # target/non-target balance is unaffected by pixel-count disparity.
        targetLoss = (targetDilated * lossImage).sum() / targetDilated.sum().clamp(min=1e-6)
        nonTargetLoss = (nonTargetRegion * lossImage).sum() / nonTargetRegion.sum().clamp(min=1e-6)
        totalLoss = (
            self.targetRegionLossWeight * targetLoss
            + self.nonTargetRegionLossWeight * nonTargetLoss
        )

        if torch.isnan(totalLoss):
            print(" - [compute_loss()] Warning: NaN in totalLoss")

        # Randomly save one debug grid per epoch.
        if (
            epoch not in self._savedLossDebugEpochs
            and np.random.uniform(0.0, 1.0) < LOSS_DEBUG_SAVE_PROBABILITY
        ):
            alphaBar_t = self.alphaBar[t]
            x0_hat = (xt - torch.sqrt(1 - alphaBar_t) * noise_pred) / torch.sqrt(alphaBar_t)
            sampleIndex = int(torch.randint(0, x0.shape[0], (1,)).item())
            saveClassBasedLossDebugGrid(
                gt=gt,
                prediction=prediction,
                generatedScribble=x0_hat,
                originalScribble=x0,
                weightHeatmap=debugWeights,
                epoch=epoch,
                batchIndex=imgIndex,
                outputDir=self.lossDebugDir,
                sampleIndex=sampleIndex,
                scribbleClass=int(scribbleClass[sampleIndex].item()),
            )
            self._savedLossDebugEpochs.add(epoch)

        # Debug logging
        if imgIndex % 100 == 0:
            print(
                " - [compute_loss()] loss | "
                f"epoch={epoch} imgIndex={imgIndex} "
                f"fg={int(fgIdx.sum())} bg={int(bgIdx.sum())} "
                f"targetLoss={targetLoss.item():.4f} nonTargetLoss={nonTargetLoss.item():.4f} "
                f"total={totalLoss.item():.4f}"
            )

        return totalLoss

    def _get_dilated_region(self, region: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
        """
        Dilate binary region with Gaussian for tolerance zone.

        Parameters
        ----------
        region : torch.Tensor, shape=(B, 1, H, W)
            Binary region mask
        sigma : float, default=2.0
            Gaussian sigma for dilation

        Returns
        -------
        torch.Tensor
            Dilated region (values in range [0, 1])
        """
        # Separable Gaussian blur on-device, matching scipy.ndimage.gaussian_filter
        # defaults (truncate=4.0, reflect padding) but without a CPU round trip.
        radius = int(4.0 * sigma + 0.5)
        coords = torch.arange(-radius, radius + 1, device=region.device, dtype=region.dtype)
        kernel1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel1d = kernel1d / kernel1d.sum()
        kernelH = kernel1d.view(1, 1, 1, -1)
        kernelV = kernel1d.view(1, 1, -1, 1)
        padded = F.pad(region, (radius, radius, radius, radius), mode="reflect")
        dilated = F.conv2d(padded, kernelH)
        dilated = F.conv2d(dilated, kernelV)
        return dilated

    @torch.no_grad()
    def sample(
        self,
        inputSize: Optional[tuple[int, int, int, int]] = None,
        xt: Optional[torch.Tensor] = None,
        conditions: Optional[list[torch.Tensor]] = None,
        scribbleClass: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples using reverse diffusion process.

        Parameters
        ----------
        inputSize : tuple[int, int, int, int], optional
            Shape of input tensor (B, C, H, W)
        xt : torch.Tensor, optional
            Initial noisy tensor
        conditions : list[torch.Tensor], optional
            Conditioning tensors for guided generation
        scribbleClass : torch.Tensor, optional
            Scribble class type indices (0=background, 1=foreground)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (xt, x0) - final image and initial noise

        Notes
        -----
        Refer to DDPM paper (https://arxiv.org/pdf/2006.11239) - Algorithm 2
        Loop from time=steps to time=0
        """
        # Step 1 - Initialize tensors
        if inputSize is not None and xt is None:
            localXt = torch.randn(inputSize, device=self.device)
        elif xt is not None and inputSize is None:
            localXt = xt.to(self.device)
        else:
            print(" - [sample()] Invalid input")
            return torch.tensor([]), torch.tensor([])

        x0 = localXt.detach().clone()

        # Step 2 - Reverse diffusion loop
        for t in range(self.steps, 0, -1):
            # Note z is zero at the last step to not add noise at the end
            z = torch.randn_like(localXt, device=self.device) if t > 1 else 0
            tTensor = torch.tensor([t], device=self.device)

            if conditions is not None:
                noise = self.decoder(localXt, tTensor, conditions, scribbleClass=scribbleClass)
            else:
                noise = self.decoder(localXt, tTensor, scribbleClass=scribbleClass)

            alpha = 1 - self.beta[t]
            alphaBar = self.alphaBar[t]
            sigma = torch.sqrt(self.beta[t])

            localXt = (
                (1 / torch.sqrt(alpha))
                * (localXt - ((1 - alpha) / torch.sqrt(1 - alphaBar) * noise))
            )

            localXt += sigma * z
        return localXt, x0


def loadConfig(configPath: str) -> Config:
    """Load a JSON config file into the validated Pydantic Config schema."""
    with open(configPath, "r", encoding="utf-8") as handle:
        configData = json.load(handle)
    return Config(**configData)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a torch summary for the diffusion UNet")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "default.json"),
        help="Path to the JSON config file to load",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for the dummy run")
    args = parser.parse_args()

    config = loadConfig(args.config)
    trainParams = config.trainParams
    dataParams = config.dataParams
    modelParams = config.modelParams

    print(f"Loading config from {args.config}")
    print(
        f"Using image size {dataParams.imageSize.width}x{dataParams.imageSize.height} "
        f"and model settings from the config"
    )

    torch.manual_seed(trainParams.seed)
    model = Diffusion4BlocksUNet(
        blockOutChannels=tuple(modelParams.blockOutChannels),
        layersPerBlock=modelParams.layersPerBlock,
        normNumGroups=modelParams.normNumGroups,
        conditional=modelParams.conditional,
    )
    model.eval()

    dummy_input = torch.randn(1, 1, dataParams.imageSize.height, dataParams.imageSize.width)
    summary_conditions = None
    if modelParams.conditional:
        summary_conditions = [
            torch.randn(1, 1, dataParams.imageSize.height, dataParams.imageSize.width),
            torch.randn(1, 1, dataParams.imageSize.height, dataParams.imageSize.width),
        ]

    summary(
        model,
        input_data=dummy_input,
        t=torch.tensor([5]),
        conditions=summary_conditions,
        device=args.device,
    )
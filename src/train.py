######################################################################
# TRAINING SCRIPT - Morphologically Dependent Interactive Scribble Segmentation
######################################################################
"""
Main training script for diffusion-based scribble segmentation model.

This script handles:
    - Data loading and preprocessing
    - Model initialization and training
    - Validation and evaluation
    - Logging with Weights & Biases
    - Checkpoint saving and model persistence
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import sys
import json
import time
import tqdm
import wandb
import datetime
import argparse
import traceback
from pathlib import Path
from typing import Optional

# Pip/Third-party installs
import numpy as np
import torch
import torchvision
from PIL import Image as PILImage, ImageDraw
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Application/Local module imports
import src.configConstants as configConstants
import src.utils.utils as utils
import src.data.dataset as dataset
from src.models.diffusion import Diffusion
from src.configModels import Config
from src.data.fidMetric import FID

######################################################################
# CONSTANTS
######################################################################

BACKGROUND_SCRIBBLE_CLASS = configConstants.SCRIBBLE_CLASS_BACKGROUND_VALUE
FOREGROUND_SCRIBBLE_CLASS = configConstants.SCRIBBLE_CLASS_FOREGROUND_VALUE

######################################################################
# HELPERS
######################################################################

def _annotateImageTensor(imgTensor: torch.Tensor, text: str) -> torch.Tensor:
    """
    Overlay a short text label at the top-left of a (1, H, W) image tensor in [0, 1].
    Returns a new (1, H, W) tensor in [0, 1] with the text burned in.
    """
    arr = (imgTensor[0].detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    pil = PILImage.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(pil)
    # Black background rectangle so white text is always legible
    tw = len(text) * 6 + 4
    draw.rectangle([0, 0, tw, 12], fill=0)
    draw.text((2, 2), text, fill=255)
    out = torch.from_numpy(np.array(pil)).float() / 255.0
    return out.unsqueeze(0)  # (1, H, W)

######################################################################
# TRAINING FUNCTIONS
######################################################################

def trainEpoch(
    diffusion: Diffusion,
    model: torch.nn.Module,
    accelerator: Accelerator,
    trainDataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lrScheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    configInst: Config,
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    diffusion : Diffusion
        Diffusion process wrapper (loss / sampling)
    model : torch.nn.Module
        The model as returned by ``accelerator.prepare`` (may be a DDP wrapper)
    accelerator : Accelerator
        Accelerate accelerator for distributed training
    trainDataloader : DataLoader
        Training dataloader
    optimizer : torch.optim.Optimizer
        Optimizer
    lrScheduler : torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    epoch : int
        Epoch number
    configInst : Config
        Training configuration

    Returns
    -------
    float
        Average loss over the non-NaN batches of the epoch (NaN if there were none)

    Notes
    -----
    Errors are NOT swallowed here: a failing batch aborts the run instead of
    logging a bogus epoch loss and carrying on.
    """
    # Step 0 - Get training arguments
    trainParams = configInst.trainParams
    batchSize = trainParams.batchSize

    # Step 1 - Define variables
    totalLoss = 0.0
    numLossBatches = 0
    doScribbleGauss = trainParams.scribbleGauss
    clipNorm = trainParams.clipNorm

    # Step 2 - Loop over dataloader
    with tqdm.tqdm(
        total=len(trainDataloader) * batchSize, desc=" - [Training] "
    ) as pbar:
        for i, batch in enumerate(trainDataloader):
            # collateFn returns {} when every sample of the batch was filtered out
            if not batch:
                pbar.update(batchSize)
                continue

            with accelerator.accumulate(model):
                # Step 2.1 - Get scribble
                if not doScribbleGauss:
                    scribble = batch[configConstants.KEY_SCRIBBLE]
                else:
                    scribble = batch[configConstants.KEY_SCRIBBLE_GAUSS]

                # Step 2.2 - Compute loss
                gt = batch[configConstants.KEY_GT]
                prediction = batch[configConstants.KEY_PREDICTION]
                scribbleClass = batch.get(configConstants.KEY_SCRIBBLE_CLASS, None)
                loss = diffusion.compute_loss(
                    scribble,
                    [gt, prediction],
                    scribbleClass=scribbleClass,
                    imgIndex=i,
                    epoch=epoch,
                )

                # Step 2.3 - Handle NaN loss or perform optimization
                if torch.isnan(loss).any():
                    # Skip this micro-batch entirely (no backward, gradients untouched)
                    scribbleSum = scribble.sum(axis=(1, 2, 3))
                    print(f"   --- [trainEpoch()] NaN loss detected. Scribble sum: {scribbleSum.detach().cpu().numpy()}")
                else:
                    totalLoss += loss.detach().cpu().item()
                    numLossBatches += 1
                    avgLoss = totalLoss / numLossBatches

                    # accelerator.backward handles loss scaling / accumulation bookkeeping
                    accelerator.backward(loss)

                    # Under accumulation, gradients sync only on the last micro-batch;
                    # clipping before that point would act on a partial gradient.
                    if accelerator.sync_gradients and clipNorm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=clipNorm)

                    optimizer.step()
                    lrScheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    pbar.set_description(f" - [Training] Avg Loss: {avgLoss:.4f}")

            pbar.update(len(scribble))

    # Step 3 - Compute epoch average loss
    lossAvgEpoch = totalLoss / numLossBatches if numLossBatches > 0 else float("nan")

    return lossAvgEpoch


######################################################################
# EVALUATION FUNCTIONS
######################################################################

def evaluateTrain(
    diffusion: Diffusion,
    valDataset: torch.utils.data.Dataset,
    valDatasetValIdxs: torch.Tensor,
    pathFolderModel: Path,
    configInst: Config,
    scribbleClassOverride: Optional[int] = None,
) -> tuple[list, list]:
    """
    Evaluate on training set.

    Parameters
    ----------
    diffusion : Diffusion
        Diffusion model
    valDataset : torch.utils.data.Dataset
        Validation dataset
    valDatasetValIdxs : torch.Tensor
        Validation indices
    pathFolderModel : Path
        Path to model folder
    configInst : Config
        Training configuration
    scribbleClassOverride : int, optional
        Override scribble class for generation (0=background, 1=foreground)
        If None, uses the class from the dataset samples

    Returns
    -------
    tuple[list, list]
        (images, imagesRGB)
    """
    # Step 0 - Initialize
    trainParams, modelParams = configInst.trainParams, configInst.modelParams
    images: list = []
    imagesRGB: list = []
    doScribbleGauss = trainParams.scribbleGauss

    # Step 1 - Loop over validation batches
    try:
        print("\n  -- [evaluateTrain()] Sampling images")
        for iterId, batchIdxs in enumerate(valDatasetValIdxs):
            tSampleStart = time.time()
            with torch.no_grad():
                # Step 1.0 - Collate batch
                gt, prediction, scribble, scribbleClass = [], [], [], []
                for datasetIdx in batchIdxs:
                    datasetIdxItem = valDataset[datasetIdx]
                    gt.append(datasetIdxItem[configConstants.KEY_GT])
                    prediction.append(datasetIdxItem[configConstants.KEY_PREDICTION])
                    if not doScribbleGauss:
                        scribble.append(datasetIdxItem[configConstants.KEY_SCRIBBLE])
                    else:
                        scribble.append(datasetIdxItem[configConstants.KEY_SCRIBBLE_GAUSS])
                    scribbleClass.append(datasetIdxItem[configConstants.KEY_SCRIBBLE_CLASS])

                gt = torch.stack(gt)
                prediction = torch.stack(prediction)
                scribble = torch.stack(scribble)
                scribbleClass = torch.stack(scribbleClass)
                
                # Step 1.0.1 - Apply scribbleClass override if provided
                if scribbleClassOverride is not None:
                    scribbleClass = torch.full_like(scribbleClass, scribbleClassOverride)

                # Step 1.1 - Forward diffusion
                _, scribbleNoisy = diffusion.forward(x0=scribble.to(diffusion.device), t=torch.tensor(diffusion.steps, device=diffusion.device))

                # Step 1.2 - Sample
                tSampleOnlyStart = time.time()
                if modelParams.conditional:
                    gt = gt.to(diffusion.device)
                    prediction = prediction.to(diffusion.device)
                    scribbleClass = scribbleClass.to(diffusion.device)
                    sampled, noise = diffusion.sample(
                        xt=scribbleNoisy, 
                        conditions=[gt, prediction], 
                        scribbleClass=scribbleClass
                    )
                else:
                    sampled, noise = diffusion.sample(xt=scribbleNoisy)
                tSampleOnlyEnd = time.time()

                noiseClone = utils.minMaxNormalize(noise)
                # Clamp instead of min-max normalising: x0 targets live in [0, 1]
                # (background=0, scribble=1). Min-max anchors the per-sample negative
                # min to 0, lifting the near-zero background into a visible grey haze.
                sampledClone = sampled.clamp(0.0, 1.0)
                utils.plotForInference(
                    sampled=sampledClone,
                    noise=noiseClone,
                    scribble=scribble,
                    pathFolderModel=pathFolderModel,
                    mode=configConstants.KEY_TRAIN,
                    meta=iterId,
                )

                gtAndPredRegion = gt * 0.5 + prediction * 0.5
                gtAndPredRegion = gtAndPredRegion.detach().clone()
                gtBoundary = torch.nn.functional.max_pool2d(gt, 3, 1, 1) - gt
                gtAndPredRegion[gtBoundary > 0.5] = 1.0

                # Annotate sampled tiles with SC: class label
                scribbleClassCpu = scribbleClass.cpu() if scribbleClass is not None else None
                sampledCpu = sampledClone.detach().cpu()
                annotatedSampledCpu = []
                for bIndex in range(sampledCpu.shape[0]):
                    tile = sampledCpu[bIndex]  # (1, H, W)
                    if scribbleClassCpu is not None and bIndex < len(scribbleClassCpu):
                        classVal = scribbleClassCpu[bIndex].item()
                        scLabel = "B" if classVal == BACKGROUND_SCRIBBLE_CLASS else "F"
                        tile = _annotateImageTensor(tile, f"SC:{scLabel}")
                    annotatedSampledCpu.append(tile)
                annotatedSampledCpu = torch.stack(annotatedSampledCpu)

                wandbImages = torch.concat(
                    [annotatedSampledCpu, scribble.detach().cpu(), gtAndPredRegion.detach().cpu()], dim=0
                ).clamp(0.0, 1.0)

                grid = torchvision.utils.make_grid(wandbImages, nrow=8, padding=2, normalize=False)
                images.append(wandb.Image(grid))

                # RGB visualization
                wandbImagesRGB = torch.cat(
                    [
                        annotatedSampledCpu,
                        gtAndPredRegion.detach().cpu(),
                        scribble.detach().cpu(),
                    ],
                    dim=1,
                )
                
                # Make the ground-truth boundary precisely yellow (R=1, G=1, B=0)
                gtBoundaryMask = (gtBoundary.detach().cpu() > 0.5).squeeze(1)
                wandbImagesRGB[:, 0, :, :][gtBoundaryMask] = 1.0  # Red
                wandbImagesRGB[:, 1, :, :][gtBoundaryMask] = 1.0  # Green
                wandbImagesRGB[:, 2, :, :][gtBoundaryMask] = 0.0  # Blue
                wandbImagesRGB = wandbImagesRGB.clamp(0.0, 1.0)

                gridRGB = torchvision.utils.make_grid(wandbImagesRGB, nrow=8, padding=2, normalize=False)
                imagesRGB.append(wandb.Image(gridRGB))

            print(
                f"   --- [evaluateTrain()][{iterId}/{len(valDatasetValIdxs)}] Time: {time.time()-tSampleStart:.2f}s (sample={tSampleOnlyEnd-tSampleOnlyStart:.2f}s, gauss={doScribbleGauss})"
            )

    except Exception as e:
        print(f" - [evaluateTrain()] Error: {e}")
        traceback.print_exc()

    return images, imagesRGB


def logHazeStats(
    sampled: torch.Tensor,
    sampledNorm: torch.Tensor,
    gt: torch.Tensor,
    prediction: torch.Tensor,
    scribbleClass: Optional[torch.Tensor],
    pathFolderModel: Path,
    epoch: int,
    processedEvalImages: int,
) -> None:
    """
    Append per-sample, region-separated statistics of the sampled output to a JSONL file.

    Purpose
    -------
    Diagnose the background "haze": separates the pure-background (true-negative)
    region from the artifact region (GT|prediction) and logs statistics of both
    the RAW sampled tensor and the min-max-normalised tensor. If the background is
    a tight band just above the per-sample global min in the RAW tensor but shows a
    large mean in the NORMALISED tensor, then minMaxNormalize is amplifying tiny
    residuals into the visible haze.

    Parameters
    ----------
    sampled : torch.Tensor, shape=(B, 1, H, W)
        Raw diffusion output before normalisation.
    sampledNorm : torch.Tensor, shape=(B, 1, H, W)
        The display-normalised tensor actually rendered (clamp(0,1)).
    gt : torch.Tensor, shape=(B, 1, H, W)
        Ground-truth foreground mask.
    prediction : torch.Tensor, shape=(B, 1, H, W)
        Predicted foreground mask.
    scribbleClass : torch.Tensor, optional
        Per-sample scribble class (0=background, 1=foreground).
    pathFolderModel : Path
        Model output folder; stats are written under <folder>/haze_debug/.
    epoch : int
        Current epoch (for grouping rows across an evaluation).
    processedEvalImages : int
        Running count of evaluated images (row identifier).
    """
    try:
        debugDir = Path(pathFolderModel) / "haze_debug"
        debugDir.mkdir(parents=True, exist_ok=True)
        statsPath = debugDir / "haze_stats.jsonl"

        # Artifact region = GT|pred foreground; background (TN) = its complement
        artifactMask = (gt + prediction).clamp(max=1.0) > 0.5
        backgroundMask = ~artifactMask

        rawCpu = sampled.detach().cpu()
        normCpu = sampledNorm.detach().cpu()
        bgCpu = backgroundMask.detach().cpu()
        artCpu = artifactMask.detach().cpu()
        scCpu = scribbleClass.detach().cpu() if scribbleClass is not None else None

        rows = []
        for b in range(rawCpu.shape[0]):
            raw = rawCpu[b]            # (1, H, W)
            norm = normCpu[b]
            bgM = bgCpu[b]
            artM = artCpu[b]

            rawBg = raw[bgM]           # pure-background pixels (raw)
            rawArt = raw[artM]         # artifact-region pixels (raw)
            normBg = norm[bgM]         # pure-background pixels (normalised) → the haze

            def stat(t: torch.Tensor) -> dict:
                if t.numel() == 0:
                    return {"n": 0}
                return {
                    "n": int(t.numel()),
                    "mean": float(t.mean()),
                    "std": float(t.std()),
                    "min": float(t.min()),
                    "max": float(t.max()),
                }

            rows.append({
                "epoch": int(epoch),
                "processedEvalImages": int(processedEvalImages),
                "batchIndex": int(b),
                "scribbleClass": int(scCpu[b]) if scCpu is not None and b < len(scCpu) else None,
                # Per-sample global range of the RAW output (the min-max divisor anchors)
                "rawGlobalMin": float(raw.min()),
                "rawGlobalMax": float(raw.max()),
                # RAW region stats
                "rawBackground": stat(rawBg),
                "rawArtifact": stat(rawArt),
                # NORMALISED background stats — quantifies the visible haze
                "normBackground": stat(normBg),
            })

        with open(statsPath, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f" - [logHazeStats()] Error: {e}")
        traceback.print_exc()


def evaluate(
    diffusion: Diffusion,
    valDataloader: DataLoader,
    pathFolderModel: Path,
    configInst: Config,
    epoch: int = 0,
    scribbleClassOverride: Optional[int] = None,
    fidMetric: Optional[FID] = None,
) -> tuple[list, list, list, float, float, float, list, float, float, list, list, int, int, int]:
    """
    Evaluate on validation set.

    Parameters
    ----------
    diffusion : Diffusion
        Diffusion model
    valDataloader : DataLoader
        Validation dataloader
    pathFolderModel : Path
        Path to model folder
    configInst : Config
        Training configuration
    epoch : int, default=0
        Current training epoch, used in saved image filenames
    scribbleClassOverride : int, optional
        Override scribble class for generation (0=background, 1=foreground)
        If None, uses the class from the batch samples
    fidMetric : FID, optional
        FID metric instance used to compute per-image FID scores.

    Returns
    -------
    tuple
        (images, realImages, sampledImages, avgPrecision, avgFid, imagesRGB,
         avgPrecisionReal, avgPrecisionSynthetic, imagesValReal, imagesValSynthetic,
         precisionCount, precisionCountReal, precisionCountSynthetic)
    """
    # Step 0 - Initialize
    trainParams, modelParams, dataParams = configInst.trainParams, configInst.modelParams, configInst.dataParams
    evalSaveEveryNImages = trainParams.evalSaveEveryNImages
    evalSaveMode = trainParams.evalSaveMode
    images: list = []
    imagesRGB: list = []
    imagesValReal: list = []
    imagesValSynthetic: list = []
    realImages = []
    sampledImages = []

    # Step 1 - Loop over validation batches
    print("\n  -- [evaluate()] Sampling images")
    try:
        precisionSum = 0.0
        precisionSoftSum = 0.0
        precisionSoftCount = 0
        precisionCount = 0
        precisionSumReal = 0.0
        precisionCountReal = 0
        precisionSumSynthetic = 0.0
        precisionCountSynthetic = 0
        fidSum = 0.0
        fidCount = 0
        processedEvalImages = 0

        for i, batch in enumerate(valDataloader):
            if not batch:  # every sample of the batch was filtered by collateFn
                continue
            tSampleStart = time.time()
            with torch.no_grad():
                gt = batch[configConstants.KEY_GT].to(diffusion.device)
                prediction = batch[configConstants.KEY_PREDICTION].to(diffusion.device)
                scribbleClass = batch.get(configConstants.KEY_SCRIBBLE_CLASS, None)
                if scribbleClass is not None:
                    scribbleClass = scribbleClass.to(diffusion.device)
                
                # Step 1.1 - Apply scribbleClass override if provided
                if scribbleClassOverride is not None:
                    batchSize = len(gt)
                    scribbleClass = torch.full(
                        (batchSize,), scribbleClassOverride, dtype=torch.long, device=diffusion.device
                    )

                inputSize = (
                    len(gt),
                    modelParams.imageChannels,
                    dataParams.imageSize.height,
                    dataParams.imageSize.width,
                )

                # Step 1.2 - Sample
                tSampleOnlyStart = time.time()
                if modelParams.conditional:
                    sampled, noise = diffusion.sample(
                        inputSize=inputSize, 
                        conditions=[gt, prediction], 
                        scribbleClass=scribbleClass
                    )
                else:
                    sampled, noise = diffusion.sample(inputSize=inputSize)
                tSampleOnlyEnd = time.time()

                noiseClone = utils.minMaxNormalize(noise)
                # Clamp instead of min-max normalising: x0 targets live in [0, 1]
                # (background=0, scribble=1). Min-max anchors the per-sample negative
                # min to 0, lifting the near-zero background into a visible grey haze.
                sampledClone = sampled.clamp(0.0, 1.0)

                # Log region-separated stats (raw vs normalised) to diagnose the background haze.
                logHazeStats(
                    sampled=sampled,
                    sampledNorm=sampledClone,
                    gt=gt,
                    prediction=prediction,
                    scribbleClass=scribbleClass,
                    pathFolderModel=pathFolderModel,
                    epoch=epoch,
                    processedEvalImages=processedEvalImages,
                )

                # Save debug plots using either deterministic intervals or stochastic sampling.
                previousProcessedEvalImages = processedEvalImages
                processedEvalImages += len(gt)
                saveInferencePlot = False
                if evalSaveEveryNImages > 0:
                    if evalSaveMode == "deterministic":
                        saveInferencePlot = (
                            (processedEvalImages // evalSaveEveryNImages)
                            > (previousProcessedEvalImages // evalSaveEveryNImages)
                        )
                    else:
                        # Expected one save every N images on average.
                        saveProb = min(1.0, len(gt) / float(evalSaveEveryNImages))
                        saveInferencePlot = np.random.random() < saveProb

                # Compute error region (blend of gt and prediction) with GT boundary overlay
                gtAndPredRegion = gt * 0.5 + prediction * 0.5
                gtBoundary = torch.nn.functional.max_pool2d(gt, 3, 1, 1) - gt
                gtAndPredRegion[gtBoundary > 0.5] = 1.0
                if saveInferencePlot:
                    utils.plotForInference(
                        sampled=sampledClone,
                        noise=noiseClone,
                        pathFolderModel=pathFolderModel,
                        mode=configConstants.KEY_VAL,
                        meta=processedEvalImages,
                    )

                scribbled = batch[configConstants.KEY_SCRIBBLE_GAUSS].to(diffusion.device)
                realImages.append(scribbled)
                sampledImages.append(sampledClone)

                # ------------------------------------------------------------------
                # Step 1.4 - Per-image precision (moved before grid creation so that
                #            the scores can be burned into the wandb image tiles).
                # Compute accuracy (precision) and F1 score
                # Scribbles are sparse line structures placed on either:
                #   - FP region (background scribble, class=0): pred=1 & gt=0 → "remove" correction
                #   - FN region (foreground scribble, class=1): pred=0 & gt=1 → "add" correction
                # Primary metric: precision = fraction of generated scribble pixels that land on
                # the correct target error region. Recall measures coverage of that region.
                # Use the min-max normalized output (sampledClone) so that thresholding at 0.5 is
                # meaningful; raw diffusion output has arbitrary range centered near 0.
                sampledCpu = sampledClone.detach().cpu()
                scribbleClassCpu = scribbleClass.cpu() if scribbleClass is not None else None
                batchMeta = batch.get(configConstants.KEY_META, None)
                batchPrecisions: list[float | None] = []  # one entry per image in this batch
                batchSoftPrecisions: list[float | None] = []  # one entry per image in this batch
                for bIndex in range(prediction.shape[0]):
                    predI = batch[configConstants.KEY_PREDICTION][bIndex]
                    gtI = batch[configConstants.KEY_GT][bIndex]
                    sampledI = sampledCpu[bIndex]
                    # Degenerate case (prediction all background): the whole GT is FN, so a
                    # foreground scribble on the object scores high and a background scribble
                    # has no FP target (precision 0). No special-casing needed.

                    # Step 1 - Derive the two possible target error regions
                    bgRegion = (predI > 0.5) & (gtI < 0.5)   # false positives → background scribble target
                    fgRegion = (predI < 0.5) & (gtI > 0.5)   # false negatives → foreground scribble target

                    # 4 variables comprising FN, TN, FP, TP of the pred versus gt
                    fnRegion = (predI < 0.5) & (gtI > 0.5)   # false negative
                    tnRegion = (predI < 0.5) & (gtI < 0.5)   # true negative
                    fpRegion = (predI > 0.5) & (gtI < 0.5)   # false positive
                    tpRegion = (predI > 0.5) & (gtI > 0.5)   # true positive

                    softFgPrecisionMask = fnRegion | tnRegion  # union of pred-tn and pred-fn
                    softBgPrecisionMask = fpRegion | tpRegion  # union of pred-tp and pred-fp

                    # Step 2 - Select correct target region using scribble class label
                    # class=0 → background scribble → target is FP region
                    # class=1 → foreground scribble → target is FN region
                    if scribbleClassCpu is not None:
                        classVal = scribbleClassCpu[bIndex].item()
                        targetMask = bgRegion if classVal == BACKGROUND_SCRIBBLE_CLASS else fgRegion
                        targetSoftMask = softBgPrecisionMask if classVal == BACKGROUND_SCRIBBLE_CLASS else softFgPrecisionMask
                    else:
                        # Fallback: use whichever error region is larger
                        targetMask = bgRegion if bgRegion.sum() >= fgRegion.sum() else fgRegion
                        targetSoftMask = softBgPrecisionMask if bgRegion.sum() >= fgRegion.sum() else softFgPrecisionMask

                    # Step 3 - Threshold sampled output to binary scribble pixels
                    sampledPixels = sampledI > 0.5

                    # Step 4 - Build full confusion matrix of scribble vs target mask
                    tpCount = (sampledPixels &  targetMask).sum().item()
                    fpCount = (sampledPixels & ~targetMask).sum().item()

                    if (tpCount + fpCount) == 0:
                        # No scribble pixels generated: keep both per-image lists aligned
                        batchPrecisions.append(None)
                        batchSoftPrecisions.append(None)
                        continue

                    # Step 5 - Precision: fraction of generated scribble pixels correctly placed
                    # on the target error region. This is the only metric not dominated by the
                    # large number of uncovered target pixels (fnCount).
                    precision = tpCount / (tpCount + fpCount)
                    softPrecision = (sampledPixels & targetSoftMask).sum().item() / sampledPixels.sum().item() if sampledPixels.sum().item() > 0 else None

                    # MT: hard versus soft precision
                    # soft precision - fg:
                    #   - masking_area = is union of pred-tn U pred-fn, ignore output scribble outside of masking
                    #   - masked_scribble = output-scribble * masking_area
                    #   - 
                    # soft precision - bg: 
                    
                    batchPrecisions.append(precision)
                    batchSoftPrecisions.append(softPrecision)

                    precisionSum += precision
                    precisionCount += 1
                    if softPrecision is not None:
                        precisionSoftSum += softPrecision
                        precisionSoftCount += 1

                    source = configConstants.DATA_SOURCE_REAL
                    if isinstance(batchMeta, list) and bIndex < len(batchMeta):
                        source = batchMeta[bIndex].get(
                            configConstants.KEY_DATA_SOURCE,
                            configConstants.DATA_SOURCE_REAL,
                        )

                    if source == configConstants.DATA_SOURCE_REAL:
                        precisionSumReal += precision
                        precisionCountReal += 1
                    else:
                        precisionSumSynthetic += precision
                        precisionCountSynthetic += 1

                # ------------------------------------------------------------------
                # Step 1.5 - Per-image FID
                batchFids: list[float | None] = []
                for bIndex in range(sampledCpu.shape[0]):
                    imgFid: float | None = None
                    if fidMetric is not None:
                        try:
                            imgFid = float(fidMetric.computeForImage(sampledCpu[bIndex]))  # (1, H, W)
                            fidSum += imgFid
                            fidCount += 1
                        except Exception as e:
                            print(f" - [evaluate()] Error computing FID for image {bIndex}: {e}")
                            traceback.print_exc()
                            imgFid = None
                    batchFids.append(imgFid)

                # ------------------------------------------------------------------
                # Step 1.6 - Annotate each sampled image tile with SC:, F: and P: scores
                annotatedSampledCpu = []
                for bIndex in range(sampledCpu.shape[0]):
                    tile = sampledCpu[bIndex]  # (1, H, W)
                    prec = batchPrecisions[bIndex] if bIndex < len(batchPrecisions) else None
                    softPrec = batchSoftPrecisions[bIndex] if bIndex < len(batchSoftPrecisions) else None
                    imgFid = batchFids[bIndex] if bIndex < len(batchFids) else None
                    labelParts = []
                    if scribbleClassCpu is not None and bIndex < len(scribbleClassCpu):
                        classVal = scribbleClassCpu[bIndex].item()
                        scLabel = "B" if classVal == BACKGROUND_SCRIBBLE_CLASS else "F"
                        labelParts.append(f"SC:{scLabel}")
                    if imgFid is not None:
                        labelParts.append(f"F:{imgFid:.1f}")
                    if prec is not None:
                        labelParts.append(f"P:{prec:.2f}")
                    if softPrec is not None:
                        labelParts.append(f"SP:{softPrec:.2f}")
                    if labelParts:
                        tile = _annotateImageTensor(tile, " - ".join(labelParts))
                    annotatedSampledCpu.append(tile)
                annotatedSampledCpu = torch.stack(annotatedSampledCpu)  # (B, 1, H, W)

                # ------------------------------------------------------------------
                # Step 1.7 - Build wandb image grids (use annotated tiles for sampled)
                wandbImages = torch.concat(
                    [annotatedSampledCpu, scribbled.detach().cpu(), gtAndPredRegion.detach().cpu()], dim=0
                ).clamp(0.0, 1.0)

                # RGB visualization
                wandbImagesRGB = torch.cat(
                    [
                        annotatedSampledCpu,
                        gtAndPredRegion.detach().cpu(),
                        scribbled.detach().cpu(),
                    ],
                    dim=1,
                )
                
                # Make the ground-truth boundary precisely yellow (R=1, G=1, B=0)
                gtBoundaryMask = (gtBoundary.detach().cpu() > 0.5).squeeze(1)
                wandbImagesRGB[:, 0, :, :][gtBoundaryMask] = 1.0  # Red
                wandbImagesRGB[:, 1, :, :][gtBoundaryMask] = 1.0  # Green
                wandbImagesRGB[:, 2, :, :][gtBoundaryMask] = 0.0  # Blue
                wandbImagesRGB = wandbImagesRGB.clamp(0.0, 1.0)

                grid = torchvision.utils.make_grid(wandbImages, nrow=8, padding=2, normalize=False)
                images.append(wandb.Image(grid))

                sourceValues = []
                if isinstance(batchMeta, list):
                    sourceValues = [
                        sampleMeta.get(
                            configConstants.KEY_DATA_SOURCE,
                            configConstants.DATA_SOURCE_REAL,
                        )
                        for sampleMeta in batchMeta
                    ]

                if len(sourceValues) > 0 and all(
                    source == configConstants.DATA_SOURCE_REAL for source in sourceValues
                ):
                    imagesValReal.append(wandb.Image(grid))
                elif len(sourceValues) > 0:
                    imagesValSynthetic.append(wandb.Image(grid))

                gridRGB = torchvision.utils.make_grid(wandbImagesRGB, nrow=8, padding=2, normalize=False)
                imagesRGB.append(wandb.Image(gridRGB))

            print(
                f"   --- [evaluate()][{i}/{len(valDataloader)}] Time: {time.time()-tSampleStart:.2f}s "
                f"(sample={tSampleOnlyEnd-tSampleOnlyStart:.2f}s)"
            )
            # Keep the fast-path cap (`logNImages`) but avoid a misleading synthetic score
            # by trying to include at least one valid precision sample from both sources.
            if (
                i + 1 >= trainParams.logNImages
                and precisionCountReal > 0
                and precisionCountSynthetic > 0
            ):
                break

    except Exception as e:
        print(f" - [evaluate()] Error: {e}")
        traceback.print_exc()
        raise e

    avgPrecision = precisionSum / precisionCount if precisionCount > 0 else float("nan")
    avgPrecisionSoft = precisionSoftSum / precisionSoftCount if precisionSoftCount > 0 else float("nan")
    avgPrecisionReal = precisionSumReal / precisionCountReal if precisionCountReal > 0 else float("nan")
    avgPrecisionSynthetic = (
        precisionSumSynthetic / precisionCountSynthetic if precisionCountSynthetic > 0 else float("nan")
    )
    avgFid = fidSum / fidCount if fidCount > 0 else 0.0

    print(
        " - [evaluate()] Precision counts: "
        f"all={precisionCount}, real={precisionCountReal}, synthetic={precisionCountSynthetic}"
    )
    if precisionCountSynthetic == 0:
        print(
            " - [evaluate()] Warning: no valid synthetic precision samples were evaluated; "
            "Precision/Synthetic is NaN."
        )

    return (
        images,
        realImages,
        sampledImages,
        avgPrecision,
        avgPrecisionSoft,
        avgFid,
        imagesRGB,
        avgPrecisionReal,
        avgPrecisionSynthetic,
        imagesValReal,
        imagesValSynthetic,
        precisionCount,
        precisionCountReal,
        precisionCountSynthetic,
    )


######################################################################
# MAIN TRAINING LOOP
######################################################################

def main(configInst: Config):
    """
    Main training function.

    Parameters
    ----------
    configInst : Config
        Validated Pydantic Config instance
    """
    try:
        validateConfiguredRealDataPaths(configInst)

        # ----- Step 0 - Initialize
        trainParams, modelParams, dataParams = configInst.trainParams, configInst.modelParams, configInst.dataParams
        seedValue = trainParams.seed
        logWandb = trainParams.logWandb
        torch.manual_seed(seedValue)
        tScriptStart = time.time()

        print("\n ------------------------------------ [main()] Initializing")
        print(f" - [main()] Seed value: {seedValue}")
        print(f" - [main()] Log to wandb: {logWandb}")

        # ----- Step 1 - Initialize wandb
        if logWandb:
            wandbRunName = trainParams.wandbName
            if wandbRunName is None and configInst.configPath is not None:
                wandbRunName = Path(configInst.configPath).stem
            if wandbRunName is not None:
                wandbRunName = wandbRunName + "__" + str(datetime.datetime.now())
            wandbNotes = trainParams.wandbNotes
            wandbProject = trainParams.wandbProject
            wandb.init(
                project=wandbProject,
                name=wandbRunName,
                notes=wandbNotes,
                entity="interactiveScribbles",
                config=configInst.model_dump(),
            )
        else:
            print("\n - [main()] Not logging to wandb")

        # ----- Step 2 - Initialize data
        print("\n ------------------------------------ [main()] Initializing data")
        # Tensors are (C, H, W); CenterCrop / ScribbleDataset expect (height, width)
        imageSize = (dataParams.imageSize.height, dataParams.imageSize.width)
        # Flip probabilities are 0.5 - the dataset samples each transform's parameters once
        # per sample and applies the identical op to gt / prediction / scribble.
        transformsList = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop(imageSize),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(degrees=360),
            ]
        )

        # Step 2.1 - Disjoint train / validation split of the real images
        realDataPaths = [Path(p) for p in dataParams.pathsToRealData]
        allRealImagePaths = dataset.collectRealImagePaths(realDataPaths)
        trainRealImagePaths, valRealImagePaths = dataset.splitRealImagePaths(
            allRealImagePaths, dataParams.valFraction, seedValue
        )
        if dataParams.valFraction <= 0:
            print(" - [main()] WARNING: valFraction=0 -> validation runs on the training images")
        print(
            f" - [main()] Real images: {len(allRealImagePaths)} total, "
            f"{len(trainRealImagePaths)} train, {len(valRealImagePaths)} val "
            f"(valFraction={dataParams.valFraction})"
        )

        trainDataset = dataset.ScribbleDataset(
            imageSize=imageSize,
            realProbability=dataParams.realProbability,
            pathsToRealData=realDataPaths,
            realImagePaths=trainRealImagePaths,
            transforms=transformsList,
            valMode=False,
            shape=dataParams.shape,
            syntheticPerlinGeneratedProbability=dataParams.syntheticPerlinGeneratedProbability,
            computeScribbleGauss=trainParams.scribbleGauss,
        )

        valTransformsList = torchvision.transforms.Compose(
            [torchvision.transforms.CenterCrop(imageSize)]
        )
        valDataset = dataset.ScribbleDataset(
            imageSize=imageSize,
            realProbability=dataParams.realProbability,
            pathsToRealData=realDataPaths,
            realImagePaths=valRealImagePaths,
            transforms=valTransformsList,
            valMode=True,
            shape=dataParams.shape,
            syntheticPerlinGeneratedProbability=dataParams.syntheticPerlinGeneratedProbability,
            computeScribbleGauss=True,
        )

        # Step 2.2 - Fixed sample indices for evaluateTrain (whole batches only)
        numEvalTrainImages = min(len(valDataset), trainParams.logNImages * trainParams.batchSize)
        numEvalTrainImages -= numEvalTrainImages % trainParams.batchSize
        if numEvalTrainImages == 0:
            raise ValueError(
                f"Validation set ({len(valDataset)} samples) is smaller than one batch "
                f"({trainParams.batchSize}); lower batchSize or add data"
            )
        valDatasetValIdxs = (
            torch.randperm(len(valDataset))[:numEvalTrainImages]
            .reshape(-1, trainParams.batchSize)
        )

        doScribbleGauss = trainParams.scribbleGauss
        numWorkers = trainParams.numWorkers

        print(f" - [main()] do_scribble_gauss: {doScribbleGauss}")
        print(f" - [main()] Real Sample Probability: {dataParams.realProbability}")
        print(f" - [main()] Number of train samples: {len(trainDataset)}")
        print(f" - [main()] Number of validation samples: {len(valDataset)}")
        print(f" - [main()] Image size: {imageSize}")
        print(f" - [main()] num_workers: {numWorkers}")
        print(f" - [main()] batch_size: {trainParams.batchSize}")
        print(f" - [main()] val_shuffle: {trainParams.valShuffle}")

        # Keep workers alive across epochs and prefetch more batches so the
        # CPU-heavy pipeline (EDT / synthetic generation) stays ahead of the GPU.
        # persistent_workers/prefetch_factor are only valid when num_workers > 0.
        trainLoaderKwargs = {}
        if numWorkers > 0:
            trainLoaderKwargs["persistent_workers"] = True
            trainLoaderKwargs["prefetch_factor"] = 4
        trainDataloader = DataLoader(
            trainDataset,
            batch_size=trainParams.batchSize,
            shuffle=True,
            pin_memory=True,
            num_workers=numWorkers,
            collate_fn=dataset.collateFn,
            **trainLoaderKwargs,
        )
        valDataloader = DataLoader(
            valDataset,
            batch_size=trainParams.batchSize,
            shuffle=trainParams.valShuffle,
            pin_memory=True,
            num_workers=1,
            collate_fn=dataset.collateFn,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # ----- Step 3 - Initialize model, optimizer, lr
        numTrainingSteps = len(trainDataloader) * trainParams.numEpochs
        print("\n ------------------------------------ [main()] Initializing model")
        print(f" - [main()] Steps: {trainParams.steps}")
        print(f" - [main()] Learning Rate: {trainParams.learningRate}")

        modelName = modelParams.model
        print(f" - [main()] Model Name: {modelName}")
        print(f" - [main()] Num Training Steps: {numTrainingSteps}")
        print(f" - [main()] Num Epochs: {trainParams.numEpochs}")
        print(f" - [main()] Batch Size: {trainParams.batchSize}")

        # ----- Step 3.1 - Setup accelerator first so it owns device selection
        print("\n ------------------------------------ [main()] Initializing accelerator")
        print(f" - [main()] Gradient Accumulation Steps: {trainParams.gradientAccumulationSteps}")
        accelerator = Accelerator(
            mixed_precision="no" if not trainParams.mixedPrecision else trainParams.mixedPrecision,
            project_dir=trainParams.savePath,
            gradient_accumulation_steps=trainParams.gradientAccumulationSteps,
        )
        print(f" - [main()] Device: {accelerator.device}")

        diffusion = Diffusion(
            configInst.trainParams, 
            configInst.dataParams, 
            configInst.modelParams, 
            str(accelerator.device),
        )

        model = diffusion.decoder

        optimizerName = trainParams.optimizer
        if optimizerName == "Adam":
            print(" - [main()] Optimizer: Adam(weight_decay=None)")
            optimizer = torch.optim.Adam(model.parameters(), lr=trainParams.learningRate)
        else:
            print(" - [main()] Optimizer: AdamW(weight_decay=0.01)")
            optimizer = torch.optim.AdamW(model.parameters(), lr=trainParams.learningRate)

        lrScheduler = utils.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=trainParams.lrWarmupSteps,
            num_training_steps=numTrainingSteps,
        )

        print(f" - [main()] Loss: {trainParams.loss}")
        clipNorm = trainParams.clipNorm
        print(f" - [main()] Clip Norm: {clipNorm}")

        # ----- Step 4 - Prepare model / optimizer / dataloader with accelerate
        trainDataloader, model, optimizer, lrScheduler = accelerator.prepare(
            trainDataloader, model, optimizer, lrScheduler
        )
        # Route the diffusion wrapper through the prepared model (DDP wrapper / autocast)
        diffusion.decoder = model

        # ----- Step 5 - Training loop
        logEvery = trainParams.logEvery
        numEpochs = trainParams.numEpochs

        fid64 = FID(
            device=str(diffusion.device), featureCount=64, imageSize=dataParams.imageSize.width
        )

        for epoch in range(trainParams.numEpochs):
            try:
                currentLr = optimizer.param_groups[0]["lr"]
                hoursPassed = (time.time() - tScriptStart) / 3600

                print(
                    f"\n ------------------------- [main()] Epoch {epoch:04d}/{numEpochs:04d} (lr={currentLr:.5f}) (dataloader={len(trainDataloader)} iters @ batch={trainParams.batchSize}) ({datetime.datetime.now()}, hours={hoursPassed:.2f}) -------------------------"
                )

                # Step 5.1 - Train epoch
                lossAvgEpochTrain = trainEpoch(
                    diffusion, model, accelerator, trainDataloader, optimizer, lrScheduler, epoch, configInst
                )

                # Step 5.2 - Save checkpoint
                saveModel = (epoch > 0 and epoch % logEvery == 0) or (epoch == numEpochs - 1)
                pathFolderModel = utils.save_checkpoint(
                    model, accelerator, optimizer, lrScheduler, trainParams.savePath, epoch, saveModel
                )

                # Step 5.3 - Evaluate
                evalScribbleClass = trainParams.evalScribbleClassOverride
                (
                    imagesVal,
                    realImages,
                    sampledImages,
                    avgPrecision,
                    avgSoftPrecision,
                    fidScore64,
                    imagesValRGB,
                    avgPrecisionReal,
                    avgPrecisionSynthetic,
                    imagesValReal,
                    imagesValSynthetic,
                    precisionCount,
                    precisionCountReal,
                    precisionCountSynthetic,
                ) = evaluate(
                    diffusion, valDataloader, pathFolderModel, configInst, epoch=epoch,
                    scribbleClassOverride=evalScribbleClass, fidMetric=fid64
                )
                imagesTrain, imagesTrainRGB = evaluateTrain(
                    diffusion, valDataset, valDatasetValIdxs, pathFolderModel, configInst, scribbleClassOverride=evalScribbleClass
                )

                # Step 5.4 - Log to wandb
                if logWandb:
                    hoursPassed = (time.time() - tScriptStart) / 3600
                    wandb.log(
                        {
                            # Inception-feature distance to a single noise template (NOT a
                            # population FID); goes up as outputs become more structured.
                            "Inception Template Distance (64)": fidScore64,
                            "Precision": avgPrecision,
                            "Soft Precision": avgSoftPrecision,
                            "Precision/Real": avgPrecisionReal,
                            "Precision/Synthetic": avgPrecisionSynthetic,
                            "Precision/Count": precisionCount,
                            "Precision/RealCount": precisionCountReal,
                            "Precision/SyntheticCount": precisionCountSynthetic,
                            "epoch": epoch,
                            "Learning Rate": optimizer.param_groups[0]["lr"],
                            "Hours Passed": hoursPassed,
                            "Avg Loss (Epoch)": lossAvgEpochTrain,
                            # "Images (Validation)": imagesVal,
                            # "Images (Validation Real)": imagesValReal,
                            # "Images (Validation Synthetic)": imagesValSynthetic,
                            "Images/ValidationRealGridCount": len(imagesValReal),
                            "Images/ValidationSyntheticGridCount": len(imagesValSynthetic),
                            # "Images (Train)": imagesTrain,
                            "Images (Validation) - RGB": imagesValRGB,
                            "Images (Train) - RGB": imagesTrainRGB,
                        },
                        step=epoch,
                        commit=True,
                    )

            except Exception as e:
                print(f"\n ------------------------- [main()] Error in epoch {epoch}: {e}")
                traceback.print_exc()
                sys.exit(1)

    except ValueError as e:
        print(f" - [main()] Config validation error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f" - [main()] Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


######################################################################
# CONFIG PARSING
######################################################################

def loadConfig(configPath: str) -> Config:
    """Load JSON config file and create Pydantic Config instance.
    
    Parameters
    ----------
    configPath : str
        Path to JSON config file
        
    Returns
    -------
    Config
        Validated Pydantic Config instance with defaults applied
    """
    # Step 1 - Load JSON from file
    with open(configPath, "r") as f:
        configData = json.load(f)
    
    # Step 2 - Create Pydantic Config instance (validates and applies defaults)
    config = Config(**configData)
    
    return config


def parseArgs(argsParameters=None) -> Optional[Config]:
    """Parse command-line arguments and load config.
    
    Parameters
    ----------
    argsParameters : list, optional
        Command-line arguments for testing
        
    Returns
    -------
    Config or None
        Validated Pydantic Config instance or None if no config provided
    """
    # Step 1 - Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train diffusion model for scribble segmentation")
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    if argsParameters is not None:
        args = parser.parse_args(argsParameters)
    else:
        args = parser.parse_args()

    # Step 2 - Load config if provided
    if args.config:
        print(f"\n - [parseArgs()] Loading config from {args.config}")
        configInstance = loadConfig(args.config)
        # Store original config path for reference
        configInstance.configPath = args.config
        return configInstance
    else:
        print(" - [parseArgs()] No config provided")
        return None


def validateConfiguredRealDataPaths(configInst: Config) -> None:
    """Validate configured real-data directories before dataset construction.

    Parameters
    ----------
    configInst : Config
        Loaded training configuration.

    Raises
    ------
    ValueError
        If one or more configured real-data paths are missing or not directories.
    """
    missingPaths = []
    nonDirectoryPaths = []

    for pathText in configInst.dataParams.pathsToRealData:
        path = Path(pathText)
        if not path.exists():
            missingPaths.append(pathText)
        elif not path.is_dir():
            nonDirectoryPaths.append(pathText)

    if missingPaths or nonDirectoryPaths:
        errorParts = []
        if missingPaths:
            errorParts.append("missing data path(s): " + ", ".join(missingPaths))
        if nonDirectoryPaths:
            errorParts.append("non-directory path(s): " + ", ".join(nonDirectoryPaths))

        configLabel = configInst.configPath or "<unknown config>"
        raise ValueError(
            f"Config {configLabel} has invalid real-data path(s): " + "; ".join(errorParts)
        )


######################################################################
# ENTRY POINT
######################################################################

if __name__ == "__main__":
    configInst = parseArgs()
    if configInst is not None:
        main(configInst)

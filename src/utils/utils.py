######################################################################
# UTILITY FUNCTIONS - CLEANED VERSION
######################################################################

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import copy
import datetime
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# Pip/Third-party installs
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Application/Local module imports
import src.configConstants as configConstants

######################################################################
# GLOBALS
######################################################################

DIR_SRC  = Path(__file__).parent.parent.absolute()
DIR_ROOT = DIR_SRC.parent
DIR_TMP  = DIR_ROOT / configConstants.FOLDERNAME_TMP
Path(DIR_TMP).mkdir(parents=True, exist_ok=True)

######################################################################
# NORMALIZATION FUNCTIONS
######################################################################

def minMaxNormalize(someTensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range per sample.

    Parameters
    ----------
    someTensor : torch.Tensor
        Input tensor to normalize

    Returns
    -------
    torch.Tensor
        Normalized tensor with values in [0, 1] range per sample
    """
    someTensorClone = someTensor.clone()
    try:
        someTensorClone = someTensorClone.view(someTensor.size(0), -1)
        someTensorCloneMin = someTensorClone.min(1, keepdim=True)[0].unsqueeze(1).unsqueeze(1)
        someTensorCloneMax = someTensorClone.max(1, keepdim=True)[0].unsqueeze(1).unsqueeze(1)
        # Guard against constant samples (max == min) producing NaN
        someTensorRange = (someTensorCloneMax - someTensorCloneMin).clamp(min=1e-8)
        someTensorClone = (someTensor - someTensorCloneMin) / someTensorRange
    except Exception as e:
        print(f" - [minMaxNormalize()] Error: {e}")
        traceback.print_exc()
    return someTensorClone


######################################################################
# PLOTTING FUNCTIONS
######################################################################

def plot_batch_sgp_images(scribble, gt, prediction, title, fileName, dir=DIR_TMP):
    """
    Plots a batch of RGB images combining scribble, gt, and prediction.

    Parameters
    ----------
    scribble : torch.Tensor
        Batch of scribble images with shape (batch_size, 1, H, W)
    gt : torch.Tensor
        Batch of ground truth images with shape (batch_size, 1, H, W)
    prediction : torch.Tensor
        Batch of prediction images with shape (batch_size, 1, H, W)
    title : str
        Title for the figure
    fileName : str
        Name of the file to save the figure
    dir : Path
        Directory to save the figure
    """
    
    try:
        # Step 0 - Combine scribble, gt, and prediction into 3-channel RGB image
        imgs = torch.cat([prediction, gt, scribble], dim=1).detach().cpu()
        batchSize = imgs.shape[0]
        
        # Step 1 - Create figure
        f = plt.figure()
        f.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        
        # Step 2 - Plot each image in batch
        cols = math.ceil(math.sqrt(batchSize))
        rows = math.ceil(batchSize / cols)
        for i in range(batchSize):
            img = imgs[i].numpy().transpose(1, 2, 0)
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Image {i}", fontsize=8)

        # Step 3 - Save figure
        plt.savefig(dir / fileName)
        plt.close()
        
    except Exception:
        traceback.print_exc()
        print(" - [Error] Unable to plot and save RGB batch images")

def plot_batch_images(imgs, title, fileName, dir=DIR_TMP, scribble_meta=None):
    """
    Plots a batch of grayscale images as subplots and saves.

    Parameters
    ----------
    imgs : torch.Tensor
        Batch of images with shape (batch_size, 1, H, W)
    title : str
        Title for the figure
    fileName : str
        Name of the file to save the figure
    dir : Path
        Directory to save the figure
    scribble_meta : list, optional
        Metadata for each image in the batch
    """
    
    try:
        # Step 0 - Init
        batchSize = imgs.shape[0]
        
        # Step 1 - Create figure
        f = plt.figure()
        f.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        
        # Step 2 - Plot each image in batch
        cols = math.ceil(math.sqrt(batchSize))
        rows = math.ceil(batchSize / cols)
        for i in range(batchSize):
            img = imgs[i].squeeze().detach().cpu().numpy()
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img, cmap="Oranges", vmin=0, vmax=1)
            ax.set_title(f"Batch {i}", fontsize=8)
        
        # Step 3 - Save figure
        plt.savefig(dir / fileName)
        plt.close()

        # Step 4 - Save image filenames if metadata provided
        if scribble_meta is not None:
            filenames = [str(scribble_meta[i][configConstants.KEY_SCRIBBLE_PATH]) for i in range(batchSize)]
            filenamesFile = dir / (Path(fileName).stem + "_filenames.txt")
            with open(filenamesFile, "w") as fNames:
                for fname in filenames:
                    fNames.write(fname + "\n")

    except:
        traceback.print_exc()
        print(" - [Error] Unable to visualize images")

def plot_image(img, title: str, fileName: str, dir=DIR_TMP):
    """
    Plots a single image and saves it.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor
    title : str
        Title for the plot
    fileName : str
        Name of the file to save
    dir : Path
        Directory to save the figure
    """
    
    try:
        plt.figure(figsize=(5, 5))
        plt.imshow(img.squeeze().detach().cpu().numpy(), cmap="Oranges", vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
        plt.savefig(dir / fileName)
        plt.close()
        
    except Exception:
        traceback.print_exc()
        print(" - [Error] Unable to plot and save image")


def plotForInference(
    sampled: torch.Tensor,
    noise: torch.Tensor,
    scribble: Optional[torch.Tensor] = None,
    pathFolderModel: Optional[Path] = None,
    mode: Optional[str] = None,
    meta: Optional[Any] = None,
):
    """
    Plot inference results for visualization and debugging.
    
    Parameters
    ----------
    sampled : torch.Tensor
        Generated/sampled images
    noise : torch.Tensor
        Noise tensor used in diffusion
    scribble : torch.Tensor, optional
        Scribble input condition
    pathFolderModel : Path, optional
        Directory to save plots (defaults to DIR_TMP)
    mode : str, optional
        Mode identifier for filename
    meta : Any, optional
        Metadata for filename
    """
    try:
        cols = 2 if scribble is None else 3
        f, axarr = plt.subplots(len(sampled), cols, figsize=(3, len(sampled) * 2))
        fontsize = 6

        for batchId in range(len(sampled)):
            axarr[batchId, 0].imshow(sampled[batchId, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            axarr[batchId, 1].imshow(noise[batchId, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            axarr[batchId, 0].axis("off")
            axarr[batchId, 1].axis("off")

            if scribble is not None:
                axarr[batchId, 2].imshow(scribble[batchId, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                axarr[batchId, 2].axis("off")

            if batchId == 0:
                axarr[batchId, 0].set_title("Sampled", fontsize=fontsize)
                axarr[batchId, 1].set_title("Noise", fontsize=fontsize)
                if scribble is not None:
                    axarr[batchId, 2].set_title(
                        f"Scribble (sum={scribble[batchId, 0].sum().item():.2f})", fontsize=fontsize
                    )

        if pathFolderModel is None:
            pathFolderModel = DIR_TMP
        pathFolderModel.mkdir(parents=True, exist_ok=True)

        plt.suptitle(f"Model: {'/'.join(Path(pathFolderModel).parts[-2:])}", y=0.9)
        pathFile = f"{pathFolderModel}/sampled__{mode}__{meta}.png"
        print(f"   --- [plotForInference()] Saving image to: {pathFile}")
        plt.savefig(pathFile, bbox_inches="tight", dpi=100)
        plt.close()

    except Exception as e:
        print(f" - [plotForInference()] Error: {e}")
        traceback.print_exc()

def plotScribbleDatasetBatch(
    gt: torch.Tensor,
    prediction: torch.Tensor,
    scribble: torch.Tensor,
    scribbleGauss: torch.Tensor,
    scribbleType: torch.Tensor,
    meta: list,
    dirSave: Path,
):
    """
    Visualize a batch of scribble dataset samples with GT, prediction, and scribbles.
    
    Parameters
    ----------
    gt : torch.Tensor, shape=(B, 1, H, W)
        Ground truth masks
    prediction : torch.Tensor, shape=(B, 1, H, W)
        Prediction masks
    scribble : torch.Tensor, shape=(B, 1, H, W)
        Scribble masks
    scribbleGauss : torch.Tensor, shape=(B, 1, H, W)
        Gaussianized scribble masks
    scribbleType : torch.Tensor, shape=(B,)
        Scribble types (0/1 for background/foreground)
    meta : list[dict]
        Metadata for each sample
    dirSave : Path
        Directory to save plots
    """
    try:
        # Step 0 - Initialize figure
        f, axarr = plt.subplots(len(gt), 3, figsize=(6, 2 * len(gt)))
        titleFontSize = 3

        for batchId in range(len(gt)):
            # Step 1 - Extract data
            gtViz = gt[batchId].squeeze().cpu().numpy()
            predictionViz = prediction[batchId].squeeze().cpu().numpy()
            scribbleViz = scribble[batchId].squeeze().cpu().numpy()
            scribbleGaussViz = scribbleGauss[batchId].squeeze().cpu().numpy()
            scribbleTypeViz = scribbleType[batchId]
            metaViz = meta[batchId]
            
            metaScribbleGenType = metaViz[configConstants.KEY_SCRIBBLE_GENERATION_TYPE]
            metaScribblePath = metaViz[configConstants.KEY_SCRIBBLE_PATH]
            metaTransforms = copy.deepcopy(metaViz[configConstants.KEY_TRANSFORMS])
            
            if len(metaTransforms):
                for transformKey in metaTransforms:
                    metaTransforms[transformKey] = round(metaTransforms[transformKey], 2)
                metaTransforms = str(metaTransforms).replace('{', '').replace('}', '').replace(',', '\n').replace(':', '=')

            # Step 2 - Plot images
            gtLikeArray = np.ones((gtViz.shape[0], gtViz.shape[1]), dtype=np.uint8)
            axarr[batchId, 0].imshow(gtLikeArray, cmap='gray')
            axarr[batchId, 1].imshow(gtLikeArray, cmap='gray')
            axarr[batchId, 2].imshow(scribbleGaussViz, cmap='gray')

            # Step 3 - Plot contours
            axarr[batchId, 0].contour(gtViz, levels=[0.5], colors=[configConstants.KEY_COLOR_GREEN])
            axarr[batchId, 1].contour(gtViz, levels=[0.5], colors=[configConstants.KEY_COLOR_GREEN])
            axarr[batchId, 2].contour(gtViz, levels=[0.5], colors=[configConstants.KEY_COLOR_GREEN])

            axarr[batchId, 0].contour(predictionViz, levels=[0.5], colors=[configConstants.KEY_COLOR_RED])
            axarr[batchId, 1].contour(predictionViz, levels=[0.5], colors=[configConstants.KEY_COLOR_RED])
            axarr[batchId, 2].contour(predictionViz, levels=[0.5], colors=[configConstants.KEY_COLOR_RED])

            # Step 4 - Plot scribble
            scribbleColor = configConstants.KEY_COLOR_YELLOW if scribbleTypeViz == 1 else configConstants.KEY_COLOR_BLUE
            axarr[batchId, 1].contour(scribbleViz, levels=[0.5], colors=[scribbleColor])

            # Step 5 - Format axes
            axarr[batchId, 0].set_title(
                f'Type: {scribbleTypeViz} ({metaScribblePath.replace("/", chr(10)).replace("-Series-SEG", chr(10)).replace("-interaction.png", "")})',
                fontsize=titleFontSize, color='pink'
            )
            axarr[batchId, 1].set_title(
                f'Type: {scribbleTypeViz} ({metaScribbleGenType})',
                fontsize=titleFontSize, color='pink'
            )
            axarr[batchId, 2].set_title(
                f'Type: {scribbleTypeViz} ({metaScribbleGenType})\n{metaTransforms}',
                fontsize=titleFontSize, color='pink'
            )
            axarr[batchId, 0].set_xticks([])
            axarr[batchId, 1].axis('off')
            axarr[batchId, 2].axis('off')
            axarr[batchId, 0].set_ylabel(f'batchId:{batchId:02d}', fontsize=10, color='pink')

        # Step 6 - Save figure
        plt.suptitle("Foreground (yellow) vs Background (blue) scribble", y=0.95)
        Path(dirSave).mkdir(parents=True, exist_ok=True)
        pathFile = dirSave / f"scribble_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(pathFile, dpi=300, bbox_inches='tight')
        plt.close()

    except:
        traceback.print_exc()


def visLossV2(
    maskRegionWeight: float,
    errorRegion: torch.Tensor,
    errorRegionDilated: torch.Tensor,
    weights: torch.Tensor,
    scaledWeights: torch.Tensor,
    tmpDir: Optional[Path] = None,
):
    """
    Visualizes the error region, dilated error region, weights, and scaled weights.

    Parameters
    ----------
    maskRegionWeight : float
        Weight of the mask region
    errorRegion : torch.Tensor
        Error region tensor
    errorRegionDilated : torch.Tensor
        Dilated error region tensor
    weights : torch.Tensor
        Weight tensor
    scaledWeights : torch.Tensor
        Scaled weight tensor
    tmpDir : Path, optional
        Directory to save the visualization
    """
    try:
        # Step 0 - Initialize
        errorRegionViz = errorRegion.squeeze().detach().cpu().numpy()
        errorRegionDilatedViz = errorRegionDilated.squeeze().detach().cpu().numpy()
        weightsViz = weights.squeeze().detach().cpu().numpy()
        scaledWeightsViz = scaledWeights.squeeze().detach().cpu().numpy()
        batchSize = errorRegion.shape[0]

        # Step 1 - Create visualization grid
        fontsize = 6
        errorRegionMax = np.max(errorRegionViz)
        f, axarr = plt.subplots(batchSize, 4, figsize=(20, batchSize * 5))

        # Step 2 - Plot each batch
        for batchIdx in range(batchSize):
            axarr[batchIdx, 0].imshow(errorRegionViz[batchIdx], cmap="gray", vmin=0, vmax=errorRegionMax)
            axarr[batchIdx, 0].set_title(f"Error region\nUnique: {np.unique(errorRegionViz[batchIdx])}", fontsize=fontsize)

            axarr[batchIdx, 1].imshow(errorRegionDilatedViz[batchIdx], cmap="gray", vmin=0, vmax=errorRegionMax)
            axarr[batchIdx, 1].set_title(f"Dilated error region\nUnique: {np.unique(errorRegionDilatedViz[batchIdx])}", fontsize=fontsize)

            axarr[batchIdx, 2].imshow(weightsViz[batchIdx], cmap="gray", vmin=0, vmax=1)
            weightsUnique = sorted(np.unique(weightsViz[batchIdx]))
            weightsUniqueStr = ', '.join([f'{w:.3f}' for w in weightsUnique[:4]]) + ", ... " + ', '.join([f'{w:.3f}' for w in weightsUnique[-4:]])
            axarr[batchIdx, 2].set_title(f"Weights (AvgPool2D)\nsum={int(np.sum(weightsViz[batchIdx]))}, max={np.max(weightsViz[batchIdx]):.3f}\nUnique: {weightsUniqueStr}", fontsize=fontsize)

            scaledWeightsMin = np.min(scaledWeightsViz[batchIdx])
            scaledWeightsMax = np.max(scaledWeightsViz[batchIdx])
            axarr[batchIdx, 3].imshow(scaledWeightsViz[batchIdx], cmap="gray", vmin=0, vmax=scaledWeightsMax)
            scaledWeightsUnique = sorted(np.unique(scaledWeightsViz[batchIdx]))
            scaledWeightsUniqueStr = ', '.join([f'{w:.3f}' for w in scaledWeightsUnique[:4]]) + ", ... " + ', '.join([f'{w:.3f}' for w in scaledWeightsUnique[-4:]])
            axarr[batchIdx, 3].set_title(f"Scaled weights\nsum={int(np.sum(weightsViz))}, max={scaledWeightsMax:.3f}, min={scaledWeightsMin:.3f}\nUnique: {scaledWeightsUniqueStr}", fontsize=fontsize)

        plt.suptitle(f'Mask Region Weight={maskRegionWeight:.3f} | Error Region vmax={errorRegionMax}, Weights vmax=1', y=0.95)

        if tmpDir is None:
            tmpDir = DIR_TMP
        tmpDir.mkdir(parents=True, exist_ok=True)
        filePath = tmpDir / f"loss-test-{maskRegionWeight:.2f}-{np.random.randint(int(1e6))}.png"
        print(f" - [visLossV2()] Saving image to: {filePath}")
        plt.savefig(filePath, dpi=300)
        plt.close()

    except Exception as e:
        print(f" - [visLossV2()] Error: {e}")
        traceback.print_exc()

######################################################################
# CHECKPOINT FUNCTIONS
######################################################################

def get_last_checkpoint(save_dir: str):
    """
    Get the number of checkpoints in a directory.

    Parameters
    ----------
    save_dir : str
        Directory containing checkpoints

    Returns
    -------
    int
        Number of next checkpoint
    """
    
    nCheckpoints = len(
        list(filter(lambda x: x.startswith("checkpoint"), os.listdir(save_dir)))
    )
    return nCheckpoints + 1

def load_checkpoint(
    model, accelerator, optimizer, scheduler, save_dir: str, checkpoint_number: int
):
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load
    accelerator : Accelerator
        Accelerator to use
    optimizer : torch.optim.Optimizer
        Optimizer to load
    scheduler : torch.optim.lr_scheduler
        Scheduler to load
    save_dir : str
        Directory to load the checkpoint from
    checkpoint_number : int
        Checkpoint number to load
    """
    
    try:
        # Step 0 - Wait for all processes
        accelerator.wait_for_everyone()

        # Step 1 - Load model
        path = Path(save_dir) / f"checkpoint-{checkpoint_number}"
        unwrappedModel = accelerator.unwrap_model(model)
        unwrappedModel.load_state_dict(torch.load(os.path.join(path, "model.bin")))

        # Step 2 - Load optimizer and scheduler (main process only)
        if accelerator.is_main_process:
            optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))
            
    except:
        traceback.print_exc()

def save_checkpoint(
    model, accelerator, optimizer, scheduler, save_dir: str, epoch: int, save: bool = False
):
    """
    Save a checkpoint to disk.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    accelerator : Accelerator
        Accelerator to use
    optimizer : torch.optim.Optimizer
        Optimizer to save
    scheduler : torch.optim.lr_scheduler
        Scheduler to save
    save_dir : str
        Directory to save the checkpoint
    epoch : int
        Epoch number
    save : bool, default=False
        Whether to actually save the checkpoint

    Returns
    -------
    Path
        Path to the checkpoint folder
    """
    
    pathFolder = None

    try:
        # Step 0 - Init
        tModelSaveStart = time.time()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        modelFolderName = f"checkpoint-{epoch:04d}"
        pathFolder = Path(save_dir) / modelFolderName
        Path(pathFolder).mkdir(parents=True, exist_ok=True)
    
        # Step 1 - Save checkpoint
        if save:
            print(f"   --- [save_checkpoint()] Saving checkpoint at {pathFolder}")
            
            accelerator.wait_for_everyone()
            unwrappedModel = accelerator.unwrap_model(model)
            torch.save(unwrappedModel.state_dict(), Path(pathFolder) / configConstants.FILENAME_MODEL)

            if accelerator.is_main_process:
                torch.save(optimizer.state_dict(), Path(pathFolder) / configConstants.FILENAME_OPTIMIZER)
                torch.save(scheduler.state_dict(), Path(pathFolder) / configConstants.FILENAME_SCHEDULER)
            
            tModelSaveEnd = time.time()
            print(f"   --- [save_checkpoint()] Model saved in {tModelSaveEnd - tModelSaveStart:.2f} sec")
            
    except:
        traceback.print_exc()
        raise

    return pathFolder


######################################################################
# LEARNING RATE SCHEDULER
######################################################################

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule with cosine annealing and warmup.
    
    The learning rate decreases following cosine values from initial lr to 0,
    after a warmup period during which it increases linearly from 0 to initial lr.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer for which to schedule the learning rate
    num_warmup_steps : int
        The number of steps for the warmup phase
    num_training_steps : int
        The total number of training steps
    num_cycles : float, default=0.5
        The number of periods of the cosine function
    last_epoch : int, default=-1
        The index of the last epoch when resuming training

    Returns
    -------
    LambdaLR
        Learning rate scheduler with the appropriate schedule
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        return max(
            1e-4, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
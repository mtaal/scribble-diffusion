"""Inference module for morphologically-dependent interactive scribble segmentation."""

# Standard library imports
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Third-party imports
import json
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from tqdm import tqdm

# Local imports (torch-only, no training stack)
from src.models.schedule import computeVarianceSchedule

BACKGROUND_SCRIBBLE_CLASS = 0
FOREGROUND_SCRIBBLE_CLASS = 1

# Defaults matching configs/default.json; override via `configPath` to stay in sync
# with the checkpoint that was exported.
DEFAULT_STEPS = 50
DEFAULT_SCHEDULER_TYPE = "cosine"
DEFAULT_SCHEDULER_ARGS = {"beta1": 1e-4, "betaT": 0.2, "s": 0.008, "maxBeta": 0.999}
DEFAULT_INPUT_SIZE = 128
# Same threshold as the training-time evaluation (train.evaluate uses > 0.5)
DEFAULT_SCRIBBLE_THRESHOLD = 0.5

######################################################################
# HELPER FUNCTIONS
######################################################################

def decoder(x, t, conditions, scribbleClass, ortSession):
    """Run decoder inference using ONNX runtime.

    Args:
        x: Input tensor.
        t: Time step tensor.
        conditions: Condition tensor.
        ortSession: ONNX runtime session.

    Returns:
        Decoder output or None if session is not initialized.
    """
    # Step 0: Validate ONNX session
    if ortSession is None:
        print("Error: ONNX session is not initialized.")
        return None
    
    # Step 1: Prepare input dictionary for ONNX model
    inputs = {
        "x": x,
        "t": t,
        "scribbleClass": scribbleClass,
        "conditionGt": conditions[0],
        "conditionPred": conditions[1],
    }
    
    # Step 2: Run inference
    result = ortSession.run(None, inputs)
    
    return result

######################################################################
# CORE PROCESSING FUNCTIONS
######################################################################

def loadInferenceSettings(configPath: Optional[str]) -> dict:
    """Read the diffusion settings the ONNX model was trained with from a config JSON.

    Args:
        configPath: Path to a training config (configs/*.json) or None for the defaults.

    Returns:
        dict with keys steps, schedulerType, schedulerArgs, inputSize.
    """
    settings = {
        "steps": DEFAULT_STEPS,
        "schedulerType": DEFAULT_SCHEDULER_TYPE,
        "schedulerArgs": dict(DEFAULT_SCHEDULER_ARGS),
        "inputSize": DEFAULT_INPUT_SIZE,
    }
    if configPath is None:
        return settings
    with open(configPath, "r", encoding="utf-8") as handle:
        configData = json.load(handle)
    trainParams = configData.get("trainParams", {})
    dataParams = configData.get("dataParams", {})
    settings["steps"] = int(trainParams.get("steps", settings["steps"]))
    settings["schedulerType"] = trainParams.get("schedulerType", settings["schedulerType"])
    settings["schedulerArgs"].update(trainParams.get("schedulerArgs", {}))
    imageSize = dataParams.get("imageSize", {})
    width, height = imageSize.get("width"), imageSize.get("height")
    if width is not None and height is not None:
        if width != height:
            raise ValueError(f"Inference expects a square image size, config has {width}x{height}")
        settings["inputSize"] = int(width)
    return settings


def diffusionProcess(
    gtPredImage: Image.Image,
    scribbleClass: int,
    onnxModelPath: str,
    device: str = "cpu",
    configPath: Optional[str] = None,
    scribbleThreshold: float = DEFAULT_SCRIBBLE_THRESHOLD,
    debugDir: Optional[str] = None,
) -> Image.Image:
    """Process an image using a diffusion loop.

    - Resizes image to the model input size.
    - Separates channels: red (prediction) and green (ground truth).
    - Runs the DDPM reverse process (same update rule as Diffusion.sample in training).
    - Returns a new image with result in blue channel.

    Args:
        gtPredImage: Input PIL Image with 3 channels (R=prediction, G=ground truth).
        scribbleClass: 0 = background scribble (on FP), 1 = foreground scribble (on GT).
        onnxModelPath: Path to the ONNX model file.
        device: Device to run computation on (default: "cpu").
        configPath: Training config JSON of the exported checkpoint; provides steps,
            scheduler type/args and image size. None uses the defaults of configs/default.json.
        scribbleThreshold: Threshold applied to the clamped output to binarise the scribble.
        debugDir: When set, the conditions and clipped output are saved as PNGs there.

    Returns:
        Processed PIL Image with diffusion result in blue channel.
    """
    try:
        # Step 0: Initialize configuration parameters
        settings = loadInferenceSettings(configPath)
        steps = settings["steps"]
        schedulerArgs = settings["schedulerArgs"]
        schedulerType = settings["schedulerType"]
        inputSize = settings["inputSize"]

        print("Diffusion Configuration:")
        print(f"  device: {device}")
        print(f"  steps: {steps}")
        print(f"  schedulerType: {schedulerType}")
        print(f"  schedulerArgs: {schedulerArgs}")
        print(f"  inputSize: {inputSize}")

        # Step 1: Compute variance schedule for diffusion
        varScheduler = computeVarianceSchedule(steps, schedulerType, **schedulerArgs)
        alphaBar = varScheduler[0]
        beta = varScheduler[1]

        # Step 2: Convert PIL image to NumPy and resize
        gtPredNp = np.array(gtPredImage)
        gtPredNp = cv2.resize(
            gtPredNp, (inputSize, inputSize), interpolation=cv2.INTER_NEAREST
        )

        # Step 3: Binarize channels (ensure values are 0 or 255)
        predChannel = gtPredNp[:, :, 0]  # Red channel = prediction
        gtChannel = gtPredNp[:, :, 1]    # Green channel = ground truth
        predChannel[predChannel > 0] = 255
        gtChannel[gtChannel > 0] = 255
        gtPredNp[:, :, 0] = predChannel  # Red channel = prediction
        gtPredNp[:, :, 1] = gtChannel    # Green channel = ground truth
        gtPredNp[:, :, 2] = 0

        # Step 4: Extract and prepare tensor channels
        pred = torch.tensor(gtPredNp[:, :, 0], dtype=torch.uint8)  # Red channel = prediction
        gt = torch.tensor(gtPredNp[:, :, 1], dtype=torch.uint8)    # Green channel = ground truth
        predR = pred
        gtR = gt
        pred = pred.numpy()
        gt = gt.numpy()

        # Step 5: Add batch and channel dimensions
        pred = torch.tensor(pred, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
        gt = torch.tensor(gt, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)

        # Step 6: Normalize conditions to [0, 1] (kept separate for ONNX input)
        conditionGt = gt.float().to(device) / 255.0
        conditionPred = pred.float().to(device) / 255.0

        # Step 7: Save conditions for debugging (opt-in)
        if debugDir is not None:
            Path(debugDir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            conditionsRgb = torch.stack([predR, gtR, torch.zeros_like(gtR)], dim=0)
            plt.imsave(str(Path(debugDir) / f"conditions_{timestamp}.png"),
                       conditionsRgb.numpy().transpose(1, 2, 0))

        # Step 8: Initialize random noise tensor
        batchSize = 1
        xt = torch.randn(batchSize, 1, inputSize, inputSize, device=device)

        # Step 9: Load ONNX model session
        ortSession = ort.InferenceSession(onnxModelPath)

        # Step 10: Run backward diffusion loop
        with tqdm(total=steps, desc="Diffusion", unit="step") as pbar:
            for idx, t in enumerate(range(steps, 0, -1)):
                # Step 10a: Generate random noise (0 for final step)
                z = torch.randn_like(xt, device=device) if t > 1 else 0
                tTensor = torch.tensor([t], device=device)

                # Step 10b: Predict noise using decoder
                noiseOutput = decoder(xt.numpy(), tTensor.numpy(), (conditionGt.numpy(), conditionPred.numpy()),
                                      np.array([scribbleClass], dtype=np.int64), ortSession)[0]
                noise = torch.tensor(noiseOutput, dtype=torch.float32, device=device) if not isinstance(noiseOutput, torch.Tensor) else noiseOutput.to(device)

                # Step 10c: Compute diffusion parameters (identical to Diffusion.sample)
                alpha = 1 - beta[tTensor]
                alphaBar_ = alphaBar[tTensor]
                sigma = torch.sqrt(beta[tTensor])

                # Step 10d: Update xt using DDPM reverse process
                xt = (
                    (1 / torch.sqrt(alpha))
                    * (xt - ((1 - alpha) / torch.sqrt(1 - alphaBar_) * noise))
                )

                # Step 10e: Add noise if not final step
                if t > 1:
                    xt += sigma * z
                
                pbar.update(1)

        # Step 11: Clamp and threshold output (same threshold as training-time evaluation)
        xtClipped = torch.clamp(xt, 0.0, 1.0)
        xtClipped = (xtClipped > scribbleThreshold).to(torch.float32)

        # Step 12: Create RGB result image
        resultImg = np.zeros((inputSize, inputSize, 3), dtype=np.uint8)
        resultImg[:, :, 0] = pred.numpy()  # Red channel = prediction
        resultImg[:, :, 1] = gt.numpy()    # Green channel = ground truth
        resultImg[:, :, 2] = (
            xtClipped.squeeze(0).squeeze(0).numpy() * 255
        ).astype(np.uint8)  # Blue channel = diffusion result

        # Step 13: Save intermediate result for debugging (opt-in)
        if debugDir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = str(Path(debugDir) / f"xt_clipped_{timestamp}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(xtClipped.squeeze(0).squeeze(0).numpy(), cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.close()

        return Image.fromarray(resultImg, mode="RGB")
    except Exception as e:
        # Re-raise so callers cannot mistake the unchanged input for a result
        print(f"Error during diffusion process: {e}")
        raise


######################################################################
# FILE I/O FUNCTIONS
######################################################################

def processImageFile(
    inputPath: str,
    outputPath: str,
    onnxModelPath: str,
    scribbleClass: int,
    device: str = "cpu",
    configPath: Optional[str] = None,
    scribbleThreshold: float = DEFAULT_SCRIBBLE_THRESHOLD,
    debugDir: Optional[str] = None,
) -> bool:
    """Process a PNG image file using diffusion and save the result.

    Args:
        inputPath: Path to input PNG file with 3 channels (R=prediction, G=ground truth).
        outputPath: Path where the output image will be saved.
        onnxModelPath: Path to the ONNX model file.
        scribbleClass: 0 = background scribble, 1 = foreground scribble.
        device: Device to run computation on (default: "cpu").
        configPath: Training config JSON of the exported checkpoint (see diffusionProcess).
        scribbleThreshold: Binarisation threshold for the generated scribble.
        debugDir: Optional directory for debug PNGs.

    Returns:
        True if processing succeeded, False otherwise.
    """
    try:
        # Step 1: Validate input path exists
        inputFile = Path(inputPath)
        if not inputFile.exists():
            print(f"Error: Input file not found: {inputPath}")
            return False

        # Step 2: Verify file extension
        if not inputFile.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            print(f"Warning: File extension is {inputFile.suffix}, expected .png")

        # Step 3: Load the input image
        gtPredImage = Image.open(inputPath)
        
        # Step 4: Ensure image is RGB (3 channels)
        if gtPredImage.mode != "RGB":
            print(f"Converting image from {gtPredImage.mode} to RGB")
            gtPredImage = gtPredImage.convert("RGB")

        # Step 5: Run diffusion process
        resultImage = diffusionProcess(
            gtPredImage, scribbleClass, onnxModelPath, device,
            configPath=configPath, scribbleThreshold=scribbleThreshold, debugDir=debugDir,
        )

        # Step 6: Create output directory if needed
        outputFile = Path(outputPath)
        outputFile.parent.mkdir(parents=True, exist_ok=True)

        # Step 7: Save the result image
        resultImage.save(outputPath)
        print(f"Successfully saved result to: {outputPath}")
        
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing image file: {e}")
        return False


######################################################################
# MAIN FUNCTION
######################################################################

def main() -> int:
    """
    Command-line interface for scribble generation with an exported ONNX model.

    Returns
    -------
    int
        Process exit code (0 on success, 1 on failure).
    """
    # Step 0 - Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate a scribble for a prediction/GT image pair with an exported ONNX model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input PNG with R=prediction mask, G=ground-truth mask (B is ignored)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG: R=prediction, G=ground truth, B=generated scribble"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="model-definition/scribbleModel.onnx",
        help="Path to the exported ONNX model"
    )
    parser.add_argument(
        "--scribble-class",
        type=int,
        required=True,
        choices=[BACKGROUND_SCRIBBLE_CLASS, FOREGROUND_SCRIBBLE_CLASS],
        help="0 = background scribble (on false positives), 1 = foreground scribble (on GT)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config of the exported checkpoint (steps, scheduler, image size); "
             "omit to use the defaults of configs/default.json"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCRIBBLE_THRESHOLD,
        help="Binarisation threshold for the generated scribble"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Optional directory for debug PNGs (conditions and clipped output)"
    )
    args = parser.parse_args()

    # Step 1 - Validate the model path up front for a clear error message
    if not Path(args.onnx).exists():
        print(f"Error: ONNX model not found: {args.onnx} (export one with python -m src.utils.saveAsOnnx)")
        return 1

    # Step 2 - Run inference
    ok = processImageFile(
        inputPath=args.input,
        outputPath=args.output,
        onnxModelPath=args.onnx,
        scribbleClass=args.scribble_class,
        configPath=args.config,
        scribbleThreshold=args.threshold,
        debugDir=args.debug_dir,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

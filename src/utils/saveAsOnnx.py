######################################################################
# ONNX EXPORT UTILITY - Model Conversion and Testing
######################################################################
"""
This module handles exporting PyTorch models to ONNX format and testing
the exported model with ONNX Runtime.

Functions:
    exportToOnnx(): Exports model to ONNX and runs inference test
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import json
import argparse
from pathlib import Path
from typing import Any, Sized

# Pip/Third-party installs
import torch
import onnxruntime as ort  # pip install onnxruntime

# Application/Local module imports
import src.configConstants as configConstants
from src.models.duUnetBlocks4 import Diffusion4BlocksUNet
from src.configModels import Config, ModelParams

######################################################################
# CONSTANTS
######################################################################

DIR_SRC = Path(__file__).parent.parent.absolute()
DIR_ROOT = DIR_SRC.parent
DIR_MODEL_DEF = DIR_ROOT / "model-definition"

######################################################################
# HELPER FUNCTIONS
######################################################################

def loadModel(
    modelParams: ModelParams,
    weightsPath: Path,
    device: str = "cpu",
    imageSize: int = 128,
    allowMissingKeys: bool = False,
):
    """
    Load model from checkpoint.

    Parameters
    ----------
    modelParams : ModelParams
        Model configuration parameters
    weightsPath : Path
        Path to model weights checkpoint
    device : str, default="cpu"
        Device to load model on
    imageSize : int, default=128
        Spatial size recorded in the model config
    allowMissingKeys : bool, default=False
        If False (default) any missing/unexpected key aborts the export, so a
        config/checkpoint mismatch cannot silently produce a partly random model.

    Returns
    -------
    torch.nn.Module
        Loaded model in eval mode
    """
    # Step 1 - Create model
    modelName = modelParams.model
    if modelName == configConstants.MODEL_DU_UNET_4BLOCKS:
        model = Diffusion4BlocksUNet(**modelParams.model_dump(), sampleSize=imageSize)
    else:
        raise ValueError(f"Model {modelName} is not supported.")
    
    model.to(torch.device(device))  # type: ignore
    model.eval()

    # Step 2 - Load weights
    stateDict = torch.load(weightsPath, map_location=device)
    
    # Step 2.1 - If checkpoint is a dict with 'state_dict' key, use that
    if "state_dict" in stateDict:
        stateDict = stateDict["state_dict"]
    
    # Step 2.2 - Remove 'module.' prefix if present (from DataParallel)
    stateDict = {k.replace("module.", ""): v for k, v in stateDict.items()}

    # Step 2.3 - Load and report every mismatch explicitly
    loadResult = model.load_state_dict(stateDict, strict=False)
    if loadResult.missing_keys:
        print(f"     ! {len(loadResult.missing_keys)} missing key(s) in checkpoint: {loadResult.missing_keys}")
    if loadResult.unexpected_keys:
        print(f"     ! {len(loadResult.unexpected_keys)} unexpected key(s) in checkpoint: {loadResult.unexpected_keys}")
    if (loadResult.missing_keys or loadResult.unexpected_keys) and not allowMissingKeys:
        raise RuntimeError(
            "Checkpoint does not match the model built from the config (see keys above). "
            "Check --config / --weights, or pass --allow-missing-keys to export anyway."
        )

    return model


def loadConfig(configPath: Path) -> Config:
    """
    Load a configuration JSON file.

    Parameters
    ----------
    configPath : Path
        Path to the config JSON file.

    Returns
    -------
    Config
        Parsed config object.
    """
    with open(configPath, "r") as handle:
        configData = json.load(handle)
    return Config(**configData)


######################################################################
# MAIN EXPORT FUNCTION
######################################################################

def exportToOnnx(
    modelParams: ModelParams,
    weightsPath: Path,
    onnxPath: Path,
    imageSize: int = 128,
    device: str = "cpu",
    testInference: bool = True,
    allowMissingKeys: bool = False,
) -> None:
    """
    Export PyTorch model to ONNX format and optionally test with ONNX Runtime.

    Parameters
    ----------
    modelParams : ModelParams
        Model configuration parameters
    weightsPath : Path
        Path to model weights checkpoint
    onnxPath : Path
        Path to save ONNX model
    imageSize : int, default=128
        Image dimension (assumes square images)
    device : str, default="cpu"
        Device to use for export
    testInference : bool, default=True
        Whether to test ONNX model after export
    allowMissingKeys : bool, default=False
        Export even if the checkpoint has missing/unexpected keys (see loadModel)
    """
    # Step 0 - Initialize
    print("\n --- [exportToOnnx()] Starting ONNX export")
    print(f"     Model weights: {weightsPath}")
    print(f"     ONNX output: {onnxPath}")

    # Step 1 - Load model
    model = loadModel(modelParams, weightsPath, device, imageSize=imageSize, allowMissingKeys=allowMissingKeys)

    # Step 2 - Prepare dummy input for export
    batchSize = 1
    t = torch.tensor([5], dtype=torch.long, device=device)
    x = torch.randn(batchSize, 1, imageSize, imageSize, device=device)
    scribbleClass = torch.tensor([0], dtype=torch.long, device=device)  # Example class (background scribble)
    
    # Step 2.1 - Create conditions if model is conditional
    conditionGt = torch.randn(batchSize, 1, imageSize, imageSize, device=device)
    conditionPred = torch.randn(batchSize, 1, imageSize, imageSize, device=device)

    # Step 3 - Export to ONNX
    inputNames = ["x", "t", "conditionGt", "conditionPred", "scribbleClass"]
    outputNames = ["output"]

    dynamicAxes = {
        "x": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        "output": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        "conditionGt": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        "conditionPred": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        "scribbleClass": {0: "batch_size"},
    }
    exportArgs = (x, t, (conditionGt, conditionPred), scribbleClass)

    torch.onnx.export(
        model,
        exportArgs,
        str(onnxPath),
        input_names=inputNames,
        output_names=outputNames,
        opset_version=17,
        dynamic_axes=dynamicAxes,
    )
    print(f"     ✓ Model exported to {onnxPath}")

    # Step 4 - Test ONNX model with ONNX Runtime
    if testInference:
        print("\n --- [exportToOnnx()] Testing ONNX inference")
        session = ort.InferenceSession(str(onnxPath))
        
        # Step 4.1 - Prepare inputs
        ortInputs = {
            "x": x.numpy(),
            "t": t.numpy(),
            "conditionGt": conditionGt.numpy(),
            "conditionPred": conditionPred.numpy(),
            "scribbleClass": scribbleClass.numpy(),
        }

        # Step 4.2 - Run inference
        ortOutputs = session.run(None, ortInputs)
        
        outputArray: Any = ortOutputs[0]
        if hasattr(outputArray, "shape"):
            outputShape = outputArray.shape
        elif isinstance(outputArray, Sized):
            outputShape = len(outputArray)
        else:
            outputShape = "unknown"
        print(f"     ✓ ONNX inference successful. Output shape: {outputShape}")


######################################################################
# MAIN FUNCTION
######################################################################

def main():
    """
    Command-line interface for ONNX export.
    """
    # Step 0 - Parse arguments
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to the config JSON file (relative to project root)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model-definition/model.bin",
        help="Path to model weights (relative to project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model-definition/scribbleModel.onnx",
        help="Output path for ONNX model (relative to project root)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="duUnet4Blocks",
        choices=["duUnet4Blocks"],
        help="Model architecture"
    )
    parser.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help="Export even if checkpoint keys do not match the config-built model (default: abort)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size from config (assumes square images)"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip ONNX inference test"
    )
    
    args = parser.parse_args()

    # Step 1 - Convert paths to absolute
    configPath = DIR_ROOT / args.config
    weightsPath = DIR_ROOT / args.weights
    onnxPath = DIR_ROOT / args.output

    # Step 1.1 - Ensure output directory exists
    onnxPath.parent.mkdir(parents=True, exist_ok=True)

    # Step 2 - Load config and derive export parameters from it
    configInst = loadConfig(configPath)
    modelParams = configInst.modelParams
    imageSize = args.image_size if args.image_size is not None else configInst.dataParams.imageSize.width

    if configInst.dataParams.imageSize.width != configInst.dataParams.imageSize.height:
        raise ValueError(
            "Export requires a square image size from the config, "
            f"got {configInst.dataParams.imageSize.width}x{configInst.dataParams.imageSize.height}"
        )

    # Step 3 - Export to ONNX
    exportToOnnx(
        modelParams=modelParams,
        weightsPath=weightsPath,
        onnxPath=onnxPath,
        imageSize=imageSize,
        device="cpu",
        testInference=not args.no_test,
        allowMissingKeys=args.allow_missing_keys,
    )

    print("\n --- [main()] Export complete!")


if __name__ == "__main__":
    main()

######################################################################
# MODEL SUMMARY UTILITY - Parameter Counts and Torchinfo Summary
######################################################################
"""
This module prints a parameter summary for a configured diffusion model.

Functions:
    printModelSummary(): Instantiates a model from config and prints summary data
"""

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
import argparse
import json
from pathlib import Path

# Pip/Third-party installs
import torch
from torchinfo import summary

# Application/Local module imports
from src.configModels import Config
from src.models.duUnetBlocks4 import Diffusion4BlocksUNet

######################################################################
# CONSTANTS
######################################################################

DIR_SRC = Path(__file__).parent.parent.absolute()
DIR_ROOT = DIR_SRC.parent

######################################################################
# HELPER FUNCTIONS
######################################################################

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


def printModelSummary(configPath: Path, device: str = "cpu", depth: int = 3) -> None:
    """
    Print parameter counts and a torchinfo summary for the configured model.

    Parameters
    ----------
    configPath : Path
        Path to the JSON config file.
    device : str, default="cpu"
        Device used for the dummy forward pass.
    depth : int, default=3
        Depth passed to torchinfo.summary.
    """
    configInst = loadConfig(configPath)
    modelParams = configInst.modelParams
    dataParams = configInst.dataParams

    if dataParams.imageSize.width != dataParams.imageSize.height:
        raise ValueError(
            "Model summary expects a square image size in the config, "
            f"got {dataParams.imageSize.width}x{dataParams.imageSize.height}"
        )

    imageSize = dataParams.imageSize.width
    model = Diffusion4BlocksUNet(**modelParams.model_dump())
    model.eval()

    x = torch.randn(1, 1, imageSize, imageSize, device=device)
    t = torch.tensor([5], dtype=torch.long, device=device)
    scribbleClass = torch.tensor([0], dtype=torch.long, device=device)

    if modelParams.conditional:
        conditionGt = torch.randn(1, 1, imageSize, imageSize, device=device)
        conditionPred = torch.randn(1, 1, imageSize, imageSize, device=device)
        inputData = (x, t, [conditionGt, conditionPred], scribbleClass)
    else:
        inputData = (x, t, None, scribbleClass)

    print(f"Config: {configPath}")
    print(f"Model: {modelParams.model}")
    print()

    summary(
        model,
        input_data=inputData,
        verbose=1,
        depth=depth,
        device=device,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        row_settings=("depth",),
    )


######################################################################
# MAIN FUNCTION
######################################################################

def main() -> None:
    """Command-line entry point for model summaries."""
    parser = argparse.ArgumentParser(description="Print a model summary and parameter counts")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to the config JSON file (relative to project root)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for the dummy forward pass",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth passed to torchinfo.summary",
    )

    args = parser.parse_args()

    configPath = DIR_ROOT / args.config
    printModelSummary(configPath=configPath, device=args.device, depth=args.depth)


if __name__ == "__main__":
    main()
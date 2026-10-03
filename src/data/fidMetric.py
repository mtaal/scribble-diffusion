######################################################################
# INCEPTION TEMPLATE DISTANCE - Image Structure Metric (FID machinery)
######################################################################
"""
Per-image "distance from noise" score built on torchmetrics' FID machinery.

NOTE: this is NOT a population FID. Each call compares ONE generated image
(duplicated to satisfy the >1-sample requirement) against ONE fixed noise
template image. With identical duplicates the covariance term vanishes, so the
score reduces to the squared distance between Inception feature means. Higher
means the output looks less like the noise template, i.e. more structured.
It is logged to wandb as "Inception Template Distance (64)".

Classes:
    FID: Computes the per-image template distance described above
"""

######################################################################
# IMPORTS
######################################################################

# Standard library
import pathlib

# Pip/Third-party installs
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# Default template: <project-root>/assets/templateImage.png
_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets"
_DEFAULT_TEMPLATE = _ASSETS_DIR / "templateImage.png"

######################################################################
# MAIN CLASS
######################################################################

class FID:
    
    def __init__(
        self,
        device: str = "cpu",
        featureCount: int = 192,
        imageSize: int = 128,
        templatePath: pathlib.Path = _DEFAULT_TEMPLATE,
    ):
        """
        Initialize FID metric calculator.

        Parameters
        ----------
        device : str, default="cpu"
            Device for computation ('cpu' or 'cuda')
        featureCount : int, default=192
            Number of Inception features to use
        imageSize : int, default=128
            Target image size for noise reference generation
        templatePath : str or None, default=None
            Path to the noise template image used as the real reference distribution.
            Defaults to ``_DEFAULT_TEMPLATE`` (``assets/templateImage.png`` in the project root).
        """
        # Step 0 - Initialize FID metric from torchmetrics
        self.fid = FrechetInceptionDistance(feature=featureCount)
        self.fid.set_dtype(torch.float64)
        self.fid = self.fid.to(device)

        self.device = device
        self.imageSize = imageSize

        # Step 1 - Load and pre-process the template image (RGB uint8, imageSize x imageSize)
        templatePil = Image.open(templatePath).convert("RGB")
        toTensor = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),                     # [0, 1] float32
        ])
        # Shape after toTensor: (3, H, W) float32; after unsqueeze: (1, 3, H, W) uint8
        templateTensorTemp: torch.Tensor = toTensor(templatePil)  # type: ignore
        self.templateTensor: torch.Tensor = (templateTensorTemp.unsqueeze(0) * 255).byte()

    def computeForImage(self, img: torch.Tensor) -> float:
        """
        Compute FID between a single generated image and the noise template reference.

        Higher FID = generated image diverges more from the template = more structured
        and scribble-like = better quality. Chart goes UP with training.

        Parameters
        ----------
        img : torch.Tensor
            Single generated image tensor shaped (1, H, W) or (3, H, W) in [0, 1].

        Returns
        -------
        float
            FID score (higher is better — measures divergence from the template distribution).
        """
        # Step 0 - Reset state
        self.fid.reset()

        # Step 1 - Convert to (1, 3, H, W) uint8
        if img.dim() == 3:
            img = img.unsqueeze(0)          # (1, C, H, W)
        imgBatch = (img * 255).byte().to(self.device)
        if imgBatch.shape[1] == 1:
            imgBatch = imgBatch.repeat(1, 3, 1, 1)

        # Step 2 - Update with fake (generated) and real (template) images
        # FID requires >1 sample per distribution to compute covariance; repeat if needed
        if imgBatch.shape[0] < 2:
            imgBatch = imgBatch.repeat(2, 1, 1, 1)
        realBatch = self.templateTensor.to(self.device)
        if realBatch.shape[0] < 2:
            realBatch = realBatch.repeat(2, 1, 1, 1)
        self.fid.update(imgBatch, real=False)
        self.fid.update(realBatch, real=True)

        # Step 3 - Compute and return
        score = self.fid.compute()
        self.fid.reset()
        return float(score)
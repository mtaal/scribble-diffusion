######################################################################
# DIFFUSION UNET 4 BLOCKS
######################################################################

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
from typing import Optional, Union, Tuple

# Pip/Third-party installs
import torch
from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput

# Application/Local module imports
from src.models.duBlocks4ControlNet import Diffusion4BlocksControlNet

######################################################################
# MAIN CLASS
######################################################################

class Diffusion4BlocksUNet(UNet2DModel):
    """
    Custom 4-block UNet architecture for diffusion models with optional ControlNet.
    
    This model extends the standard UNet2DModel with conditional generation capabilities
    through an integrated ControlNet. The architecture consists of 4 encoder blocks,
    a mid block, and 4 decoder blocks with skip connections.
    
    Architecture:
    - Encoder: DownBlock2D → DownBlock2D → AttnDownBlock2D → DownBlock2D
    - Mid: UNetMidBlock2D with attention
    - Decoder: UpBlock2D → AttnUpBlock2D → UpBlock2D → UpBlock2D
    - Channels: (128, 128, 256, 256)
    - Layers per block: 2
    
    Parameters
    ----------
    *args : tuple
        Variable positional arguments passed to parent
    **kwargs : dict
        Keyword arguments:
        - conditional (bool): Enable ControlNet for conditional generation
        - use_encoder_convs (bool): Use zero convolutions in ControlNet (default: True)
        - imageChannels (int): Channels of the (noisy) scribble sample (default: 1)
        - sampleSize (int or tuple): Spatial size recorded in the diffusers config (default: 128)
        - blockOutChannels, layersPerBlock, normNumGroups: architecture size (see ModelParams)
    
    Attributes
    ----------
    controlnet : Diffusion4BlocksControlNet or None
        ControlNet module for conditional generation (None if not conditional)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the 4-block UNet with optional ControlNet.
        
        Step 0 - Initialize parent UNet2DModel with 4-block architecture
        Step 1 - Conditionally create ControlNet if requested
        """
        
        # Step 0 - Resolve configurable architecture size (shared with ControlNet)
        # blockOutChannels/layersPerBlock/normNumGroups come straight from ModelParams
        # via **model_dump(); defaults reproduce the original (128,128,256,256) model.
        blockOutChannels = tuple(kwargs.get('blockOutChannels', (128, 128, 256, 256)))
        layersPerBlock = kwargs.get('layersPerBlock', 2)
        normNumGroups = kwargs.get('normNumGroups', 32)
        imageChannels = kwargs.get('imageChannels', 1)
        sampleSize = kwargs.get('sampleSize', 128)

        # The 4-block down/up architecture is fixed, so the channel list must have 4 entries.
        assert len(blockOutChannels) == 4, (
            f"duUnet4Blocks needs exactly 4 blockOutChannels, got {len(blockOutChannels)}"
        )
        # GroupNorm requires every channel count to be divisible by the group count.
        for channels in blockOutChannels:
            assert channels % normNumGroups == 0, (
                f"blockOutChannels value {channels} is not divisible by normNumGroups "
                f"{normNumGroups}; GroupNorm would fail."
            )

        # Step 0.1 - Initialize parent UNet2DModel
        super().__init__(
            sample_size=sampleSize,        # Input/output spatial resolution (from dataParams)
            in_channels=imageChannels,     # Scribble sample channels (from ModelParams)
            out_channels=imageChannels,    # Predicted noise has the same shape as the sample
            layers_per_block=layersPerBlock,    # ResNet layers per block (configurable)
            block_out_channels=blockOutChannels,  # Channel progression (configurable)
            norm_num_groups=normNumGroups,        # GroupNorm groups (configurable)
            down_block_types=(       # Encoder block architecture
                "DownBlock2D",       # Block 1: Basic downsampling
                "DownBlock2D",       # Block 2: Basic downsampling
                "AttnDownBlock2D",   # Block 3: Downsampling with self-attention
                "DownBlock2D",       # Block 4: Basic downsampling
            ),
            up_block_types=(         # Decoder block architecture (reversed)
                "UpBlock2D",         # Block 4: Basic upsampling
                "AttnUpBlock2D",     # Block 3: Upsampling with self-attention
                "UpBlock2D",         # Block 2: Basic upsampling
                "UpBlock2D",         # Block 1: Basic upsampling
            ),
        )

        # Step 1 - Conditionally create ControlNet for conditional generation
        # ControlNet residuals are added element-wise to this UNet's encoder/mid
        # outputs, so it must use the SAME channel/group/layer configuration.
        if kwargs.get('conditional', False):
            useEncoderConvs = kwargs.get('use_encoder_convs', True)
            self.controlnet = Diffusion4BlocksControlNet(
                useEncoderConvs=useEncoderConvs,
                blockOutChannels=blockOutChannels,
                layersPerBlock=layersPerBlock,
                normNumGroups=normNumGroups,
                imageChannels=imageChannels,
            ).to(dtype=self.dtype)
        else:
            self.controlnet = None
        
        # Step 2 - Create scribble class type embedding (foreground vs background)
        # Outputs 512-d directly to match tEmb — added straight in, no MLP reuse.
        # normal_(std=0.02) ensures the two class rows are distinguishable from epoch 0.
        self.scribbleClassEmbedding = torch.nn.Embedding(
            num_embeddings=2,       # 0=background (scribbles on FP), 1=foreground (scribbles on GT)
            embedding_dim=self.time_embedding.linear_2.out_features  # 512 — matches tEmb
        )
        torch.nn.init.normal_(self.scribbleClassEmbedding.weight, std=0.02)

    # ======= FORWARD METHODS ========
    
    def forward( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[list[torch.Tensor]] = None,
        scribbleClass: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ):
        """
        Main forward pass routing to conditional or standard UNet.
        
        Parameters
        ----------
        x : torch.Tensor
            Noisy input tensor, shape (batch, 1, height, width)
        t : torch.Tensor
            Timestep values for diffusion process
        conditions : list[torch.Tensor], optional
            List of conditioning tensors for ControlNet (if enabled)
        scribbleClass : torch.Tensor, optional
            Scribble class type indices, shape (batch,)
            0=background (scribbles on FP), 1=foreground (scribbles on GT)
        return_dict : bool, default=False
            Whether to return as dictionary (standard UNet)
        
        Returns
        -------
        torch.Tensor or UNet2DOutput
            Denoised output tensor or output dictionary
        """
        if self.controlnet is not None:
            return self.conditionalForward(x=x, t=t, conditions=conditions, scribbleClass=scribbleClass)
        return super().forward(x, timestep=t, return_dict=return_dict)

    def conditionalForward(
        self,
        x: torch.Tensor,
        t: Union[torch.Tensor, float, int],
        conditions: Optional[list[torch.Tensor]] = None,
        scribbleClass: Optional[torch.Tensor] = None
    ) -> Union[UNet2DOutput, Tuple]:
        """
        UNet forward pass with ControlNet-based conditional generation.
        
        This method implements the full diffusion UNet with ControlNet integration:
        1. Time embedding generation from timestep values
        2. Scribble class embedding (foreground vs background)
        3. Encoder pass through down blocks collecting noise residuals
        4. ControlNet pass generating conditional residuals from conditions
        5. Residual combination (noise + conditional) for skip connections
        6. Mid block processing with conditional residual addition
        7. Decoder pass through up blocks using combined residuals
        8. Final post-processing and output generation

        Parameters
        ----------
        x : torch.Tensor
            The noisy input tensor with shape (batch, 1, height, width)
            Represents the noisy sample at timestep t to be denoised
        t : torch.Tensor or float or int
            The diffusion timestep(s) for denoising schedule
            Can be scalar or tensor, will be broadcasted to batch size
        conditions : list[torch.Tensor], optional
            List of conditioning tensors (e.g., scribbles, masks, semantic maps)
            Each tensor has shape (batch, channels, height, width)
            Concatenated and passed to ControlNet for guidance
        scribbleClass : torch.Tensor, optional
            Scribble class type indices, shape (batch,)
            0=background (scribbles on FP), 1=foreground (scribbles on GT)
            Used to condition the model on scribble type

        Returns
        -------
        tuple
            Tuple containing the denoised sample tensor (batch, 1, height, width)
        """
        
        # Step 1 - Time embedding generation
        # Step 1.1 - Convert timestep to tensor if needed
        timesteps = t
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # Step 1.2 - Expand timesteps to match batch size
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        # Step 1.3 - Project timesteps to embedding space
        tProj = self.time_proj(timesteps).to(dtype=self.dtype)
        tEmb = self.time_embedding(tProj)
        
        # Step 1.4 - Add scribble class embedding to time embedding
        if scribbleClass is not None:
            # Ensure scribbleClass is on correct device and has correct shape
            scribbleClassLocal = scribbleClass.to(x.device)
            if len(scribbleClassLocal.shape) == 0:
                scribbleClassLocal = scribbleClassLocal.unsqueeze(0)
            # Embed to 512-d and add directly to tEmb.
            scribbleClassEmb = self.scribbleClassEmbedding(scribbleClassLocal)
            tEmb = tEmb + scribbleClassEmb

        # Step 2 - Pre-process
        # Step 2.1 - Store original sample for ControlNet
        originalSample = x
        # Step 2.2 - Apply initial convolution to sample
        hiddenStates = self.conv_in(x)

        # Step 3 - Execute encoder blocks and collect residuals
        # Step 3.1 - Initialize residuals tuple
        # start with one value, why because the last encoder block
        # does not do down sample causing one less entry in the residuals list
        # adding the original sample makes sure the number of entries match
        # for the decoder blocks
        noiseEncoderResiduals = (hiddenStates, )
        # Step 3.2 - Process each encoder block and collect outputs
        for encoderBlock in self.down_blocks:
            hiddenStates, resHiddenStates = encoderBlock(hidden_states=hiddenStates, temb=tEmb)
            noiseEncoderResiduals += resHiddenStates
        # Step 4 - ControlNet conditional residuals generation
        # Step 4.1 - Pass original sample through ControlNet to get conditional residuals
        # ControlNet processes noisy sample + conditions and returns:
        #   - encoderBlockConditionalResiduals: tuple of tensors matching encoder outputs
        #   - midBlockConditionalResidual: tensor for mid block addition
        # pyright: ignore[reportOptionalCall]
        noiseEncoderConditionalResiduals, midNoiseConditionalResidual = self.controlnet(
            originalSample, timesteps, conditions=conditions, scribbleClass=scribbleClass
        ) # pyright: ignore[reportOptionalCall]
        
        # Step 4.2 - Combine noise and conditional residuals element-wise
        # Each encoder block output (noise residual) is added to corresponding
        # ControlNet output (conditional residual) for enhanced skip connections
        combinedNoiseEncoderResiduals = list(
            noiseEncoderResidual + noiseEncoderConditionalResidual
            for noiseEncoderResidual, noiseEncoderConditionalResidual in zip(
                noiseEncoderResiduals, noiseEncoderConditionalResiduals
            )
        )

        # Step 5 - Mid block processing
        if self.mid_block is not None:
            # Step 5.1 - Add mid block conditional residual to hidden states
            # This injects ControlNet guidance at the bottleneck of the UNet
            hiddenStates = hiddenStates + midNoiseConditionalResidual
            # Step 5.2 - Process through mid block with attention
            hiddenStates = self.mid_block(hiddenStates, tEmb)

        # Step 6 - Decoder blocks (upsampling path)
        for decoderBlock in self.up_blocks:
            # Step 6.1 - Extract residuals for current decoder block (LIFO order)
            # Each decoder block needs residuals from corresponding encoder block
            # Number of residuals matches number of resnets in the decoder block
            resHiddenStates = combinedNoiseEncoderResiduals[-len(decoderBlock.resnets) :]
            combinedNoiseEncoderResiduals = combinedNoiseEncoderResiduals[: -len(decoderBlock.resnets)]
            # Step 6.2 - Process through decoder block with skip connections
            # Decoder upsamples and combines with residuals via concatenation
            hiddenStates = decoderBlock(hiddenStates, resHiddenStates, tEmb)

        # Step 7 - Post-processing to generate final output
        # Step 7.1 - Apply group normalization
        hiddenStates = self.conv_norm_out(hiddenStates)
        # Step 7.2 - Apply SiLU activation function
        hiddenStates = self.conv_act(hiddenStates)
        # Step 7.3 - Apply final 3x3 convolution to get output channels
        hiddenStates = self.conv_out(hiddenStates)

        return hiddenStates
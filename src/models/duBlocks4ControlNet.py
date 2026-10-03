######################################################################
# DIFFUSION 4 BLOCKS CONTROLNET
######################################################################

######################################################################
# IMPORTS
######################################################################

# General/Standard library imports
from typing import Optional

# Pip/Third-party installs
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import get_down_block, UNetMidBlock2D
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

######################################################################
# MAIN CLASS
######################################################################

class Diffusion4BlocksControlNet(nn.Module):
    """
    ControlNet implementation for conditional diffusion model guidance.
    
    This module implements the ControlNet architecture from "Adding Conditional Control
    to Text-to-Image Diffusion Models" (Zhang et al., 2023). It creates a trainable
    copy of the encoder structure that processes conditioning inputs and generates
    residuals to guide the main UNet's generation process.
    
    Architecture:
    - Input: Noisy sample (1 channel) + conditions (2 channels concatenated)
    - Encoder: 4 downsampling blocks matching main UNet structure
      * DownBlock2D → DownBlock2D → AttnDownBlock2D → DownBlock2D
      * Each block has 2 ResNet layers (layersPerBlock=2)
    - Mid block: UNetMidBlock2D with self-attention
    - Zero convolutions: Optional 1x1 convs after each ResNet layer output
    
    The zero convolutions (when enabled) are initialized to zero, allowing the
    ControlNet to start as an identity function and gradually learn to add guidance.
    
    Parameters
    ----------
    useEncoderConvs : bool, default=True
        Whether to use zero-initialized 1x1 convolutions after encoder ResNet layers
        When True: Adds trainable zero convs for gradual learning
        When False: Directly uses encoder outputs as residuals
    
    Attributes
    ----------
    xConvIn : nn.Conv2d
        Initial convolution for noisy sample input (1→128 channels)
    conditionConvIn : nn.Conv2d
        Initial convolution for conditioning inputs (2→128 channels)
    encoderBlocks : nn.ModuleList
        List of 4 encoder blocks (DownBlock2D variants)
    encoderConvs : nn.ModuleList
        List of 8 zero-initialized 1x1 convolutions (2 per block, one per ResNet layer)
    midBlock : UNetMidBlock2D
        Middle block with self-attention at the bottleneck
    midBlockConv : nn.Conv2d
        Zero-initialized 1x1 convolution for mid block output
    timeProj : Timesteps
        Sinusoidal timestep positional encoding
    timeEmbedding : TimestepEmbedding
        MLP to embed timesteps into time embedding space
    scribbleClassEmbedding : nn.Embedding
        Embedding layer for scribble class type (foreground/background)
    """

    def __init__(
        self,
        useEncoderConvs: bool = True,
        blockOutChannels: tuple = (128, 128, 256, 256),
        layersPerBlock: int = 2,
        normNumGroups: int = 32,
        imageChannels: int = 1,
    ):
        """
        Initialize ControlNet with encoder structure and optional zero convolutions.

        Step 0 - Initialize input convolutions for sample and conditions
        Step 1 - Build encoder blocks matching main UNet architecture
        Step 2 - Initialize mid block with attention
        Step 3 - Initialize time and scribble class embedding layers

        Parameters
        ----------
        useEncoderConvs : bool, default=True
            Whether to use zero-initialized 1x1 convolutions after encoder ResNet layers.
        blockOutChannels : tuple, default=(128, 128, 256, 256)
            Channel count per encoder block. MUST match the main UNet, as the residuals
            this module produces are added element-wise onto the UNet encoder outputs.
        layersPerBlock : int, default=2
            ResNet layers per encoder block. MUST match the main UNet.
        normNumGroups : int, default=32
            GroupNorm groups for ResNet/attention norms. MUST match the main UNet and
            divide every value in blockOutChannels.
        imageChannels : int, default=1
            Channels of the noisy sample x. MUST match the main UNet's in_channels.
        """
        super().__init__()

        # Architecture configuration (kept in lock-step with the main UNet)
        blockOutChannels = tuple(blockOutChannels)  # Channel progression through encoder
        timeEmbedDim = blockOutChannels[0] * 4   # Time embedding dimension
        encoderBlockTypes = (
            "DownBlock2D",       # Block 1: Basic downsampling
            "DownBlock2D",       # Block 2: Basic downsampling
            "AttnDownBlock2D",   # Block 3: Downsampling with self-attention
            "DownBlock2D",       # Block 4: Basic downsampling
        )
        inChannels = 2               # Conditioning input channels (gt, prediction)
        inXChannels = imageChannels  # Noisy sample input channels

        self.useEncoderConvs = useEncoderConvs

        # Step 0 - Init convolutions
        self.xConvIn = nn.Conv2d(inXChannels, blockOutChannels[0], kernel_size=3, stride=1, padding=(1, 1))
        self.conditionConvIn = nn.Conv2d(inChannels, blockOutChannels[0], kernel_size=3, stride=1, padding=(1, 1))

        # Step 0.1 - Zero conv for the conv_in-level residual (the first skip connection).
        # Without it the raw (randomly initialised) input features would be injected into
        # the UNet from step 0, defeating the zero-init property of the ControlNet.
        self.convInZeroConv = None
        if self.useEncoderConvs:
            self.convInZeroConv = nn.Conv2d(blockOutChannels[0], blockOutChannels[0], kernel_size=1, stride=1, padding=0)
            self.setWeightsToZero(self.convInZeroConv)

        self.encoderBlocks = nn.ModuleList([])
        self.encoderConvs = nn.ModuleList([])
        
        # Step 1 - Init encoder blocks
        outputChannel = blockOutChannels[0]
        for i, encoderBlockType in enumerate(encoderBlockTypes):
            inputChannel = outputChannel
            outputChannel = blockOutChannels[i]
            isFinalBlock = i == len(blockOutChannels) - 1

            encoderBlock = get_down_block(
                encoderBlockType,
                num_layers=layersPerBlock,
                in_channels=inputChannel,
                out_channels=outputChannel,
                temb_channels=timeEmbedDim,
                add_downsample=not isFinalBlock,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=normNumGroups,
                attention_head_dim=outputChannel,
                downsample_padding=1,
                resnet_time_scale_shift="default",
                downsample_type="conv",
                dropout=0.0,
            )
            self.encoderBlocks.append(encoderBlock)

            # Add optional zero-initialized convolutions
            # One conv per resnet layer within each encoder block
            # MT: Options: random, zero-initialized, nothing
            if self.useEncoderConvs:
                for layerIdx in range(layersPerBlock):
                    conv = nn.Conv2d(outputChannel, outputChannel, kernel_size=1, stride=1, padding=0)
                    self.setWeightsToZero(conv)
                    self.encoderConvs.append(conv)
                # add one more as the downsample output is added as an extra residual
                if not isFinalBlock:
                    conv = nn.Conv2d(outputChannel, outputChannel, kernel_size=1, stride=1, padding=0)
                    self.setWeightsToZero(conv)
                    self.encoderConvs.append(conv)

        # Step 2 - Initialize mid block at bottleneck
        self.midBlock = UNetMidBlock2D(
            in_channels=blockOutChannels[-1],        # 256 channels
            temb_channels=blockOutChannels[0] * 4,   # 512 time embedding channels
            dropout=0.0,                             # No dropout
            resnet_eps=1e-5,                         # GroupNorm epsilon
            resnet_act_fn="silu",                    # SiLU activation
            output_scale_factor=1,                   # No output scaling
            resnet_time_scale_shift="default",       # AdaGN time conditioning
            attention_head_dim=8,                    # 8-dimensional attention heads
            resnet_groups=normNumGroups,             # GroupNorm groups (configurable)
            attn_groups=None,                        # Use default attention groups
            add_attention=True,                      # Include self-attention
        )
        # Step 2.1 - Add zero-initialized conv for mid block output
        self.midBlockConv = nn.Conv2d(blockOutChannels[-1], blockOutChannels[-1], kernel_size=1, stride=1, padding=0)
        self.setWeightsToZero(self.midBlockConv)

        # Step 3 - Initialize timestep embedding layers
        freqShift = 0         # No frequency shift in sinusoidal encoding
        flipSinToCos = True   # Start with cosine instead of sine
        # Step 3.1 - Create sinusoidal positional encoding for timesteps
        self.timeProj = Timesteps(blockOutChannels[0], flipSinToCos, freqShift)
        timestepInputDim = blockOutChannels[0]  # 128 dimensions
        # Step 3.2 - Create MLP to project timesteps to embedding space
        self.timeEmbedding = TimestepEmbedding(timestepInputDim, timeEmbedDim)  # 128→512
        
        # Step 3.3 - Create scribble class type embedding (foreground vs background)
        # Outputs 512-d directly to match tEmb — added straight in, no MLP reuse.
        # normal_(std=0.02) ensures the two class rows are distinguishable from epoch 0.
        self.scribbleClassEmbedding = nn.Embedding(
            num_embeddings=2,       # 0=background (scribbles on FP), 1=foreground (scribbles on GT)
            embedding_dim=timeEmbedDim  # 512 — matches tEmb dimension directly
        )
        nn.init.normal_(self.scribbleClassEmbedding.weight, std=0.02)

    # ======= HELPER FUNCTIONS ========
    
    def setWeightsToZero(self, module):
        """
        Initialize all parameters of a module to zero.
        
        This is crucial for ControlNet training: zero-initialized convolutions
        ensure the ControlNet starts as an identity function (adding zero guidance)
        and gradually learns to provide meaningful conditioning signals.
        
        Parameters
        ----------
        module : nn.Module
            PyTorch module whose parameters will be zeroed
        
        Returns
        -------
        nn.Module
            The same module with zeroed parameters (for chaining)
        """
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    # ======= FORWARD METHODS ========
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[list[torch.Tensor]] = None,
        scribbleClass: Optional[torch.Tensor] = None,
    ):
        """
        Process noisy sample and conditions to generate control residuals.
        
        This method implements the ControlNet forward pass:
        1. Embed timestep and scribble class information
        2. Combine noisy sample with conditioning via initial convolutions
        3. Process through encoder blocks, extracting residuals per ResNet layer
        4. Apply zero convolutions to residuals (if enabled)
        5. Process through mid block
        6. Return encoder and mid block residuals for main UNet injection
        
        The residuals are added to corresponding locations in the main UNet to
        provide conditioning guidance throughout the generation process.
        
        Parameters
        ----------
        x : torch.Tensor
            Noisy sample tensor, shape (batch, 1, height, width)
            The same noisy input being processed by the main UNet
        t : torch.Tensor
            Timestep values, shape (batch,) or scalar
            Current diffusion timestep for time conditioning
        conditions : list[torch.Tensor], optional
            List of conditioning tensors (e.g., scribbles, masks)
            Each tensor has shape (batch, channels, height, width)
            Will be concatenated along channel dimension
        scribbleClass : torch.Tensor, optional
            Scribble class type indices, shape (batch,)
            0=background (scribbles on FP), 1=foreground (scribbles on GT)
            Used to condition residual generation for specific scribble type
        
        Returns
        -------
        tuple
            (controlnetEncoderResiduals, midBlockResidual)
            - controlnetEncoderResiduals: tuple of 8 tensors (one per ResNet layer)
              Each tensor has shape (batch, channels, height, width)
            - midBlockResidual: tensor (batch, 256, height, width)
              Mid block output for bottleneck injection
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
        tProj = self.timeProj(timesteps).to(dtype=x.dtype)
        tEmb = self.timeEmbedding(tProj)
        
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
        # Step 2.1 - Convert conditions to tensor
        conditionsTensor = torch.cat(conditions, dim=1) if conditions is not None else None

        # Step 2.2 - Apply convolutions and combine
        h = self.xConvIn(x)
        if conditionsTensor is not None:
            h = h + self.conditionConvIn(conditionsTensor)

        originalH = h  # conv_in-level features, become the first skip residual

        # Step 3 - Encoder blocks processing
        # Step 3.1 - Process through each encoder block and flatten residuals
        # Each encoder block returns (h, resHiddenStates) where:
        #   - h: updated hidden states for next block
        #   - resHiddenStates: tuple of residuals (one per ResNet layer)
        # We flatten the nested structure: 4 blocks × 2 layers = 8 total residuals
        noiseEncoderResiduals: list[torch.Tensor] = []
        for encoderBlock in self.encoderBlocks:
            h, resHiddenStates = encoderBlock(hidden_states=h, temb=tEmb)
            # Flatten: extract each ResNet layer's output individually
            noiseEncoderResiduals += list(resHiddenStates)
        # Step 4 - Apply zero convolutions to residuals
        # Step 4.1 - Process each residual through its zero-initialized conv
        # Zero convolutions enable gradual training: starting from identity (zero output)
        # and slowly learning to provide meaningful guidance signals
        # Referenced as "zero convolutions" in the ControlNet paper
        controlnetEncoderResiduals = ()
        if self.useEncoderConvs:
            for encoderBlockResidual, controlnetConv in zip(noiseEncoderResiduals, self.encoderConvs):
                encoderBlockResidual = controlnetConv(encoderBlockResidual)
                controlnetEncoderResiduals += (encoderBlockResidual,)
        else:
            # Directly use residuals without zero convs
            controlnetEncoderResiduals = tuple(noiseEncoderResiduals)

        # Step 5 - Mid block processing
        # Step 5.1 - Process through mid block with attention at bottleneck
        midBlockResSample = self.midBlock(hidden_states=h, temb=tEmb)
        # Step 5.2 - Zero conv so the bottleneck residual starts at exactly zero
        midBlockResSample = self.midBlockConv(midBlockResSample)

        # Step 6 - Return residuals for main UNet injection
        # Step 6.1 - Prepend the conv_in-level residual (through its zero conv when enabled)
        if self.convInZeroConv is not None:
            originalH = self.convInZeroConv(originalH)
        controlnetEncoderResiduals = (originalH, ) + controlnetEncoderResiduals
        return controlnetEncoderResiduals, midBlockResSample
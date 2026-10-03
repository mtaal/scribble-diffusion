######################################################################
# VARIANCE SCHEDULE - Shared by training, sampling and ONNX inference
######################################################################
"""
Variance schedule for the DDPM forward/reverse process.

Kept torch-only (no diffusers/model imports) so that the ONNX inference
path can import it without pulling in the training stack.

Indexing convention
-------------------
Both schedulers return tensors of length ``steps + 1`` indexed by timestep
``t`` in ``[0, steps]``:

- ``alphaBar[0] == 1`` and ``beta[0] == 0`` (no noise at t=0).
- Training samples ``t`` uniformly from ``[1, steps]`` so that every sample
  contains noise for the network to predict.
- Reverse sampling loops ``t = steps, ..., 1`` and reads ``alphaBar[t]``
  and ``beta[t]`` with the same index.
"""

######################################################################
# IMPORTS
######################################################################

# Pip/Third-party installs
import torch

######################################################################
# SCHEDULE
######################################################################

def computeVarianceSchedule(
    steps: int,
    schedulerType: str = "cosine",
    beta1: float = 1e-4,
    betaT: float = 0.2,
    s: float = 0.008,
    maxBeta: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the variance schedule for the diffusion process.

    Parameters
    ----------
    steps : int
        Number of diffusion steps T
    schedulerType : str, default="cosine"
        Type of scheduler: "linear" or "cosine"
    beta1 : float, default=1e-4
        Beta at time 1 (linear scheduler only)
    betaT : float, default=0.2
        Beta at time T (linear scheduler only)
    s : float, default=0.008
        Smoothing offset for the cosine scheduler
    maxBeta : float, default=0.999
        Maximum beta value (cosine scheduler only)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (alphaBar, beta), both of length ``steps + 1``, indexed by t in [0, steps]

    Notes
    -----
    Refer to DDPM (https://arxiv.org/pdf/2006.11239) and
    Improved DDPM (https://arxiv.org/pdf/2102.09672).
    """
    if schedulerType == "linear":
        # Step 1 - beta_1..beta_T, with a zero-noise entry prepended for t=0
        beta = torch.cat([torch.zeros(1), torch.linspace(beta1, betaT, steps)])
        alphaBar = torch.cumprod(1 - beta, 0)
        return alphaBar, beta

    elif schedulerType == "cosine":
        # Step 1 - alphaBar_t = f(t)/f(0), t in [0, steps]
        times = torch.arange(0, steps + 1, 1)
        f = torch.cos((times / steps + s) / (1 + s) * torch.pi / 2) ** 2
        alphaBar = f / f[0]

        # Step 2 - beta_t = 1 - alphaBar_t / alphaBar_{t-1}, clipped at maxBeta
        alphaBarPrev = torch.cat([torch.ones(1), alphaBar[:-1]])
        beta = torch.clip(1 - alphaBar / alphaBarPrev, 0.0, maxBeta)
        return alphaBar, beta

    else:
        raise ValueError(
            f"Unknown scheduler type {schedulerType}. Supported schedulers: linear, cosine."
        )

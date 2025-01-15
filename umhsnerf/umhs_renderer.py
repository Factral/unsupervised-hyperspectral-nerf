import torch
from torch import nn
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor
from nerfstudio.model_components.renderers import SemanticRenderer

class SpectralRenderer(SemanticRenderer):
    """Calculate spectral along the ray."""

    @classmethod
    def forward(
        cls,
        spectral: Float[Tensor, "*bs num_samples num_classes"],
        weights: Float[Tensor, "*bs num_samples 1"]
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate spectral along the ray."""
        return torch.sum(weights * spectral, dim=-2)



def get_weights_spectral(deltas, densities):
    """Return weights based on predicted densities

    Args:
        densities: Predicted densities for samples along ray

    Returns:
        Weights for each sample
    """

    delta_density = deltas * densities
    alphas = 1 - torch.exp(-delta_density)

    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [torch.zeros((*transmittance.shape[:1], 1, transmittance.shape[-1]), device=densities.device), transmittance], dim=-2
    )
    transmittance = torch.exp(-transmittance)

    weights = alphas * transmittance
    weights = torch.nan_to_num(weights)

    return weights
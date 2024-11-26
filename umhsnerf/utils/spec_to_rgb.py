
import torch


def spec_to_rgb(spec, cmf):
    """
    Converts spectral data (rays) to RGB using a color matching function (CMF) and gamma correction.

    Args:
        spec (torch.Tensor): Spectral data of shape [R, S, C], where R is the number of rays, 
                             S is the spatial dimension, and C is the spectral dimension.
        cmf (torch.Tensor): Color Matching Function of shape [C, 3].

    Returns:
        torch.Tensor: RGB data of shape [R, S, 3].
    """
    # Check shapes
    if spec.shape[-1] != cmf.shape[0]:
        raise ValueError("The last dimension of 'spec' must match the first dimension of 'cmf'.")
    
    # Compute XYZ by collapsing the spectral dimension
    # spec: [R, S, C], cmf: [C, 3]
    # Result: [R, S, 3]
    if spec.ndim == 2:
        xyz = torch.einsum('sc,cm->sm', spec, cmf)
    else:
        xyz = torch.einsum('rsc,cm->rsm', spec, cmf)
    
    # Convert XYZ to sRGB
    srgb_matrix = torch.tensor([
        [3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [0.0557, -0.2040,  1.0570]
    ], dtype=spec.dtype, device=spec.device)
    
    # Apply the sRGB transformation
    if xyz.ndim == 2:
        srgb = torch.einsum('ij,smj->sm', srgb_matrix, xyz.unsqueeze(-1)).squeeze(-1)
    else:
        srgb = torch.einsum('ij,rsmj->rsm', srgb_matrix, xyz.unsqueeze(-1)).squeeze(-1)

    # Gamma correction
    def gamma_correction(x):
        threshold = 0.0031308
        return torch.where(x <= threshold, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)
    
    rgb = gamma_correction(srgb)
    # normalize between 0 and 1 with min max normalization
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    return rgb
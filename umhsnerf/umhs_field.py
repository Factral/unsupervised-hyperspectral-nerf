"""
UMHS Field with semantic-guided spectral unmixing.
"""

from typing import Literal, Optional, Any, Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames

from umhsnerf.utils.spec_to_rgb import ColourSystem
from umhsnerf.seg_field import SemanticField

class UMHSField(NerfactoField):
    """UMHS Field with semantic-guided spectral unmixing."""

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "torch",
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        wavelengths: int = 21,
        method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb",
        num_classes: int = 5,
        feature_dim: int = 256,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images,implementation=implementation, **kwargs)

        self.method = method
        self.num_classes = num_classes
        self.wavelengths = wavelengths
        self.feature_dim = feature_dim

        if self.method == "spectral" or self.method == "rgb+spectral":
            # Semantic field for abundance prediction
            self.semantic_field = SemanticField(
                position_encoding=self.position_encoding,
                num_classes=num_classes,
                feature_dim=feature_dim,
                num_heads=num_heads,
                dir_embedding_dim=self.geo_feat_dim,
                hidden_dim=hidden_dim_color,
                implementation=implementation,
                wavelengths=wavelengths
            )

            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=wavelengths*num_classes,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )

        self.converter = ColourSystem()

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[Any, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Get appearance embedding
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # Handle transients and other outputs from parent class
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # Spectral prediction using semantic unmixing
        if "spectral" in self.method:

            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                ]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)] 
                    if embedded_appearance is not None else []
                ),
                dim=-1,
            ) # direction, density features, appeareance embeddings

            features = self.mlp_head(h).view(*outputs_shape, self.wavelengths, self.num_classes)
            
            abundances = self.semantic_field(
                self._sample_locations,
                density_embedding=density_embedding.view(-1, self.geo_feat_dim),
            )
            abundances = abundances.view(*ray_samples.frustums.directions.shape[:-1], -1)

            endmembers = self.semantic_field.endmembers  # (num_classes, 1, feature_dim)
            endmembers = endmembers.squeeze().unsqueeze(0).unsqueeze(0)
            endmembers = endmembers.expand(abundances.shape[0], abundances.shape[1], -1, -1).transpose(2,3)

            endmember_spectra = features * endmembers #.permute(0, 1, 3, 2)  # (num_classes, ..., wavelengths)

            endmember_spectra = torch.clamp(endmember_spectra, 0, 1)
            spec = (endmember_spectra  @ abundances.unsqueeze(-1)).squeeze() # (..., wavelengths)

            outputs["spectral"] = spec.to(directions)
            outputs["abundances"] = abundances
            
            if self.method == "rgb+spectral":
                rgb = self.converter(spec).to(directions)
                outputs[FieldHeadNames.RGB] = rgb


        elif self.method == "rgb":
            # Original RGB prediction
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                ]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)] 
                    if embedded_appearance is not None else []
                ),
                dim=-1,
            )
            rgb = self.mlp_head(h).view(*outputs_shape, 3).to(directions)
            outputs[FieldHeadNames.RGB] = rgb

        return outputs
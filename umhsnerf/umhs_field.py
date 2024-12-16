"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional, Any, Dict

from torch import Tensor

from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions

from torch import Tensor, nn

import torch

from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)


from umhsnerf.utils.spec_to_rgb import ColourSystem

class UMHSField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        wavelengths: int = 21, # by default for the nespof dataset
        method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb",
        cmf: Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images, **kwargs)

        self.method = method
        if self.method == "spectral" or self.method == "rgb+spectral":
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=wavelengths,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
        self.cmf = cmf
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

        # appearance
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

        # transients
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

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )

        if "spectral" in self.method:
            spec = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
            
            outputs.update({"spectral": spec})
            
            if self.method == "rgb+spectral":
                rgb = self.converter(spec).to(directions)

        if self.method == "rgb":
            rgb = self.mlp_head(h).view(*outputs_shape, 3).to(directions)

        if "rgb" in self.method:
            outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

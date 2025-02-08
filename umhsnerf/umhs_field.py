"""
UMHS Field with semantic-guided spectral unmixing.
"""

from typing import Literal, Optional, Any, Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding

from umhsnerf.utils.spec_to_rgb import ColourSystem
import numpy as np
import tinycudann as tcnn


class UMHSField(NerfactoField):
    """UMHS Field with semantic-guided spectral unmixing."""

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        wavelengths: int = 128,
        method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb",
        num_classes: int = 7,
        feature_dim: int = 256,
        temperature: float = 0.5,
        converter: ColourSystem = None,
        pred_dino: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images,implementation=implementation, **kwargs)

        self.method = method
        self.num_classes = num_classes
        self.wavelengths = wavelengths
        self.feature_dim = feature_dim

        if self.method == "spectral" or self.method == "rgb+spectral":
            # Semantic field for abundance prediction
            input_dim = self.position_encoding.get_out_dim() + self.geo_feat_dim
            self.feature_mlp = MLP(
                in_dim=input_dim,
                num_layers=3,
                layer_width=hidden_dim_color,
                out_dim=num_classes,
                activation=nn.ReLU(),
                out_activation=None, # tanh ?
                implementation=implementation,
            )

            if self.training:
                endmembers = np.load("vca.npy")
                self.endmembers = nn.Parameter(torch.tensor(endmembers, dtype=torch.float32), requires_grad=True)
                #self.endmembers = nn.Parameter(torch.randn(self.num_classes, self.wavelengths), requires_grad=True)
            else:
                # will be loaded from the checkpoint
                self.endmembers = nn.Parameter(torch.randn(self.num_classes, self.wavelengths), requires_grad=True)


            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=num_classes,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )



            self.converter = converter
            self.temperature = temperature
            self.pred_dino = pred_dino

            if pred_dino:
                grid_layers = (12,12)
                grid_sizes = (19, 19)
                grid_resolutions = ((16, 128), (128, 512))

                self.encs = torch.nn.ModuleList(
                    [
                        UMHSField._get_encoding(
                            grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                        )
                        for i in range(len(grid_layers))
                    ]
                )
                tot_out_dims = sum([e.n_output_dims for e in self.encs])
            
                self.dino_mlp = MLP(
                    in_dim=self.geo_feat_dim,# + self.direction_encoding.get_out_dim(),
                    num_layers=2,
                    layer_width=256,
                    out_dim=128, # dinov2 dim featup_jbu
                    activation=nn.ReLU(),
                    out_activation=None,
                    implementation=implementation,
                )

                self.direction_encoding2 = SHEncoding(
                            levels=4,
                            implementation=implementation,
                        )
                

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc
            
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
            scalar = self.mlp_head(h).view(*outputs_shape, -1, self.num_classes)

            positions =  ray_samples.frustums.get_positions()
            positions_flat = self.position_encoding(positions.view(-1, 3))
            positions_flat = positions_flat.view(-1, density_embedding.size(1),  self.position_encoding.get_out_dim() )

            features_input = torch.cat([positions_flat, density_embedding], dim=-1) # positions, density

            size = features_input.size()
            features_input = features_input.view(-1, features_input.size(-1))
            features = self.feature_mlp(features_input)

            logits = features.view(*size[:-1], -1)

            abundances = F.softmax(logits / self.temperature, dim=-1)

            endmembers = self.endmembers.unsqueeze(0).unsqueeze(0)
    
            endmembers = endmembers.expand(abundances.shape[0], abundances.shape[1], -1, -1).transpose(2,3)

            scalar = F.sigmoid(scalar)
            #scalar = F.relu(scalar)

            adapted_endmembers = scalar * endmembers  # (B, ray_sample, wavelengths, num_classes)
            spec = (adapted_endmembers  @ abundances.unsqueeze(-1)).squeeze() # linear mixing model spec = EA

            outputs["spectral"] = spec.to(directions)
            outputs["abundances"] = abundances.to(directions)

            if self.pred_dino:
                #positions = self.spatial_distortion(positions).detach()
                #positions = (positions + 2.0) / 4.0
                # First concatenate the list of encodings
                #xs = [e(positions.view(-1, 3)) for e in self.encs]
                #x = torch.concat(xs, dim=-1)

                #d2 = self.direction_encoding2(directions_flat)
                xs = density_embedding.view(-1, self.geo_feat_dim).detach()
                #x = torch.cat([d2, xs], dim=-1)
                pred = self.dino_mlp(xs).view(*outputs_shape, 128).to(directions)
                
                outputs["dino"] = pred


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


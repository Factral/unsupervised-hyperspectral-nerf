"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

import torch
from dataclasses import dataclass, field
from typing import Type
from typing import Dict, List, Literal, Tuple, Type
from collections import defaultdict
import cv2
from tqdm import tqdm
import random

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

from umhsnerf.umhs_field import UMHSField
from umhsnerf.data.utils.dino_extractor import ViTExtractor


@dataclass
class UMHSConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: UMHSModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    patch_size: int = 96



class UMHSModel(NerfactoModel):
    """Template Model."""

    config: UMHSConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.field = UMHSField(  
                self.scene_box.aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                hidden_dim_transient=self.config.hidden_dim_transient,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.predict_normals,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                appearance_embedding_dim=appearance_embedding_dim,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation)

        self.renderer_spectral = SemanticRenderer()
        self.rgb_loss = MSELoss()
        self.spectral_loss = MSELoss()

    
    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        """
        try: 
            patch_rgb = rgb.view(-1, self.config.patch_size, self.config.patch_size, 3)
            with torch.no_grad():

                # pass patch to dino model
                extractor = ViTExtractor("dino_vits8",8)
                patch_rgb = patch_rgb.permute(0, 3, 1, 2)
                preproc_image_lst = extractor.preprocess(patch_rgb, 500)[0].to(self.device)

                dino_embeds = []
                for image in tqdm(preproc_image_lst, desc="dino", total=len(patch_rgb), leave=False):
                    print("image shape", image.shape)
                    with torch.no_grad():
                        descriptors = extractor.extract_descriptors(
                            image.unsqueeze(0),
                            [11],
                            "key",
                            False,
                        )
                    descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
                    print("dinooooooo", descriptors.shape)
                    dino_embeds.append(descriptors.cpu().detach())


                    print("dino feature", descriptors[0].shape)
                    print("min", descriptors[0].min())
                    print("max", descriptors[0].max())

                    single_feature = descriptors[:,:,0]
                    single_feature = (single_feature - single_feature.min()) / (single_feature.max() - single_feature.min())
                    print(single_feature.detach().cpu().numpy().shape)
                    single_feature = single_feature.unsqueeze(-1)
                    single_feature = torch.clamp(single_feature, 0, 1)
                    single_feature = single_feature * 255

                    random_number = random.randint(0, 10000) 

                    cv2.imwrite(f"features/dino_feature_{random_number}.png", single_feature.detach().cpu().float().numpy())

                    patch = patch_rgb[0].permute(1, 2, 0)
                    patch = torch.clamp(patch, 0, 1)
                    patch = patch * 255
                    cv2.imwrite(f"features/patch_{random_number}.png", patch.detach().cpu().numpy()[::-1])

        except Exception as e:
            print(e)
            pass
        """
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)

        with torch.no_grad():
            spectral = self.renderer_spectral(
                semantics=field_outputs["spectral"], weights=weights
            )

        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)


        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "spectral": spectral,
        }

        # update output with a for of each weavelength of spectral
        for i in range(spectral.shape[-1]):
            outputs[f"wv_{i}"] = spectral[..., i]


        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])


        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}


        image = batch["image"].to(self.device)
        gt_spectral = batch["hs_image"].to(self.device)
        
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        loss_dict["spectral_loss"] = self.spectral_loss(gt_spectral, outputs["spectral"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        
        return loss_dict


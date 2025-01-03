"""
Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Type
from typing import Dict, List, Literal, Tuple, Type, Union, Optional
from collections import defaultdict
import cv2
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colormaps
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation


from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

import wandb

from umhsnerf.umhs_field import UMHSField
from umhsnerf.data.utils.dino_extractor import ViTExtractor
from umhsnerf.utils.metrics import mse2psnr
from umhsnerf.utils.spec_to_rgb import ColourSystem
from umhsnerf.utils.hooks import nan_hook
from umhsnerf.umhs_renderer import SpectralRenderer, get_weights_spectral

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
    implementation: Literal["tcnn", "torch"] = "torch"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

    # custom configs
    method: Literal["rgb", "spectral", "rgb+spectral"] = "rgb"


class UMHSModel(NerfactoModel):
    """UMHS Model."""

    config: UMHSConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.class_colors = {
            0: torch.tensor([0., 0., 0.]),      #  - negro
            1: torch.tensor([0.9, 0., 0.]),     #  - rojo
            2: torch.tensor([0., 0.9, 0.]),     #  - verde
            3: torch.tensor([0., 0., 0.9]),     #  - azul
            4: torch.tensor([0.9, 0.9, 0.]),    #  - amarillo
            5: torch.tensor([0., 0.9, 0.9]),    #  - cian
            6: torch.tensor([0.9, 0., 0.9])     #  - magenta
            }


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
                implementation=self.config.implementation,
                method=self.config.method,
                )

        self.rgb_loss = MSELoss()

        if 'spectral' in self.config.method:
            # reuse the renderer for spectral
            # definition: https://github.com/nerfstudio-project/nerfstudio/blob/758ea1918e082aa44776009d8e755c2f3a88d2ee/nerfstudio/model_components/renderers.py#L408
            self.renderer_spectral = SpectralRenderer() 
            self.spectral_loss = MSELoss()
            self.converter = ColourSystem()

    def label_to_rgb(self, labels):
        device = labels.device
        colors = torch.stack(list(self.class_colors.values())).to(device)
        return colors[labels.long().squeeze(-1)]

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        
        ray_samples: RaySamples
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        

        weights = get_weights_spectral(ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY].repeat(1,1,21))
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)

        accumulation = self.renderer_accumulation(weights=weights[:,:,0].unsqueeze(-1))

        outputs = {
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.method == "rgb":
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            outputs["rgb"] = rgb

        if "spectral" in self.config.method:
            spectral = self.renderer_spectral(spectral=field_outputs["spectral"], weights=weights)

            outputs["spectral"] = spectral
            for i in range(spectral.shape[-1]):
                outputs[f"wv_{i}"] = spectral[..., i]

            #pseudorgb
            if self.config.method == "spectral":
                with torch.no_grad():
                    outputs["rgb"] = self.converter(spectral)
            else:
                outputs["rgb"] = self.converter(spectral)

            with torch.no_grad():
                abundaces = self.renderer_spectral(
                    spectral=field_outputs["abundances"], weights=weights[:,:,:7]
                )

                outputs["abundances"] = abundaces
                outputs["abundaces_seg"] = self.label_to_rgb(torch.argmax(abundaces, dim=-1))
                for i in range(field_outputs["abundances"].shape[-1]):
                    outputs[f"abundances_{i}"] = abundaces[:,i]


        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        image = batch["image"].to(self.device)
        gt_spectral = batch["hs_image"].to(self.device)

        if "rgb" in self.config.method:
            pred_rgb, gt_rgb = outputs["rgb"], image

        if self.config.method == "rgb":
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        elif self.config.method == "spectral":
            loss_dict["spectral_loss"] = self.spectral_loss(outputs["spectral"], gt_spectral)
        elif self.config.method == "rgb+spectral":
            loss_dict["spectral_loss"] = self.spectral_loss(outputs["spectral"], gt_spectral)
            loss_dict["rgb_loss"] = self.rgb_loss(pred_rgb, gt_rgb)

        
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        
        return loss_dict


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA

        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        predicted_rgb = outputs["rgb"]

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        se_per_pixel = (gt_rgb - predicted_rgb) ** 2
        se_per_pixel = se_per_pixel.squeeze().permute(1, 2, 0).mean(dim=-1).unsqueeze(-1)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, "se_per_pixel": se_per_pixel}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    #related to https://github.com/NVlabs/tiny-cuda-nn/issues/377
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:   
            camera: generates raybundle
        """
        with torch.autocast(device_type="cuda", enabled=True):
            return self.get_outputs_for_camera_ray_bundle(
                camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
            )


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            if self.config.method != "rgb":
                def clamp_endmembers(step):
                    with torch.no_grad():
                        self.field.endmembers[:] = self.field.endmembers.clamp(0, 1)

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.field.endmembers.cpu().detach().T.numpy())
                    wandb.log({"endmembers": wandb.Image(fig)}, step=step)
                    plt.close(fig)

                # callback to clamp between 0 and 1 the endmember parameter
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=clamp_endmembers
                    )
                )

        return callbacks
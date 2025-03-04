"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from umhsnerf.data.umhs_datamanager import (
    UMHSDataManagerConfig, UMHSDataManager
)
from umhsnerf.umhs_model import UMHSConfig
from umhsnerf.umhs_pipeline import (
    UMHSPipelineConfig,
)
from umhsnerf.data.umhs_dataparser import UMHSDataParserConfig

from umhsnerf.data.utils.hs_dataloader import HyperspectralDataset

umhs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="umhsnerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=UMHSPipelineConfig(
            datamanager=UMHSDataManagerConfig(
                _target=UMHSDataManager[HyperspectralDataset],
                dataparser=UMHSDataParserConfig(),
                train_num_rays_per_batch=9216*4,
                eval_num_rays_per_batch=9216,
            ),
            model=UMHSConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.1
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        }, 
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="umhs method",
)

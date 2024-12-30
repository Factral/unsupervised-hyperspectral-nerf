"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast


import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler


from umhsnerf.data.umhs_datamanager import UMHSDataManagerConfig
from umhsnerf.umhs_model import UMHSModel, UMHSConfig

import torch


@dataclass
class UMHSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: UMHSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = UMHSDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = UMHSConfig()
    """specifies the model config"""
    check_nan: bool = False

# based on: https://github.com/nerfstudio-project/nerfstudio/blob/758ea1918e082aa44776009d8e755c2f3a88d2ee/nerfstudio/pipelines/base_pipeline.py#L212
class UMHSPipeline(VanillaPipeline):
    """UMHS Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: UMHSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, grad_scaler=grad_scaler)
        if self.config.check_nan:
            torch.autograd.set_detect_anomaly(True)


    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        #autocast is needed when using tcnn https://github.com/NVlabs/tiny-cuda-nn/issues/377
        with torch.autocast(device_type="cuda", enabled=True):
            self.eval()
            ray_bundle, batch = self.datamanager.next_eval(step)
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
            self.train()
        return model_outputs, loss_dict, metrics_dict


    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        with torch.autocast(device_type="cuda", enabled=True):
            self.eval()
            camera, batch = self.datamanager.next_eval_image(step)
            outputs = self.model.get_outputs_for_camera(camera)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
            self.train()
        return metrics_dict, images_dict
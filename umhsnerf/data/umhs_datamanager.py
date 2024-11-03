"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Generic
from typing_extensions import TypeVar

import torch

from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion

from umhsnerf.data.umhs_dataparser import UMHSDataParserConfig

from nerfstudio.utils.rich_utils import CONSOLE
from functools import cached_property


CustomDataParserUnion = Union[UMHSDataParserConfig, AnnotatedDataParserUnion]

@dataclass
class UMHSDataManagerConfig(VanillaDataManagerConfig):
    """UMHS DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: UMHSDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=UMHSDataParserConfig)

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class UMHSDataManager(VanillaDataManager, Generic[TDataset]):
    """UMHS DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: UMHSDataManagerConfig

    def __init__(
        self,
        config: UMHSDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)


        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch


    @cached_property # this is necessary to add the hs rays to the ray bundle
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[UMHSDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is UMHSDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is UMHSDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is UMHSDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default
        
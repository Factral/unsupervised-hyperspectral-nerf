"""
Hyperspectral dataset.
"""

from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

import torch
import numpy as np

from umhsnerf.data.utils.vca import vca
import os

class HyperspectralDataset(InputDataset):
    """Dataset that returns hyperspectral images.
    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device


    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        assert scale_factor == 1, 'Scale factors not yet supported for hyperspectral'
        assert (
            "hs_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["hs_filenames"] is not None
        )
        self.hs_filenames = self.metadata["hs_filenames"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.hs_filenames[data["image_idx"]]

        hs_image = np.load(filepath)
        hs_image = torch.tensor(hs_image).float().clamp(0, 1)
        
        if not os.path.exists("vca.npy"):
            pass

        print("hola")

        return {"hs_image": hs_image}

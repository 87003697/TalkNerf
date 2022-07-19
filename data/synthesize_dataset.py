import os
import numpy as np
import imageio
import json
import torch.utils.data as data
from typing import Optional, Any, List, Dict, Tuple
from .base_dataset import ListDataset 
from pytorch3d.renderer import PerspectiveCameras
import torch
from PIL import Image
import logging
import requests

log = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = '/home/mazhiyuan/code/talknerf/datasets/Synthesize'

DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"

ALL_DATASETS = ("lego", "fern", "pt3logo")

log = logging.getLogger(__name__)

class SynthesizeDataset:
    
    train: Optional[List[data.Dataset]] = None
    val: Optional[List[data.Dataset]] = None
    test: Optional[List[data.Dataset]] = None
    meta: Dict    
    
    def __init__(self,
                 dataset_name: str, # 'lego | fern'
                 image_size: Tuple[int, int],
                 data_root: str = DEFAULT_DATA_ROOT, 
                 autodownload: bool = True,
                 preload_image: bool = True): # -> Tuple[Dataset, Dataset, Dataset] 
        """
        Obtains the training and validation dataset object for a dataset specified
        with the `dataset_name` argument.

        Arguments
        ---------
        dataset_name: str
            The name of the dataset to load.
        image_size: Tuple[int, int]
            A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: str
            The root folder at which the data is stored.
        autodownload: bool
            Auto-download the dataset files in case they are missing.
        preload_image: bool
            Preload_image to save the cost of loading images during training / validation
            For debugging, turning it off is efficient. For training, pelease do turn it on. 
        Returns
        -------
        train_dataset: List[data.Dataset]
            The training dataset object.
        val_dataset: List[data.Dataset]
            The validation dataset object.
        test_dataset: List[data.Dataset]
            The testing dataset object.
        """

        if dataset_name not in ALL_DATASETS:
            raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

        log.info("Loading dataset {}, image size={} ...".format(dataset_name, str(image_size)))

        cameras_path = os.path.join(data_root, dataset_name + ".pth")
        image_path = cameras_path.replace(".pth", ".png")

        if autodownload and any(not os.path.isfile(p) for p in (cameras_path, image_path)):
            # Automatically download the data files if missing.
            download_data((dataset_name,), data_root=data_root)

        train_data = torch.load(cameras_path)
        n_cameras = train_data["cameras"]["R"].shape[0]

        if preload_image:
            _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
            images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
            images = torch.stack(torch.chunk(images, n_cameras, dim=0))[..., :3]
            Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

            scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]
            if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
                raise ValueError(
                    "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
                )
            
            scale_factor = sum(scale_factors) * 0.5

            if scale_factor != 1.0:
                print(f"Rescaling dataset (factor={scale_factor})")
                images = torch.nn.functional.interpolate(
                    images.permute(0, 3, 1, 2),
                    size=tuple(image_size),
                    mode="bilinear",
                ).permute(0, 2, 3, 1)
                
        else:
            images = [image_path, ] * n_cameras
            
        cameras = [
            PerspectiveCameras(
                **{k: v[cami][None] for k, v in train_data["cameras"].items()}
            ).to("cpu")
            for cami in range(n_cameras)
        ]
        
        train_idx, val_idx, test_idx = train_data["split"]

        train_dataset, val_dataset, test_dataset = [
            ListDataset(
                [
                    {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                    for i in idx
                ]
            )
            for idx in [train_idx, val_idx, test_idx]
        ]

        log.info('Train dataset loaded with {} data'.format(len(train_dataset)))
        self.train = [train_dataset]
        log.info('Val Dataset loaded with {} data'.format(len(val_dataset)))
        self.val = [val_dataset]
        log.info('Test Dataset loaded with {} data'.format(len(test_dataset)))
        self.test = [test_dataset]

def download_data(
    dataset_names: Optional[List[str]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    url_root: str = DEFAULT_URL_ROOT,
): # -> None
    """
    Downloads the relevant dataset files.

    Arguments
    ---------
    dataset_names:  Optional[List[str]]
        A list of the names of datasets to download. 
        If `None`, downloads all available datasets.
    """

    if dataset_names is None:
        dataset_names = ALL_DATASETS

    os.makedirs(data_root, exist_ok=True)

    for dataset_name in dataset_names:
        cameras_file = dataset_name + ".pth"
        images_file = cameras_file.replace(".pth", ".png")
        license_file = cameras_file.replace(".pth", "_license.txt")

        for fl in (cameras_file, images_file, license_file):
            local_fl = os.path.join(data_root, fl)
            remote_fl = os.path.join(url_root, fl)

            log.info(f"Downloading dataset {dataset_name} from {remote_fl} to {local_fl}.")

            r = requests.get(remote_fl)
            with open(local_fl, "wb") as f:
                f.write(r.content)

import torch.utils.data as data
from PIL import Image
import numpy as np
from typing import List, Dict
import torch
import cv2
def image_load(data:Dict,): # -> Dict
    """
    If 'image' or 'img' in the data dict and it is not loaded, load it

    arguments
    ---------
    data: Dict
        one data point saved in the data.Dataset object
    """
    for image_key in ['image', 'img', 'parsing']:
        if image_key in data and type(data[image_key]) not in [torch.Tensor, np.array]:
            assert type(data[image_key]) == str, 'The image path {} seems not a image path'.format(data[image_key])
            data[image_key] = torch.FloatTensor(np.array(Image.open(data[image_key]))) / 255.0
            # data[image_key] = torch.FloatTensor(cv2.resize(cv2.imread(data[image_key]), (512, 512))) / 255.0
    return data

class ListDataset(data.Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List): # -> None
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(self,): #  -> int
        return len(self._entries)

    def __getitem__(self, index):
        return image_load(self._entries[index])

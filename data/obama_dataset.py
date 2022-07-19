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
import cv2

log = logging.getLogger(__name__)

class ObamaDataset:
    
    train: Optional[List[data.Dataset]] = None
    val: Optional[List[data.Dataset]] = None
    test: Optional[List[data.Dataset]] = None
    meta: Dict    
    
    def __init__(self, 
                 basedir:str,   
                 testskip:int=1, 
                 test_file:Optional[str]=None, 
                 aud_file:Optional[str]=None,
                 preload_image:bool=False,
                 image_size:Tuple[int,int]=(450, 450),
                 with_parsing:bool=False,
                 **kwargs): #  -> None
        """
        Collect data for Obama dataset, to reproduce results in AD-nerf
        
        Arguments
        ---------
        basedir : str
            The base directory where the whole dataset is saved.
        testskip : int
            Select certain percentage of data with fixed step. Only activated in the validation stage. 
        test_file : str
            The path to test annotations.
        audio_file : str
            The path to test audio_file.
        preload_image: bool
            Load all images to memotry before training
        image_size: Tuple[int,int]
            If change image size if it does not meach image_size
            
        Returns
        -------
        train_dataset: List[data.Dataset]
            The training dataset object.
        val_dataset: List[data.Dataset]
            The validation dataset object.
        test_dataset: List[data.Dataset]
            The testing dataset object.
        """
        if test_file is not None and aud_file is not None:
            self.load_test_data(basedir, testskip, image_size, test_file, aud_file)
        elif test_file is None and aud_file is None:
            self.load_train_data(basedir, testskip, image_size, preload_image, with_parsing = with_parsing)
        else:
            Exception

    def load_test_data(self,
                        basedir:str, 
                        testskip:int=1, 
                        image_size:Tuple[int,int]=(450, 450),
                        test_file:Optional[str]=None, 
                        aud_file:Optional[str]=None,): #  -> None
        """
        Collect TEST data for Obama dataset, to reproduce results in AD-nerf
        
        Arguments
        ---------
        same as self.__init__()
        
        Returns
        -------
        same as self.__init__()
        """     
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg')) # cv2.resize(cv2.imread(os.path.join(basedir, 'bc.jpg')), (512, 512))
        with open(test_file, 'r') as fp:
            meta = json.load(fp)

        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])
        
        data_list = []
        aud_features = np.load(aud_file)
        for idx, frame in enumerate(meta['frames'][::testskip]):
            data_frame = {}
            pose = torch.tensor(np.array(frame['transform_matrix']).astype(np.float32))
            data_frame['camera'] = PerspectiveCameras(
                focal_length=focal, #TODO: is that correct ?
                principal_point=torch.tensor([cx,cy])[None],
                R = pose[:3,:3][None],
                T = pose[:3,3][None])
            data_frame['audio'] = torch.tensor(
                np.array(
                    aud_features[
                        frame['frame_id']]).astype(np.float32))
            data_frame['sample_rect'] = None           
            data_frame['camera_idx'] = idx
            data_list.append(data_frame)
            
        log.info('Test dataset loaded with {} data'.format(len(data_list)))
        dataset = ListDataset(data_list)
        dataset.meta = {'bg_image': torch.FloatTensor(bc_img) / 255., }
        self.test = [dataset]
                
    def load_train_data(self,
                        basedir:str, 
                        testskip:int=1, 
                        image_size:Tuple[int,int] = (450, 450),
                        preload_image:bool=False,
                        with_parsing:bool=False): # -> List[data.Dataset]
        """
        Collect TRAINING data for Obama dataset, to reproduce results in AD-nerf
        
        Arguments
        ---------
        same as self.__init__()
        
        Returns
        -------
        same as self.__init__()
        """
        if preload_image:
            log.info('All images will be loaded before training')
        prev_nums = 0
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg')) #cv2.resize(cv2.imread(os.path.join(basedir, 'bc.jpg')), (512, 512)) #
        
        for split in ['train', 'val']:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
                meta = json.load(fp)

            H, W = bc_img.shape[0], bc_img.shape[1]
            focal, cx, cy = float(meta['focal_len']), float(
                meta['cx']), float(meta['cy'])
        
            data_list = []
            aud_features = np.load(os.path.join(basedir, 'aud.npy'))
            for idx, frame in enumerate(meta['frames'][::testskip]):
                data_frame = {}
                
                data_frame['image'] = os.path.join(basedir, 'head_imgs', str(frame['img_id']) + '.jpg')
                if preload_image:
                    data_frame['image'] = torch.FloatTensor(np.array(Image.open(data_frame['image']))) / 255.0
                
                if with_parsing:
                    data_frame['parsing'] = os.path.join(basedir, 'parsing', str(frame['img_id']) + '.png')
                    if preload_image:
                        data_frame['parsing'] = torch.FloatTensor(np.array(Image.open(data_frame['parsing']))) / 255.0
                    
                pose = torch.tensor(np.array(frame['transform_matrix']).astype(np.float32))
                data_frame['camera'] = PerspectiveCameras(
                    focal_length=focal, #TODO: is that correct ?
                    principal_point=torch.tensor([cx,cy])[None],
                    R = pose[:3,:3][None],
                    T = pose[:3,3][None])
                    #np.array(frame['transform_matrix']).astype(np.float32)
                
                data_frame['audio'] = torch.tensor(
                    aud_features[
                        min(
                            frame['aud_id'], 
                            aud_features.shape[0]-1)].astype(np.float32))
                
                data_frame['sample_rect'] = torch.tensor(
                    np.array(
                        frame['face_rect']).astype(dtype=np.int32))
                
                data_frame['camera_idx'] = prev_nums + idx
                data_list.append(data_frame)
                
            if split == 'train':
                dataset = ListDataset(data_list)
                dataset.meta = {'bg_image': torch.FloatTensor(bc_img) / 255., }
                self.train = [dataset]
                log.info('Training dataset loaded with {} data'.format(len(data_list)))
            if split == 'val':
                dataset = ListDataset(data_list)
                dataset.meta = {'bg_image': torch.FloatTensor(bc_img) / 255., }
                self.val = [dataset]
                log.info('Validation dataset loaded with {} data'.format(len(data_list)))
                # For evaluation, the test dataset is the same as validation dataset
                self.test = [dataset]
                log.info('Test dataset loaded with {} data'.format(len(data_list)))
        
            prev_nums = idx; prev_nums += 1
     


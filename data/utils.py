import torch
from torch.utils.data._utils.collate import default_collate
from pytorch3d.renderer import PerspectiveCameras, OrthographicCameras, FoVPerspectiveCameras, FoVOrthographicCameras
from copy import deepcopy

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    
    return batch


def batch_collate(batch):
    """
    The collate function that merges cameras
    """
    if type(batch[0]) == dict:
        loader = batch[0].items()  
    elif type(batch[0]) in [list, tuple]:
        loader = enumerate(batch[0])
    else:
        raise NotImplementedError #TODO
    
    sublist = lambda batch, key: [data[key] for data in batch]
    attrlist = lambda cameras, attr: [getattr(camera, attr) for camera in cameras]
    
    new_batch = deepcopy(batch[0])
    for k,v in loader:
        if type(v) in [PerspectiveCameras, OrthographicCameras, FoVPerspectiveCameras, FoVOrthographicCameras]:
            if type(v) == PerspectiveCameras:
                cameras = sublist(batch, k)
                camera_batch = PerspectiveCameras(
                    focal_length= torch.cat(attrlist(cameras, 'focal_length'), dim = 0),
                    principal_point= torch.cat(attrlist(cameras, 'principal_point'), dim = 0), 
                    R = torch.cat(attrlist(cameras, 'R'), dim = 0), 
                    T = torch.cat(attrlist(cameras, 'T'), dim = 0)
                )
                new_batch[k] = camera_batch
            else:
                raise NotImplementedError
        else: 
            new_batch[k] = default_collate(sublist(batch, k))
    return new_batch
            
        
def original_collate(batch):
    return default_collate(batch)
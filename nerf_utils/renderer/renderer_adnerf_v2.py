import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from zmq import device

from nerf_utils.renderer.renderer_drvnerf import DrivingRadianceFieldRenderer
from nerf_utils.components.ad_nerf import AudioNet

class AudioDrivenRadianceFieldRenderer_v2(nn.Module):
    """
    Implements a renderer of a Audio Driven Neural Radiance Field.
    Different from original version, it renders full size image in one shot, no matter in Training or Inference.
    Also it can take a mask as input
    """
    
    def __init__(
        self,
        seq_len: int,
        dim_aud: int,
        render_size: Tuple[int, int],
        **kwargs
    ):
        """
        Args:
            kwargs: the arguments for initializing DrivingRadianceFieldRenderer
        """
        super().__init__()

        image_size = kwargs.get('image_size')
        image_height, image_width = image_size

        kwargs.update({'dim_aud': dim_aud})
        self._renderer = DrivingRadianceFieldRenderer(
            **kwargs
        )

        self._drvnet = AudioNet(
            seq_len=seq_len,
            dim_aud=dim_aud
        )

        self.render_height, self.render_width = render_size

        self.bg_image = torch.ones(1, 3, image_height, image_width) #nn.parameter.Parameter(torch.zeros(1, 3, image_height, image_width), requires_grad=False)
            
    def forward(
        self,
        audio: torch.Tensor,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        bg_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ): #  -> Tuple[dict, dict]
        """
        Performs the coarse and fine rendering passes of the audio-driven radiance field
       
        Args:
            aud_para: audio coefficients
            rect: face bounding box rectangular, only applicable in training
            others same as vanilla Nerf
            
        Returns:
            coarse_rgb: tensor revealing the rgb rendering results in coarse stage
            coarse_feat: tensor with higher dimension than rgb for further forward
            coarse_mask: tensor representing the ray is occupied with objects or not

            if sampling_stage include "fine", the followings are also included

            fine_rgb: same as coarse_rgb, but obtained in the fine stage
            fine_feat: same as coarse_feat, but obtained in the fine stage
            fine_mask: same as coarse_mask, but obtained in the fine stage

        """

        # prepare original inputs for vanilla Nerf
        camera_hash = kwargs.get('camera_hash', None) # Optional[str]
        camera = kwargs.get('camera') # CamerasBase
        
        batch = audio.shape[0]
        device = audio.device

        image_height = image_height if image_height is not None else self.render_height
        image_width = image_width if image_width is not None else self.render_width

        resize = lambda x, h, w: F.interpolate(x, (h, w)) if x.shape[-2] != h or x.shape[-1] != w else x

        aud_para = self._drvnet(audio)
        out = self._renderer(
            aud_para = aud_para,
            mask = resize(mask, image_height, image_width) if mask is not None else mask,
            bg_image = resize(bg_image, image_height, image_width) if bg_image is not None else self.bg_image.repeat(batch, 1, 1, 1).to(device),
            image_height = image_height,
            image_width = image_width,
            camera_hash=camera_hash,
            camera=camera)

        return [out[stage + '_rgb'].permute(0, 3, 1, 2) for stage in self._renderer.sample_stages], \
                [out[stage + '_mask'].permute(0, 3, 1, 2) for stage in self._renderer.sample_stages]





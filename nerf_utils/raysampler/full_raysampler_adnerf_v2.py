# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional

import torch
from pytorch3d.renderer import RayBundle#, MultinomialRaysampler
from .utils import MultinomialRaysamplerNerf

from .coarse_raysampler_nerf import NeRFRaysampler

class FullNeRFRaysampler(NeRFRaysampler):
    """
    A variention of NerfRaysampler when you prefer sampling along an specified rectangular,
    with a fixed sample rate, in the pixel axis.
    """

    def __init__(
        self,
        **kwargs
    ):
        """
        Args:
            rect_sample_rate: the probabilty of sampling pixels within the given rectagular bbox
            image_height: the height of image in the dataset.
            image_width: the width of image in the dataset.
            kwargs: other arguments that are same as NerfRaysampler
        """
        super(FullNeRFRaysampler, self).__init__(**kwargs)
        # prepare arguments that vanilla Nerf used
        image_width = kwargs.get('image_width')
        image_height = kwargs.get('image_height')
        n_pts_per_ray = kwargs.get('n_pts_per_ray')
        min_depth = kwargs.get('min_depth')
        max_depth = kwargs.get('max_depth')
        
        # Initialize the grid ray sampler.
        # It is important for vanillar Nerf implementation

        self._grid_raysampler = MultinomialRaysamplerNerf(
            min_x = 0,
            max_x = image_height - 1,
            min_y = 0,
            max_y = image_width - 1,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,)
        

    def get_n_chunks(self, 
                     chunksize: int, 
                     batch_size: int,
                     image_height: int,
                     image_width: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.
            image_height: The height of image to render
            image_width: The width of image to render

        Returns:
            n_chunks: The total number of chunks.
        """
        n_chunks =  int(
            math.ceil(
                (self._grid_raysampler.make_grid(image_height, image_width).numel() * 0.5 ) / chunksize
            )
        )
        return n_chunks

    def forward(
        self,
        mask: Optional[torch.Tensor]=None,
        image_height: Optional[int]=None,
        image_width: Optional[int]=None ,
        **kwargs,
    ): # -> RayBundle
        """
        Args:
            Othe    vbfcvdswqqwwcvdfdfdffddfsdaw        
            Same a···················   s NeRFRaysampler
        """

        # prepare arguments that vanilla Nerf used
        cameras = kwargs.get('cameras')
        chunksize = kwargs.get('chunksize', None)
        chunk_idx = kwargs.get('chunk_idx', 0)
        camera_hash = kwargs.get('camera_hash', None)
        caching = kwargs.get('caching', False)
        
        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device
        
        if camera_hash is not None:
            # The case where we retrieve a camera from cache.
            raise NotImplementedError(
                "This raysampler does not support camera pre-cache, please turn off configs.precache_rays!"
            )
        else:
            # We generate a full ray grid from scratch.
            full_ray_bundle = self._grid_raysampler(
                cameras, 
                mask = mask, 
                image_height = image_height, 
                image_width = image_width,
            )

            full_ray_bundle = self._normalize_raybundle(full_ray_bundle)

        n_pixels = full_ray_bundle.directions.shape[1:-1].numel()

        # In case we test, we take only the requested chunk.
        if chunksize is None:
            chunksize = n_pixels * batch_size

        chunksize = min(chunksize, n_pixels)

        start = chunk_idx * chunksize
        end = min(start + chunksize, n_pixels)

        sel_rays = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=full_ray_bundle.lengths.device,
        )

        # Take the "sel_rays" rays from the full ray bundle.
        ray_bundle = RayBundle(
            *[
                v.view(batch_size, n_pixels, -1)[:, sel_rays]
                .to(device)
                for v in full_ray_bundle
            ]
        )

        if (
            (self._stratified and self.training)
            or (self._stratified_test and not self.training)
        ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        return ray_bundle

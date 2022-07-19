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
from pytorch3d.renderer.cameras import CamerasBase

from .coarse_raysampler_nerf import NeRFRaysampler
import random

class RectangularNeRFRaysampler(NeRFRaysampler):
    """
    A variention of NerfRaysampler when you prefer sampling along an specified rectangular,
    with a fixed sample rate, in the pixel axis.
    """

    def __init__(
        self,
        rect_sample_rate:float,
        fix_grid: bool,
        **kwargs
    ):
        """
        Args:
            rect_sample_rate: the probabilty of sampling pixels within the given rectagular bbox
            image_height: the height of image in the dataset.
            image_width: the width of image in the dataset.
            kwargs: other arguments that are same as NerfRaysampler
        """
        super(RectangularNeRFRaysampler, self).__init__(**kwargs)
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
            max_depth=max_depth,
            fix_grid=fix_grid)

        self.rect_sample_rate = rect_sample_rate
        self.H = image_height
        self.W = image_width
        self.fix_grid = fix_grid
        

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
        return int(
            math.ceil(
                (self._grid_raysampler.make_grid(image_height, image_width).numel() * 0.5 * batch_size) / chunksize
            )
        )

    def forward(
        self,
        rect: Optional[torch.Tensor] = None, 
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        mask: Optional[int] = None,
        **kwargs,
    ): # -> RayBundle
        """
        Args:
            Others same as NeRFRaysampler
        Returns:
            Same as NeRFRaysampler
        """
        # prepare arguments that vanilla Nerf used
        cameras = kwargs.get('cameras')
        chunksize = kwargs.get('chunksize', None)
        chunk_idx = kwargs.get('chunk_idx', 0)
        camera_hash = kwargs.get('camera_hash', None)
        caching = kwargs.get('caching', False)
        
        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        # self.coords = self.coords.to(device)
        if self.fix_grid: # Lets try if range(0, render_size) or range(0, image_size, render_size) will work
            self.coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, image_height - 1, image_height),
                    torch.linspace(0, image_width - 1, image_width),
                ), 
                dim = -1
            ).reshape([-1, 2]).to(device)
        else:
            self.coords = self._grid_raysampler.make_grid(
                image_height, 
                image_width
                ).view(-1, 2).to(device)
        
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
                image_width = image_width)
            full_ray_bundle = self._normalize_raybundle(full_ray_bundle)

        n_pixels = full_ray_bundle.directions.shape[:-1].numel()

        if self.training:
            # During training we randomly subsample rays.
            # The sampling will be within the rectular with the possiblity of rect_sample_rate.

            if batch_size != 1:
                raise NotImplementedError 

            # filter indexes within and withour the bbox
            if self.fix_grid:
                _coords_0 = self.coords[:, 0] / image_height * self.H
                _coords_1 = self.coords[:, 1] / image_width * self.W
                rect_inds = \
                    (_coords_0 >= rect[0]) & \
                    (_coords_0 <= rect[0] + rect[2]) & \
                    (_coords_1 >= rect[1]) & \
                    (_coords_1 <= rect[1] + rect[3])
            else:
                rect_inds = \
                    (self.coords[:, 0] >= rect[0]) & \
                    (self.coords[:, 0] <= rect[0] + rect[2]) & \
                    (self.coords[:, 1] >= rect[1]) & \
                    (self.coords[:, 1] <= rect[1] + rect[3])


            # balance the number of indexes within and without bbox according to rect_sample_rate
            rect_num = int(self._mc_raysampler._n_rays_per_image * self.rect_sample_rate)
            norect_num = self._mc_raysampler._n_rays_per_image - rect_num
            # sample indexes within and withour the bbox
            index_rays = torch.arange(
                n_pixels,
                dtype=torch.long,
                device=full_ray_bundle.lengths.device,)
            
            # rays in the rectangular region
            rect_rays = index_rays[rect_inds]
            rect_inds_samples = random.sample(range(len(rect_rays)), rect_num)
            select_inds_rect = rect_rays[rect_inds_samples]
            # ray out of the rectangular region
            norect_rays = index_rays[~rect_inds]
            norect_inds_samples = random.sample(range(len(norect_rays)), norect_num)
            select_inds_norect = norect_rays[norect_inds_samples]
            # concatenate up        
            sel_rays = torch.cat([select_inds_rect, select_inds_norect])        

            # # TODO: DEBUG: draw a sampling map
            # import cv2, os, numpy as np            
            # tensor2numpy = lambda x: x.detach().cpu().numpy()
            # pixels = tensor2numpy(torch.zeros_like(index_rays))
            # pixels[tensor2numpy(sel_rays)] = 255
            # from hydra.utils import get_original_cwd
            # cv2.imwrite(os.path.join(get_original_cwd(), 'wasted/sample_indexes.jpg'), pixels[...,None].reshape([450, 450]))
            
        else:
            # In case we test, we take only the requested chunk.
            if chunksize is None:
                chunksize = n_pixels * batch_size
            start = chunk_idx * chunksize * batch_size
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
                v.view(n_pixels, -1)[sel_rays]
                .view(batch_size, sel_rays.numel() // batch_size, -1)
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

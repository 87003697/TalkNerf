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

class NeRFRaysampler(torch.nn.Module):
    """
    A variention of NerfRaysampler when you prefer sampling along an specified rectangular,
    with a fixed sample rate, in the pixel axis.
    """

    def __init__(
        self,
        num_idxes_height: int,
        num_idxes_width: int,
        **kwargs
    ):
        """
        Args:
            rect_sample_rate: the probabilty of sampling pixels within the given rectagular bbox
            image_height: the height of image in the dataset.
            image_width: the width of image in the dataset.
            kwargs: other arguments that are same as NerfRaysampler
        """
        super().__init__()
        # prepare arguments that vanilla Nerf used
        image_width = kwargs.get('image_width')
        image_height = kwargs.get('image_height')
        n_pts_per_ray = kwargs.get('n_pts_per_ray')
        min_depth = kwargs.get('min_depth')
        max_depth = kwargs.get('max_depth')
        stratified = kwargs.get('stratified')
        stratified_test = kwargs.get('stratified_test')
        # Initialize the grid ray sampler.
        # It is important for vanillar Nerf implementation
        self._grid_raysampler = MultinomialRaysamplerNerf(
            min_x = 0,
            max_x = image_height - 1,
            min_y = 0,
            max_y = image_width - 1,
            image_width=num_idxes_width,
            image_height=num_idxes_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,)

        self.H = image_height
        self.W = image_width

        self.num_idxes_height = num_idxes_height
        self.num_idxes_width = num_idxes_width


        self._stratified = stratified
        self._stratified_test = stratified_test


    def forward(
        self,
        ndc : bool = False,
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
            full_ray_bundle = self._grid_raysampler(cameras, ndc = ndc)
            full_ray_bundle = self._normalize_raybundle(full_ray_bundle)
        if (
                (self._stratified and self.training)
                or (self._stratified_test and not self.training)
            ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(full_ray_bundle)
        else:
            ray_bundle = full_ray_bundle
        return ray_bundle
    
    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm.
        """
        ray_bundle = ray_bundle._replace(
            directions=torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        )
        return ray_bundle
    
    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        z_vals = lower + (upper - lower) * torch.rand_like(lower)
        return ray_bundle._replace(lengths=z_vals)

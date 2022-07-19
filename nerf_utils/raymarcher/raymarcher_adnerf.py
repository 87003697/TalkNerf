# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer import RayBundle
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,)
from typing import Optional
from nerf_utils.utils import sample_images_at_int_locs


class EmissionAbsorptionADNeRFRaymarcher(EmissionAbsorptionRaymarcher):
    """
    Apart from EmissionAbsorptionNeRFRaymarcher, EmissionAbsorptionADNeRFRaymarcher
    also takes as input the original background and replace the final color prediction with this rgb value
    """

    def forward(
        self,
        ray_bundle: RayBundle,
        bg_image: Optional[torch.Tensor] = None, 
        **kwargs,
    ) : # -> torch.Tensor
        """
        Args:
            bg_color: the specified background color
            Other arguments are same as EmissionAbsorptionRaymarcher

        Returns:
            Same as EmissionAbsorptionRaymarcher
        """

        # prepare arguments
        rays_densities = kwargs.get('rays_densities') #torch.Tensor,
        rays_features = kwargs.get('rays_features') # torch.Tensor,
        eps = kwargs.get('eps', 1e-10) # float
        return_feat = kwargs.get('return_feat', False)

        if return_feat:
            rays_features, latent_feature= rays_features 
            
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)

        # Raymarcher with background v3      
        # assign the background density to be 1
        rays_densities = rays_densities[..., 0]

        # the absorption function keeps the same
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption

        # pick up background values
        bg_pixels = []
        ba = rays_densities.shape[0]

        # reshape bg_image
        if len(bg_image.shape) == 3: # 3 channels to 4 channels
            bg_image = bg_image.unsqueeze(0)
        if bg_image.shape[1] == 3: # channel last to channel first
            bg_image = bg_image.permute(0, 2, 3, 1)

        bg_pixels = sample_images_at_int_locs(bg_image, ray_bundle.xys.type(torch.LongTensor))

        # for idx in range(ba):
        #     if len(bg_image.shape) == 4: # 4 channels to 3 channels
        #         _bg_image = bg_image[idx]
        #     else:
        #         _bg_image = bg_image
        #     if _bg_image.shape[0] == 3: # channel first to channel last
        #         _bg_image = _bg_image.permute(1, 2, 0)
        #     pixels = _bg_image[pixel_idxs[idx,:,0], pixel_idxs[idx,:,1]]
            # bg_pixels.append(pixels[None])
        # bg_pixels = torch.cat(bg_pixels)

        # append the background color to the ray_features
        
        # rays_features[...,-1:,:3] += bg_pixels.unsqueeze(-2)

        # final color
        features = (weights[..., None] * rays_features).sum(dim=-2) + \
            (1 - weights[..., None].sum(dim=-2)) * bg_pixels

        if not return_feat:
            return features, weights
        else: # return latent feature
            latent = (weights[..., None] * latent_feature).sum(dim=-2)
            return features, weights, latent
    
        # # Raymarcher with background v2        
        # # assign the background density to be 1
        # rays_densities = rays_densities[..., 0]

        # # the absorption function keeps the same
        # absorption = _shifted_cumprod(
        #     (1.0 + eps) - rays_densities, shift=self.surface_thickness
        # )
        # weights = rays_densities * absorption

        # # pick up background values
        # bg_pixels = []
        # ba = rays_densities.shape[0]
        # pixel_idxs = ray_bundle.xys.type(torch.LongTensor)
        # for idx in range(ba):
        #     pixels = bg_image[pixel_idxs[idx,:,0], pixel_idxs[idx,:,1]]
        #     bg_pixels.append(pixels[None])
        # bg_pixels = torch.cat(bg_pixels)

        # # append the background color to the ray_features
        # rays_features[...,-1:,:3] += bg_pixels.unsqueeze(-2)

        # # final color
        # features = (weights[..., None] * rays_features).sum(dim=-2)

        # return features, weights
    
        # # Raymarcher with background v1
        # # assign the background density to be 1
        # rays_densities = rays_densities[..., 0]
        # rays_densities = torch.cat(
        #     [
        #         rays_densities, 
        #         torch.ones_like(rays_densities)[...,:1]
        #     ], dim = -1)

        # # the absorption function keeps the same
        # absorption = _shifted_cumprod(
        #     (1.0 + eps) - rays_densities, shift=self.surface_thickness
        # )
        
        # weights = rays_densities * absorption
        
        # # pick up background values
        # bg_pixels = []
        # ba = rays_densities.shape[0]
        # pixel_idxs = ray_bundle.xys.type(torch.LongTensor)
        # for idx in range(ba):
        #     pixels = bg_image[pixel_idxs[idx,:,0], pixel_idxs[idx,:,1]]
        #     bg_pixels.append(pixels[None])
        # bg_pixels = torch.cat(bg_pixels)

        # # append the background color to the ray_features
        # rays_features = torch.cat(
        #     [
        #         rays_features, 
        #         bg_pixels.unsqueeze(-2)
        #     ], dim = -2)

        # # final color
        # features = (weights[..., None] * rays_features).sum(dim=-2)

        # return features, weights[...,:-1]
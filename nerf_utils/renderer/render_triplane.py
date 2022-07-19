from typing import Callable, Tuple, Optional
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneRender(torch.nn.Module):
    def __init__(self, 
                 raymarcher: Callable, 
                 normalizer: Callable,
                 raysampler: Optional[Callable]=None
                 ): #-> None:
        super().__init__()
        if not callable(raysampler):
            raise ValueError('"raysampler" has to be a "Callable" object.')
        if not callable(raymarcher):
            raise ValueError('"raymarcher" has to be a "Callable" object.')
        
        self.raysampler = raysampler
        self.raymarcher = raymarcher
        self.normalizer = normalizer

    def forward(self, 
                cameras: CamerasBase, 
                feat_xy: torch.Tensor, 
                feat_yz: torch.Tensor, 
                feat_xz: torch.Tensor,  
                density_noise_std: float = 0.0, 
                **kwargs): # -> Tuple[torch.Tensor, RayBundle]:
        
        ray_bundle = self.raysampler(
            cameras=cameras, ndc = False,
            **kwargs)
        
        rays_densities, rays_features = self.triplane_interpolate_function(
            ray_bundle=ray_bundle, 
            cameras=cameras, 
            feat_xy=feat_xy,
            feat_yz=feat_yz,
            feat_xz=feat_xz,
            density_noise_std=density_noise_std,
            **kwargs)
        
        images = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs)
        
        return images, ray_bundle
    
    
    def triplane_interpolate_function(self, 
                                      ray_bundle:torch.Tensor, 
                                      feat_xy:torch.Tensor, 
                                      feat_yz:torch.Tensor, 
                                      feat_xz:torch.Tensor, 
                                      density_noise_std:float,
                                      **kwargs):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        
        # fetch features from tri-plane
        B, H, W, N, D = rays_points_world.shape
        ray_points_normalized = self.normalizer(rays_points_world.view(B, -1 , D).transpose(1, 2)).transpose(1,2).view(B, H, W, N, D)
        # print('x minimum: {}, x maximum: {} \n y minimum: {}, y maximum: {} \n z minimum: {}, z maximum: {} \n'.format(
        #     torch.min(ray_points_normalized[..., 0]), torch.max(ray_points_normalized[..., 0]), 
        #     torch.min(ray_points_normalized[..., 1]), torch.max(ray_points_normalized[..., 1]),
        #     torch.min(ray_points_normalized[..., 2]), torch.max(ray_points_normalized[..., 2]),))
        # print('running var: {}, running mean: {}'.format(self.normalizer.running_var, self.normalizer.running_mean))
        features = self.bilinear_sample_tri_plane(
            points=ray_points_normalized,
            feat_xy=feat_xy,
            feat_yz=feat_yz,
            feat_xz=feat_xz)
        
        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle, density_noise_std
        )
        return rays_densities, rays_colors
        

        
    def _get_densities_and_colors(self, 
                                  features: torch.Tensor,
                                  ray_bundle: RayBundle,
                                  density_noise_std: float):

        
        rays_densities = [self._get_densities(
            features=feat, 
            depth_values=ray_bundle.lengths,
            density_noise_std=density_noise_std)
                            for feat in features.chunk(3, dim = 1)]

            
        rays_colors = [self._get_colors(
            features=feat,
            rays_directions=ray_bundle.directions)
                        for feat in features.chunk(3, dim = 1)]


        return \
            torch.mean(torch.cat(rays_densities, dim = -1), dim = -1, keepdim=True) ,\
            torch.cat(rays_colors, dim = -1)
            
    def _get_densities(self,
                        features: torch.Tensor,
                        depth_values: torch.Tensor,
                        density_noise_std: float):
        
        
        raw_densities = features.permute(0, 3, 4, 2, 1)[..., :1]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]
        if density_noise_std > 0.0:
            raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()

        return densities #B*H*W*N*1
    
    def _get_colors(self,
                    features: torch.Tensor,
                    rays_directions: torch.Tensor,):
        return features.permute(0, 3, 4, 2, 1) #B*H*W*N*C
    
    def bilinear_sample_tri_plane(self, points, feat_xy, feat_yz, feat_xz):
        # bilinear_sample_tri_plane
        B, H, W, N, D = points.shape
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        posi2featlist = lambda feat, posi : [F.grid_sample(feat, grid=posi[idx]) for idx in range(N)]
        
        return torch.concat([
            torch.stack(posi2featlist(
                feat_xy, 
                torch.stack([x, y], dim=-1).permute(3, 0, 1, 2, 4)),
                        dim=2), 
            torch.stack(posi2featlist(
                feat_yz, 
                torch.stack([x, z], dim=-1).permute(3, 0, 1, 2, 4)), 
                        dim=2), 
            torch.stack(posi2featlist(
                feat_xz, 
                torch.stack([y, z], dim=-1).permute(3, 0, 1, 2, 4)), 
                        dim=2)
            ]
            , dim = 1) # B*C*N*H*W

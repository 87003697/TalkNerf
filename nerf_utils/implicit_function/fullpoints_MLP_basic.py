
from typing import Tuple

import torch
from nerf_utils.implicit_function.nerf_basic import  _xavier_init_weight, MLPWithInputSkips
from pytorch3d.renderer import HarmonicEmbedding, RayBundle, ray_bundle_to_ray_points

class FullPointsMLP(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_hidden_neurons_xyz: int = 256,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        transparent_init: bool = False,
        **kwargs,
    ):
    
        super().__init__()
        
        self.harmonic_embedding_xyz =  HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        self.mlp_xyz = MLPWithInputSkips(
            n_layers_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
            transparent_init=transparent_init,
        )
        
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons_xyz, 1),
        )
        
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons_xyz, n_hidden_neurons_xyz),
            torch.nn.ReLU(True)
        )
        
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons_xyz, 3),
            torch.nn.Sigmoid(),
        )
        if not transparent_init:
            _xavier_init_weight(self.density_layer[0])
            _xavier_init_weight(self.feature_layer[0])
            _xavier_init_weight(self.color_layer[0])
            
    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ) : #-> torch.Tensor
        
        raw_densities = self.density_layer(features)
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
        return densities
    
    def _get_features(
        self, features: torch.Tensor, rays_directions: torch.Tensor
    ): #  -> torch.Tensor
        
        return self.feature_layer(features)

    def _get_colors(
        self, features: torch.Tensor,
    ): #  -> torch.Tensor
        return self.color_layer(features)
    
    
    def _get_densities_features_colors(
        self, features: torch.Tensor, 
        ray_bundle: RayBundle, 
        density_noise_std: float
    ): 
        
        rays_densities = self._get_densities(
            features, ray_bundle.lengths, density_noise_std
        )
        rays_features = self._get_features(
            features, ray_bundle.directions
        )
        rays_colors = self._get_colors(
            rays_features
        )
        return rays_densities, rays_features, rays_colors
        
        
    def forward(
        self,
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        **kwargs,
    ):
        
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]
        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]
        rays_densities, rays_features, ray_colors = self._get_densities_features_colors(
            features, ray_bundle, density_noise_std
        )
        return rays_densities, [rays_features, ray_colors]
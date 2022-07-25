from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle
from .nerf_basic import MLPWithInputSkips, NeuralRadianceField
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
    
class AudioDrivenNeuralRadianceField(NeuralRadianceField):
    def __init__(
        self,
        dim_aud: int = 64, 
        **kwargs,
    ):
        """
        Args:
            dim_aud: the dimension of audio features.
            kwargs: the arguments for initializing original nerf model.
            append_xyz: same as in vanilla Nerf
        """
        # prepare arguments that vanilla nerf used
        n_harmonic_functions_xyz = kwargs.get('n_harmonic_functions_xyz') # int
        n_layers_xyz = kwargs.get('n_layers_xyz') # int
        n_hidden_neurons_xyz = kwargs.get('n_hidden_neurons_xyz') # int
        append_xyz = kwargs.get('append_xyz',  (5,)) # Tuple[int]
        transparent_init = kwargs.get('transparent_init', True)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 
        super().__init__(**kwargs)
        self.mlp_xyz = MLPWithInputSkips(
            n_layers_xyz, 
            embedding_dim_xyz + dim_aud,
            n_hidden_neurons_xyz,
            embedding_dim_xyz + dim_aud,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
            transparent_init=transparent_init,
            )

    def forward(
        self,
        aud_para: torch.Tensor, 
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        **kwargs,
    ) : #-> Tuple[torch.Tensor, torch.Tensor]
        """
        The forwarding of the AD-Nerf implicit function.
        
        Args:
            aud_para: audio coefficients
            others same as vanilla Nerf

        Returns:
            same as vanillar Nerf
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]


        # self.mlp maps each harmonic embedding to a latent feature space.
        batch_size = embeds_xyz.shape[0]

        # embeds_xyz.shape = torch.Size([1, 1024, 64, 63])
        # aud_para.repeat(*embeds_xyz.shape[:-1],1).shape = torch.Size([1, 1024, 64, 64])
        # aud_para.shape = torch.Size([64]) or torch.Size([batch_size, 64])
        features = self.mlp_xyz(
            torch.cat([
                embeds_xyz, 
                aud_para.repeat(*embeds_xyz.shape[:-1], 1) \
                    if len(aud_para.shape) == 1 \
                    else aud_para.reshape(batch_size, 1, 1, -1).repeat(1, *embeds_xyz.shape[1:-1], 1), 
                ],
                dim = -1), 
            torch.cat([
                embeds_xyz, 
                aud_para.repeat(*embeds_xyz.shape[:-1],1) \
                    if len(aud_para.shape) == 1 \
                    else aud_para.reshape(batch_size, 1, 1, -1).repeat(1, *embeds_xyz.shape[1:-1], 1),
                ], 
            dim = -1),
        )
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle, density_noise_std
        )
        return rays_densities, rays_colors
import torch
import itertools
from typing import Callable, List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import ImplicitRenderer
from zmq import device

from nerf_utils.implicit_function.fullpoints_MLP_basic import FullPointsMLP
from nerf_utils.raysampler.full_raysampler_adnerf import NeRFRaysampler
from nerf_utils.raymarcher.raymarcher_nerfgan_bgmask import EmissionAbsorptionNeRFRaymarcher
from nerf_utils.raysampler.fine_raysampler_eg3d import ProbabilisticRaysampler

from nerf_utils.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from gan_utils.generators.nerfgan_bgmask import FullGenerator

from torch.nn import InstanceNorm1d

class MLPRadianceFieldRenderer(torch.nn.Module):
    """
    Implements a renderer of a Neural Radiance Field.

    This class holds pointers to the fine and coarse renderer objects, which are
    instances of `pytorch3d.renderer.ImplicitRenderer`, and pointers to the
    neural networks representing the fine and coarse Neural Radiance Fields,
    which are instances of `NeuralRadianceField`.

    The rendering forward pass proceeds as follows:
        1) For a given input camera, rendering rays are generated with the
            `NeRFRaysampler` object of `self._renderer['coarse']`.
            In the training mode (`self.training==True`), the rays are a set
                of `n_rays_per_image` random 2D locations of the image grid.
            In the evaluation mode (`self.training==False`), the rays correspond
                to the full image grid. The rays are further split to
                `chunk_size_test`-sized chunks to prevent out-of-memory errors.
        2) For each ray point, the coarse `NeuralRadianceField` MLP is evaluated.
            The pointer to this MLP is stored in `self._implicit_function['coarse']`
        3) The coarse radiance field is rendered with the
            `EmissionAbsorptionNeRFRaymarcher` object of `self._renderer['coarse']`.
        4) The coarse raymarcher outputs a probability distribution that guides
            the importance raysampling of the fine rendering pass. The
            `ProbabilisticRaysampler` stored in `self._renderer['fine'].raysampler`
            implements the importance ray-sampling.
        5) Similar to 2) the fine MLP in `self._implicit_function['fine']`
            labels the ray points with occupancies and colors.
        6) self._renderer['fine'].raymarcher` generates the final fine render.
        7) The fine and coarse renders are compared to the ground truth input image
            with PSNR and MSE metrics.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        num_idxes: list[int],
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        density_noise_std: float = 0.0,
        n_harmonic_functions_xyz: int = 6,
        n_hidden_neurons_xyz: int = 256,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        transparent_init: bool = False,
    ):
        """
        Args:
            To be updated
        """
        
        super().__init__()
        
        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()
        
        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        render_height, render_width = image_size
        assert render_height == render_width
        
        for render_pass in ['coarse',]:
            
            # Initialize the coarse raysampler.        
            if render_pass == 'coarse':
                
                commen_kwargs = {
                    'min_depth': min_depth, 
                    'max_depth': max_depth, 
                    'stratified': stratified,
                    'stratified_test': stratified_test,
                    'image_height': render_height,
                    'image_width': render_width
                }
            
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    num_idxes_height=num_idxes[0],
                    num_idxes_width=num_idxes[0],
                    **commen_kwargs
                )         

            # Initialize the renderer
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )
            

            # Instantiate the fine/coarse NeuralRadianceField module.
            commen_kwargs = {
                'n_harmonic_functions_xyz': n_harmonic_functions_xyz,
                'n_hidden_neurons_xyz': n_hidden_neurons_xyz,
                'n_layers_xyz': n_layers_xyz,
                'append_xyz': append_xyz,
                'transparent_init': transparent_init}
            
            self._implicit_function[render_pass] = FullPointsMLP(
                **commen_kwargs
            )

        self._density_noise_std = density_noise_std
        self._image_size = image_size

        self.bg_pixels = torch.nn.parameter.Parameter(
            torch.zeros([1, 3, num_idxes[0], num_idxes[0]]),
            requires_grad=False
        )

                        
    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,   
        audio: torch.Tensor,                               
        ref_image: Optional[torch.Tensor]=None,
        needs_multi_res: bool = False,
        needs_mask: bool = False,
    ): #  -> Tuple[dict, dict]
        
        batch = audio.shape[0]
        
        # rendering results        
        feature, color, mask = self.nerf_rendering(
            camera_hash, 
            camera)
        
        return [color + (1 - mask) * self.bg_pixels.repeat(batch, 1, 1, 1)], [mask] 
    
    def nerf_rendering(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,   
    ):
        render_kwargs = {
            'density_noise_std': (self._density_noise_std if self.training else 0.0),
            'cameras': camera
        }
        (features, rgbs, weights), ray_bundle_out = self._renderer['coarse'](
            input_ray_bundle=None, 
            ray_weights=None,
            volumetric_function=self._implicit_function['coarse'],
            **render_kwargs
        )
        
        return features.permute(0, 3, 1, 2), \
                rgbs.permute(0, 3, 1, 2), \
                weights[..., None].sum(dim=-2).permute(0, 3, 1, 2)        
        
        
        

        




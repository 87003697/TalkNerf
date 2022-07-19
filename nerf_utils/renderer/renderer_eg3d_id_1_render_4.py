import torch
import itertools
from typing import Callable, List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase

from nerf_utils.renderer.render_triplane import TriplaneRender
from nerf_utils.raysampler.full_raysampler_adnerf import NeRFRaysampler
from nerf_utils.raymarcher.raymarcher_nerf import EmissionAbsorptionNeRFRaymarcher
from nerf_utils.raysampler.fine_raysampler_eg3d import ProbabilisticRaysampler
from gan_utils.generators.eg3d_gen_id_1_render_4 import FullGenerator
from nerf_utils.utils import calc_mse, calc_psnr, sample_images_at_mc_locs

from torch.nn import InstanceNorm1d

from .renderer_eg3d import Efficient3dRadianceFieldRenderer as father

class Efficient3dRadianceFieldRenderer(father):
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
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        density_noise_std: float = 0.0,
        visualization: bool = False,
        narrow: float= 1.0,
        isconcat: bool = True,
        lr_mlp: float = 0.01,
        blur_kernel: list = [1,3,3,1],
        channel_multiplier: int = 2,
        n_mlp: int = 8,
        style_dim: int = 512,
        size: int = 512,
        audio_dim: int = 464,
        norm_momentum: float = 0.1,
        tri_plane_sizes: list = [64],
    ):

        super(Efficient3dRadianceFieldRenderer, self).__init__(        
            image_size,
            n_pts_per_ray,
            n_pts_per_ray_fine,
            min_depth,
            max_depth,
            stratified,
            stratified_test,
            density_noise_std,
            visualization,
            narrow,
            isconcat,
            lr_mlp,
            blur_kernel,
            channel_multiplier,
            n_mlp,
            style_dim,
            audio_dim,
            norm_momentum,
            tri_plane_sizes)            

        # Instantiate the fine/coarse NeuralRadianceField module.
        image_height, image_width = image_size
        self._implicit_function = FullGenerator(
            size=image_height, style_dim=style_dim, n_mlp=n_mlp, lr_mlp=lr_mlp, blur_kernel=blur_kernel, 
            channel_multiplier=channel_multiplier, narrow=narrow, isconcat=isconcat, audio_dim=audio_dim
            )

        
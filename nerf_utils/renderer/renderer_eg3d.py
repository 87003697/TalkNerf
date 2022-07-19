import torch
import itertools
from typing import Callable, List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase

from nerf_utils.renderer.render_triplane import TriplaneRender
from nerf_utils.raysampler.full_raysampler_adnerf import NeRFRaysampler
from nerf_utils.raymarcher.raymarcher_nerf import EmissionAbsorptionNeRFRaymarcher
from nerf_utils.raysampler.fine_raysampler_eg3d import ProbabilisticRaysampler
from gan_utils.generators.eg3d_gen import FullGenerator
from nerf_utils.utils import calc_mse, calc_psnr, sample_images_at_mc_locs

from torch.nn import InstanceNorm1d

class Efficient3dRadianceFieldRenderer(torch.nn.Module):
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
        audio_dim: int = 464,
        norm_momentum: float = 0.1,
        tri_plane_sizes: list = [64],
    ):
        """
        Args:
            To be updated
        """
        
        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer_coarse = torch.nn.ModuleDict()
        self._renderer_fine = torch.nn.ModuleDict()
        
        self._implicit_function = torch.nn.ModuleDict()
        image_height, image_width = image_size
              
        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()
        # tri_plane resolunction = tions
        self._tri_plane_sizes = tri_plane_sizes
        
        # components for coarse sampling 
        for render_size in self._tri_plane_sizes:
        
            # Parse out image dimensions.
            render_height, render_width = render_size, render_size
            
            commen_kwargs = {
                'min_depth': min_depth, 
                'max_depth': max_depth, 
                'stratified': stratified,
                'stratified_test': stratified_test,
                'image_height': image_height,
                'image_width': image_width,
            }
            # Initialize the coarse raysampler.
            raysampler_coarse =  NeRFRaysampler(
                num_idxes_height=render_height,
                num_idxes_width=render_width,
                n_pts_per_ray=n_pts_per_ray,
                **commen_kwargs
            )           
            
            raysampler_fine =ProbabilisticRaysampler(
                n_pts_per_ray=n_pts_per_ray_fine, 
                stratified=stratified, 
                stratified_test=stratified_test
            )  

            commen_kwargs = {
                'num_features': 3,
                'momentum': norm_momentum,
                'track_running_stats': True
            }
            
            normalizer_coarse = InstanceNorm1d(
                **commen_kwargs
            )
            
            normalizer_fine = InstanceNorm1d(
                **commen_kwargs
            )
            
            self._renderer_coarse[str(render_size)] = TriplaneRender(
                raysampler=raysampler_coarse,
                raymarcher=raymarcher,
                normalizer=normalizer_coarse
            )
            
            self._renderer_fine[str(render_size)] = TriplaneRender(
                raysampler=raysampler_fine,
                raymarcher=raymarcher,
                normalizer=normalizer_fine
            )
            
        # Instantiate the fine/coarse NeuralRadianceField module.
        self._implicit_function = FullGenerator(
            size=image_height, style_dim=style_dim, n_mlp=n_mlp, lr_mlp=lr_mlp, blur_kernel=blur_kernel, 
            channel_multiplier=channel_multiplier, narrow=narrow, isconcat=isconcat, audio_dim=audio_dim
            )

        self._density_noise_std = density_noise_std
        self._image_size = image_height
                
        self.visualization = visualization
        
    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,   
        audio: torch.Tensor,                               
        ref_image: Optional[torch.Tensor]=None,
    ): #  -> Tuple[dict, dict]
        
        
        batch = audio.shape[0]
        attr = self._implicit_function.image2attr(ref_image.repeat(batch, 1, 1, 1))
        embed = self._implicit_function.audio2embed(audio)
        attr = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in attr))[::-1]
        out, skip, latent, mix_zip = self._implicit_function.generator.prepare([embed], noise=attr[1:])

        # upsampling 
        i = 1
        for idx, (conv1, conv2, noise1, noise2, to_rgb) in enumerate(mix_zip):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            shape = out.shape[-1]  # torch.Size([2, 1024, 64, 64])
            if shape in self._tri_plane_sizes:
                out, out_coarse, weights = self._triplane_sample(
                    feat=out, 
                    cameras=camera,
                    coarse_renderer=self._renderer_coarse[str(shape)],
                    fine_render=self._renderer_fine[str(shape)],)

            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        """
        out.shape
        torch.Size([2, 1024, 8, 8])
        torch.Size([2, 1024, 16, 16])
        torch.Size([2, 1024, 32, 32])
        torch.Size([2, 1024, 64, 64])
        torch.Size([2, 512, 128, 128])
        torch.Size([2, 256, 256, 256])
        torch.Size([2, 128, 512, 512])
        """
        return skip
        
        
    def _triplane_sample(self,
                         feat: torch.Tensor,
                         cameras: CamerasBase,
                         coarse_renderer: Callable,
                         fine_render:Callable):

        # tri-plane rendering
        feat_xy, feat_yz, feat_xz = feat.chunk(3, dim = 1)
            
        render_kwargs = {
            'feat_xy': feat_xy, 
            'feat_yz': feat_yz,
            'feat_xz': feat_xz,
            'density_noise_std': (self._density_noise_std if self.training else 0.0),
            'cameras': cameras
        }
        
        (coarse, weights), ray_bundle_out = coarse_renderer(
            input_ray_bundle=None, 
            ray_weights=None,
            **render_kwargs
        )
                
        (fine, weights), _ = fine_render(
            input_ray_bundle=ray_bundle_out,
            ray_weights=weights,
            **render_kwargs
        )
        
        return fine.permute(0, 3, 1, 2).contiguous(), \
            coarse.permute(0, 3, 1, 2).contiguous(), \
            weights.permute(0, 3, 1, 2).contiguous()
        




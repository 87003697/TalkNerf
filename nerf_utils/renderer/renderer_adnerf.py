from numpy import fix
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer.cameras import CamerasBase


from nerf_utils.raymarcher.raymarcher_adnerf import EmissionAbsorptionADNeRFRaymarcher
from nerf_utils.raysampler.rect_raysampler_adnerf import RectangularNeRFRaysampler
from nerf_utils.raysampler.fine_raysampler_nerf import ProbabilisticRaysampler
from nerf_utils.implicit_function.adnerf import AudioDrivenNeuralRadianceField
from nerf_utils.utils import calc_mse, calc_psnr, sample_images_at_int_locs

from .renderer_nerf import RadianceFieldRenderer

class AudioDrivenRadianceFieldRenderer(RadianceFieldRenderer):
    """
    Implements a renderer of a Audio Driven Neural Radiance Field.
    """
    
    def __init__(
        self,
        dim_aud: int,
        render_size: Tuple[int, int],
        rect_sample_rate: float,
        fix_grid: bool = False,   
        **kwargs
    ):
        """
        Args:
            dim_aud: the dimension of audio features.
            kwargs: the arguments for initializing original nerf model.
        """
        
        super(AudioDrivenRadianceFieldRenderer, self).__init__(**kwargs)
        # prepare arguments that vanilla nerf used
        image_size = kwargs.get('image_size') #Tuple[int, int]
        n_pts_per_ray = kwargs.get('n_pts_per_ray') # int
        n_pts_per_ray_fine = kwargs.get('n_pts_per_ray_fine') # int
        n_rays_per_image = kwargs.get('n_rays_per_image') # int
        min_depth = kwargs.get('min_depth') # float
        max_depth = kwargs.get('max_depth') # float
        stratified = kwargs.get('stratified') # bool
        stratified_test = kwargs.get('stratified_test') # bool
        chunk_size_test = kwargs.get('chunk_size_test') # int       
        n_harmonic_functions_xyz = kwargs.get('n_harmonic_functions_xyz', 6) # int
        n_harmonic_functions_dir = kwargs.get('n_harmonic_functions_dir') # int
        n_hidden_neurons_xyz = kwargs.get('n_hidden_neurons_xyz') # int 
        n_hidden_neurons_dir = kwargs.get('n_hidden_neurons_dir') # int 
        n_layers_xyz = kwargs.get('n_layers_xyz', 8) # int 
        append_xyz = kwargs.get('append_xyz', (5,)) #  Tuple[int]
        density_noise_std = kwargs.get('density_noise_std', 0.0)
        visualization = kwargs.get('visualization', False)      
        transparent_init = kwargs.get('transparent_init', True)
        
        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()
        
        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionADNeRFRaymarcher()
        
        # Parse out image dimensions.
        self.image_height, self.image_width = image_size
        self.render_height, self.render_width = render_size

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                other_kwargs = { 
                    'n_pts_per_ray': n_pts_per_ray,
                    'min_depth': min_depth,
                    'max_depth': max_depth, 
                    'stratified': stratified, 
                    'stratified_test': stratified_test,
                    'n_rays_per_image': n_rays_per_image,
                    'image_height': self.image_height,
                    'image_width': self.image_width} # arguments that original NeRFRaysampler in vanillar Nerf used
                raysampler = RectangularNeRFRaysampler(
                    rect_sample_rate=rect_sample_rate,
                    fix_grid=fix_grid,
                    **other_kwargs)
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                other_kwargs = {
                    'n_pts_per_ray': n_pts_per_ray_fine,
                    'stratified': stratified,
                    'stratified_test': stratified_test} # arguments that original ProbabilisticRaysampler in vanillar Nerf used
                raysampler = ProbabilisticRaysampler(
                    **other_kwargs)
                
            # Initialize the fine/coarse renderer.
            other_kwargs = {
                'raysampler':raysampler,
                'raymarcher':raymarcher} # arguments that original ImplicitRenderer in vanillar Nerf used
            self._renderer[render_pass] = ImplicitRenderer(
                **other_kwargs)

            # Instantiate the fine/coarse NeuralRadianceField module.
            other_kwargs = {
                'n_harmonic_functions_xyz': n_harmonic_functions_xyz,
                'n_harmonic_functions_dir': n_harmonic_functions_dir, 
                'n_hidden_neurons_xyz': n_hidden_neurons_xyz,
                'n_hidden_neurons_dir': n_hidden_neurons_dir,
                'n_layers_xyz': n_layers_xyz,
                'append_xyz': append_xyz,
                'transparent_init': transparent_init} # arguments that original NeuralRadianceField in vanillar Nerf used
            self._implicit_function[render_pass] = AudioDrivenNeuralRadianceField(
                dim_aud = dim_aud,  
                **other_kwargs)

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization
            
    def forward(
        self,
        aud_para: torch.Tensor,
        rect: Optional[torch.Tensor] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        bg_image: Optional[torch.Tensor] = None,
        **kwargs
    ): #  -> Tuple[dict, dict]
        """
        Performs the coarse and fine rendering passes of the audio-driven radiance field
       
        Args:
            aud_para: audio coefficients
            rect: face bounding box rectangular, only applicable in training
            others same as vanilla Nerf
            
        Returns:
            same as vanillar Nerf
        """
        # prepare original inputs for vanilla Nerf
        camera_hash = kwargs.get('camera_hash', None) # Optional[str]
        camera = kwargs.get('camera') # CamerasBase
        image = kwargs.get('image') # torch.Tensor
        image_height = self.render_height if image_height == None else image_height
        image_width = self.render_width if image_width == None else image_width
        
        if not self.training:
            # Full evaluation pass.
            other_kwargs = {
                'chunksize': self._chunk_size_test,
                'batch_size': camera.R.shape[0]} # arguments that original NeRFRaysampler in vanillar Nerf used
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                image_height = image_height,
                image_width = image_width,
                **other_kwargs)
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        other_kwargs = {
            'camera_hash': camera_hash,
            'camera': camera,
            'image': image,} # paraameters that original _process_ray_chunk function in vanillar Nerf used, except chunk_idx
        chunk_outputs = [
            self._process_ray_chunk(
                aud_para = aud_para,
                rect = rect,
                chunk_idx = chunk_idx,
                bg_image = bg_image, 
                image_height = image_height,
                image_width = image_width,
                **other_kwargs)
            for chunk_idx in range(n_chunks)]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            # chunk_outputs[0]['rgb_fine'].shape = torch.Size([1, 6000, 3])
            # chunk_outputs[0]['rgb_coarse'].shape = torch.Size([1, 6000, 3])
            # chunk_outputs[0]['rgb_gt'].shape = torch.Size([1, 6000, 3])
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, image_height, image_width, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "fine"):
                for metric_name, metric_fun in zip(
                    ("mse", "psnr"), (calc_mse, calc_psnr)
                ):
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                    )

        return out, metrics

    def _process_ray_chunk(
        self,
        aud_para: torch.Tensor,
        rect: Optional[torch.Tensor] = None, 
        bg_image: Optional[torch.Tensor] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        **kwargs
    ): # -> dict
        """
        Performs the coarse and fine rendering passes of the audio-driven radiance field
       
        Args:
            aud_para: audio coefficients
            rect: face bounding box rectangular, only applicable in training
            others same as vanilla Nerf
            
        Returns:
            same as vanillar Nerf
        """
        # prepare inputs for original Nerf
        camera_hash = kwargs.get('camera_hash', None) # Optional[str]
        camera = kwargs.get('camera') # CamerasBase
        image = kwargs.get('image') # torch.Tensor
        chunk_idx = kwargs.get('chunk_idx') # int
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # Prepare rendering size
        image_height = self.render_height if image_height == None else image_height
        image_width = self.render_width if image_width == None else image_width

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            render_kwargs = {
                "cameras":camera,
                "volumetric_function":self._implicit_function[renderer_pass],
                "chunksize":self._chunk_size_test,
                "chunk_idx":chunk_idx,
                "density_noise_std":(self._density_noise_std if self.training else 0.0),
                "input_ray_bundle":coarse_ray_bundle,
                "ray_weights":coarse_weights,
                "camera_hash":camera_hash,}    # other arguments for the renderer in vanilla Nerf
            if renderer_pass == "coarse":
                (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                    aud_para=aud_para,
                    rect = rect,
                    bg_image = bg_image,
                    image_height = image_height,
                    image_width = image_width,
                    **render_kwargs,)
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                render_kwargs.update({
                    "input_ray_bundle":coarse_ray_bundle,
                    "ray_weights":coarse_weights,
                })

                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    other_kwargs = {
                        'target_images': F.interpolate(
                            image[None].permute(0,3,1,2,), 
                            (image_height, image_width)
                        ).permute(0,2,3,1),
                            #image[..., :3][None],
                        'sampled_rays_xy': torch.cat((
                            ray_bundle_out.xys[...,:1] / self.image_height * image_height,
                            ray_bundle_out.xys[...,1:2] / self.image_width * image_width,
                        ), dim = -1),
                            #ray_bundle_out.xys
                        } # # other arguments for sample_images_at_mc_locs function in vanilla Nerf
                    rgb_gt = sample_images_at_int_locs(
                        **other_kwargs)
                else:
                    rgb_gt = None

            elif renderer_pass == "fine":
                (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                    aud_para=aud_para,
                    bg_image = bg_image,
                    **render_kwargs,)
                rgb_fine = rgb

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {
            "rgb_fine": rgb_fine, 
            "rgb_coarse": rgb_coarse, 
            "rgb_gt": rgb_gt
            }
        if self.visualization:
            # Store the coarse rays/weights only for visualization purposes.
            out["coarse_ray_bundle"] = type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            )
            out["coarse_weights"] = coarse_weights.detach().cpu()

        return out

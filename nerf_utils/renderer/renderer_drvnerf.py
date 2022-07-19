import torch
from typing import List, Optional, Tuple
from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer.cameras import CamerasBase


from nerf_utils.raymarcher.raymarcher_adnerf import EmissionAbsorptionADNeRFRaymarcher
from nerf_utils.raysampler.full_raysampler_adnerf_v2 import FullNeRFRaysampler
from nerf_utils.raysampler.fine_raysampler_nerf import ProbabilisticRaysampler

from .renderer_nerf import RadianceFieldRenderer

class DrivingRadianceFieldRenderer(RadianceFieldRenderer):
    """
    Implements a renderer of a Audio Driven Neural Radiance Field.
    """
    
    def __init__(
        self,
        dim_aud: Optional[int]=None,
        sample_stages: list[str] = ["coarse", "fine"],
        **kwargs
    ):
        """
        Args:
            dim_aud: the dimension of driving signals, e.g. audio features.
            kwargs: the arguments for initializing original nerf model.
        """
        
        super(DrivingRadianceFieldRenderer, self).__init__(**kwargs)
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
        transparent_init = kwargs.get('transparent_init', True)
        return_feat = kwargs.get('return_feat', False)
        
        self.sample_stages = sample_stages
        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()
        
        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionADNeRFRaymarcher()
        
        # Parse out image dimensions.
        self.image_height, self.image_width = image_size

        for render_pass in self.sample_stages:
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
                raysampler = FullNeRFRaysampler(
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
                'transparent_init': transparent_init,
                'return_feat':return_feat} # arguments that original NeuralRadianceField in vanillar Nerf used
            if dim_aud is not None:
                from nerf_utils.implicit_function.adnerf import AudioDrivenNeuralRadianceField
                self._implicit_function[render_pass] = AudioDrivenNeuralRadianceField(
                    dim_aud = dim_aud,  
                    **other_kwargs)
            else:
                from nerf_utils.implicit_function.nerf_basic import NeuralRadianceField
                self._implicit_function[render_pass] = NeuralRadianceField(
                    *other_kwargs
                )

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self._return_feat = return_feat
            
    def forward(
        self,
        aud_para: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor] = None,
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
        image_height = self.image_height if image_height == None else image_height
        image_width = self.image_width if image_width == None else image_width
        
        # Full evaluation pass.
        other_kwargs = {
            'chunksize': self._chunk_size_test,
            'batch_size': camera.R.shape[0]} # arguments that original NeRFRaysampler in vanillar Nerf used
        assert "coarse" in self.sample_stages
        n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
            image_height = image_height,
            image_width = image_width,
            **other_kwargs)

        # Process the chunks of rays.
        other_kwargs = {
            'camera_hash': camera_hash,
            'camera': camera,} # paraameters that original _process_ray_chunk function in vanillar Nerf used, except chunk_idx
        chunk_outputs = [
            self._process_ray_chunk(
                aud_para = aud_para,
                image_height = image_height,
                image_width = image_width,
                mask = mask,
                chunk_idx = chunk_idx,
                bg_image = bg_image, 
                **other_kwargs)
            for chunk_idx in range(n_chunks)]

        # out = {
        #     stage + '_' + item: torch.cat(
        #         [ch_o[k] for ch_o in chunk_outputs],
        #         dim=1,
        #     ).view(-1, *self._image_size, 3)
        #     if chunk_outputs[0][k] is not None
        #     else None
        #     # for k in ("rgb_fine", "rgb_coarse")
        #     for stage in self.sample_stages
        #     for item in ("feat", "rgb", "mask")
        # }

        out = {}
        for stage in self.sample_stages:
            for item in ("feat", "rgb", "mask"):
                key = stage + '_' + item
                if chunk_outputs[0][key] is not None:
                    dim = chunk_outputs[0][key].shape[-1]
                    value = torch.cat([
                        ch_o[key] for ch_o in chunk_outputs
                    ]).view(-1, image_height, image_width, dim)                     
                else:
                    value = None
                out[key] = value
                # coarse_feat, coarse_rgb, coarse_mask
                # fine_feat, fine_rgb, fine_mask

        return out

    def _process_ray_chunk(
        self,
        aud_para: torch.Tensor,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        bg_image: Optional[torch.Tensor] = None,
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
        chunk_idx = kwargs.get('chunk_idx') # int
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        out = {}
        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in self.sample_stages:
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
                (rgb, weights, feat), ray_bundle_out = self._renderer[renderer_pass](
                    aud_para=aud_para,
                    image_height = image_height,
                    image_width = image_width,
                    mask = mask,
                    bg_image = bg_image,
                    return_feat = self._return_feat,
                    **render_kwargs,)
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                render_kwargs.update({
                    "input_ray_bundle":coarse_ray_bundle,
                    "ray_weights":coarse_weights,
                })

                # Store the results for output
                out['coarse_rgb'] = rgb
                out['coarse_mask'] = weights[..., None].sum(dim=-2)
                out['coarse_feat'] = feat


            elif renderer_pass == "fine":
                (rgb, weights, feat), ray_bundle_out = self._renderer[renderer_pass](
                    aud_para=aud_para,
                    image_height = image_height,
                    image_width = image_width,
                    mask = mask,
                    bg_image = bg_image,
                    return_feat = self._return_feat,
                    **render_kwargs,)

                # Store the results for output
                out['fine_rgb'] = rgb
                out['fine_mask'] = weights[..., None].sum(dim=-2)
                out['fine_feat'] = feat
            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        return out

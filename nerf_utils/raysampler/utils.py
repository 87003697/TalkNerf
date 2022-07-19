from matplotlib import use
import torch
from typing import Optional
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.renderer.implicit.raysampling import _jiggle_within_stratas, _safe_multinomial
from torch.nn import functional as F


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    assert (focal[:, 0] == focal[:, 1]).all() # focal_lengh for x and y axis are equal
    scale = -1./(W/(2.*focal[:, 1:, None]))
    o0 = rays_o[..., 0] / rays_o[..., 2] * scale.expand_as(rays_o[..., 0])
    o1 = rays_o[..., 1] / rays_o[..., 2] * scale.expand_as(rays_o[..., 0])
    o2 = 1. + 2. * near / rays_o[..., 2]
    
    d0 = (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2]) * scale.expand_as(rays_o[..., 0])
    d1 = (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2]) * scale.expand_as(rays_o[..., 0])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d

class MultinomialRaysamplerNerf(torch.nn.Module):
    """
    The function the performes similar to the MultinomialRaysampler in pytorch3d
    Except that how to performe camera-to-world projection is different from it, but identical to original Nerf implemetation, see also nerf-pytorch github repo.
    and the image grid change to a dynamic formular 
    """

    def __init__(
        self,
        *,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
        fix_grid: bool = False
    ) : # -> None
        """
        Args:
            same as the MultinomialRaysampler in pytorch3d
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._n_rays_per_image = n_rays_per_image
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling


        # arguments for dynamic grid
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.image_height = image_height
        self.image_width = image_width
        
        # get the initial grid of image xy coords
        self.fix_grid = fix_grid

    def make_grid(self, 
                  image_height: int, 
                  image_width: int): # -> torch.Tensor
        """
        Unlike original MultinomialRaysampler, we would make the _xy_grid changable during training,
        for the purpose of low-scale pretraining
        Args:
            min_x (float): The leftmost x-coordinate of each ray's source pixel's center.
            max_x (float): The rightmost x-coordinate of each ray's source pixel's center.
            min_y (float): The topmost y-coordinate of each ray's source pixel's center.
            max_y (float): The bottommost y-coordinate of each ray's source pixel's center.
            image_height (int): The horizontal size of the image grid.
            image_width (int): The vertical size of the image grid.
        """
        _xy_grid = torch.stack(
            tuple(
                # reversed(
                    meshgrid_ij(
                        torch.linspace(self.min_y, self.max_y, image_height, dtype=torch.float32),
                        torch.linspace(self.min_x, self.max_x, image_width, dtype=torch.float32),
                    # )
                )
            ),
            dim=-1,
        )
        return _xy_grid


    def forward(
        self,
        cameras: CamerasBase,
        *,
        mask: Optional[torch.Tensor] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        n_rays_per_image: Optional[int] = None,
        n_pts_per_ray: Optional[int] = None,
        stratified_sampling: bool = False,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        ndc: bool = False,
        **kwargs,
    ): # -> RayBundle
        """
        Args:
            same as the MultinomialRaysampler in original pytorch3d implementation.
        Returns:
            A named tuple RayBundle
        """
        batch_size = cameras.R.shape[0]
        device = cameras.device

        
        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        image_height = self.image_height if image_height == None else image_height
        image_width = self.image_height if image_height == None else image_height
        
        # xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)
        xy_grid = self.make_grid(image_height, image_width).to(device).expand(batch_size, -1, -1, -1)

        num_rays = n_rays_per_image or self._n_rays_per_image
        if mask is not None and num_rays is None:
            # if num rays not given, sample according to the smallest mask
            num_rays = num_rays or mask.sum(dim=(1, 2)).min().int().item()

        if num_rays is not None:
            if mask is not None:
                assert mask.shape == xy_grid.shape[:3]
                weights = mask.reshape(batch_size, -1)
            else:
                # it is probably more efficient to use torch.randperm
                # for uniform weights but it is unlikely given that randperm
                # is not batched and does not support partial permutation
                _, width, height, _ = xy_grid.shape
                weights = xy_grid.new_ones(batch_size, width * height)
            rays_idx = _safe_multinomial(weights, num_rays)[..., None].expand(-1, -1, 2)

            xy_grid = torch.gather(xy_grid.reshape(batch_size, -1, 2), 1, rays_idx)[
                :, :, None
            ]
            
        min_depth = min_depth if min_depth is not None else self._min_depth
        max_depth = max_depth if max_depth is not None else self._max_depth
        n_pts_per_ray = (
            n_pts_per_ray if n_pts_per_ray is not None else self._n_pts_per_ray
        )
        stratified_sampling = (
            stratified_sampling
            if stratified_sampling is not None
            else self._stratified_sampling
        )

        other_params = {
            'cameras': cameras,
            'xy_grid': xy_grid,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'n_pts_per_ray': n_pts_per_ray,
            'unit_directions': self._unit_directions,
            'stratified_sampling': stratified_sampling,}
        return _xy_to_ray_bundle_(
            image_height=image_height,
            image_width=image_width,
            ndc = ndc,
            **other_params)
        
def _xy_to_ray_bundle_(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depth: float,
    max_depth: float,
    n_pts_per_ray: int,
    unit_directions: bool,
    image_height: int,
    image_width: int,
    stratified_sampling: bool = False,
    ndc: bool = False,
): # -> RayBundle
    """
    The function the performes similar to the _xy_to_ray_bundle in pytorch3d/pytorch3d/renderer/implicit/raysampling.py
    Except that how to performe camera-to-world projection is different from it, but identical to original Nerf implemetation, see also nerf-pytorch github repo.
    Args:
        image_height: the height of rendering output
        image_width: the width of rendering output
        Others ame as _xy_to_ray_bundle in pytorch3d/pytorch3d/renderer/implicit/raysampling.py
    Returns:
        RayBundle
    """

    batch_size = xy_grid.shape[0] # xy_grid.shape = torch.Size([1, 800, 800, 2])
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore # 800 * 800 = 640000
    device = xy_grid.device
    
    rays_zs = xy_grid.new_empty((0,))
    
    # estimate depth
    if n_pts_per_ray > 0:
        # unlinearly interpolate instead of torch.linspace, as is did in nerf-pytorch
        # depths = torch.linspace(min_depth, max_depth, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device,)
        fractions = torch.linspace(0., 1., steps=n_pts_per_ray, dtype=xy_grid.dtype)
        depths = 1./(1./min_depth * (1 - fractions) + 1./max_depth * fractions)
        
        rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray).to(device)
        
        if stratified_sampling:
            rays_zs = _jiggle_within_stratas(rays_zs)
    
    # get ray directions and origins
    i, j = torch.meshgrid(
        torch.linspace(0, image_width-1, image_width), 
        torch.linspace(0, image_height-1, image_height))
    i = i.t()[None].repeat(batch_size, 1, 1).to(device)
    j = j.t()[None].repeat(batch_size, 1, 1).to(device)

    cx = cameras.principal_point[:,0].view(-1, 1, 1)
    cy = cameras.principal_point[:,1].view(-1, 1, 1)
    f_x = cameras.focal_length[:,0].view(-1, 1, 1)
    f_y = cameras.focal_length[:,1].view(-1, 1, 1)
    c2w_R = cameras.R
    c2w_T = cameras.T

    dirs = torch.stack(
        [
        (i-cx.expand_as(i))/f_x.expand_as(i), -(j-cy.expand_as(j))/f_y.expand_as(j), 
        -torch.ones_like(i)
        ], -1)
    
    rays_d = torch.sum(
        dirs.unsqueeze(-2) * \
        c2w_R.unsqueeze(1).unsqueeze(2).repeat(1, dirs.shape[1], dirs.shape[2], 1, 1), 
            -1)
    rays_o = c2w_T.unsqueeze(1).unsqueeze(2).expand(rays_d.shape)

    if ndc:
        rays_o, rays_d = ndc_rays(
            H = image_height, W = image_width, focal = cameras.focal_length, 
            near = -1, rays_o = rays_o, rays_d = rays_d)
        
    return RayBundle(
        rays_o, 
        rays_d, 
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,)
    
    
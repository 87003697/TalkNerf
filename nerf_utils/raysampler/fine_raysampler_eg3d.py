import math
from typing import List

import torch
from pytorch3d.renderer import RayBundle
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf_python

class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.

    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ): #  -> RayBundle

        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.

        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        # Calculate the mid-points between the ray depths.
        z_vals = input_ray_bundle.lengths # B*H*W*N
        batch_size = z_vals.shape[0]

        # Carry out the importance sampling.
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf_python(
            z_vals_mid.view(-1, z_vals_mid.shape[-1]), # torch.Size([8192, 47])
            ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1], # torch.Size([8192, 46])
            self._n_pts_per_ray,
            det=not (
                (self._stratified and self.training)
                or (self._stratified_test and not self.training)
            ),
        ).view(batch_size, *z_vals.shape[1:-1], self._n_pts_per_ray)

        if self._add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )
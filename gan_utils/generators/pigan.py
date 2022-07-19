import torch

def fancy_integration(rgb_sigma,
                      z_vals,
                      noise_std=0.5,
                      last_back=False,
                      white_back=False,
                      clamp_mode='relu',
                      fill_mode=None):
    """
    # modified from CIPS-3d by yangjie
    Performs NeRF volumetric rendering.

    :param rgb_sigma: (b, h x w, num_samples, dim_rgb + dim_sigma)
    :param z_vals: (b, h x w, num_samples, 1)
    :param device:
    :param noise_std:
    :param last_back:
    :param white_back:
    :param clamp_mode:
    :param fill_mode:
    :return:
    - rgb_final: (b, h x w, dim_rgb)
    - depth_final: (b, h x w, 1)
    - weights: (b, h x w, num_samples, 1)
    """
    device = rgb_sigma.device
    rgbs = rgb_sigma[..., :-1]  # (b, h x w, num_samples, c)
    sigmas = rgb_sigma[..., -1:]  # (b, h x w, num_samples, 1)

    # (b, h x w, num_samples - 1, 1)
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])  # (b, h x w, 1, 1)
    deltas = torch.cat([deltas, delta_inf], -2)  # (b, h x w, num_samples, 1)

    noise = torch.randn(sigmas.shape, device=device) * \
        noise_std  # (b, h x w, num_samples, 1)

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        # (b, h x w, num_samples, 1)
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        assert 0, "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(
        alphas[:, :, :1]), 1 - alphas + 1e-10], -2)  # (b, h x w, num_samples + 1, 1)
    # e^(x+y) = e^x + e^y 所以这里使用cumprod。 nerf原文公式（3）
    # (b, h x w, num_samples, 1)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)  # (b, h x w, num_samples, 3)
    depth_final = torch.sum(weights * z_vals, -2)  # (b, h x w, num_samples, 1)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor(
            [1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights
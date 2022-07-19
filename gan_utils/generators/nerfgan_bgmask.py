from audioop import mul
import math
import random
import functools
import operator
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from gan_utils.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from gan_utils.generators.eg3d_gen import PixelNorm, Upsample, Downsample, Blur, EqualConv2d, EqualLinear, ScaledLeakyReLU, ModulatedConv2d,  NoiseInjection, ConstantInput, StyledConv, ToRGB, ConvLayer, ResBlock

class ToMask(nn.Module):
    def __init__(self, 
                 in_channel, 
                 style_dim, 
                 blur_kernel=[1, 3, 3, 1], 
                 device='cpu'):
        super().__init__()

        self.upsample = Upsample(blur_kernel, device=device)
        
        self.style = nn.Parameter(torch.zeros([1, style_dim]))

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False, device=device)

    def forward(self, input, mask_prev=None):
        batch = input.shape[0]
        
        mask = self.conv(
            input, 
            self.style.repeat(batch, 1))

        if mask_prev.shape[3] != mask.shape[3]:
            mask_prev = self.upsample(mask_prev)

        return mask + mask_prev
    
class MaskedStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        isconcat=True,
        device='cpu',
        needs_mask=False
    ):
        super().__init__()
        
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            device=device
        )
        
        self.noise = NoiseInjection(isconcat)

        
        if needs_mask:
            self.to_mask = ToMask(
                out_channel, 
                style_dim, 
                device=device
                )
        self.needs_mask = needs_mask
            
        self.activate = FusedLeakyReLU(
            out_channel, 
            device=device)
    
    def forward(self, input, style, mask, noise=None):
        
        assert noise is not None
        
        input = self.noise(mask * input, noise= (1 - mask) * noise)
        
        drv = self.conv(input, style)
        
        # print('input shape: {}, noise shape: {}, drv shape: {}, mask shape: {} '.format(
        #     input.shape[3], noise.shape[3], drv.shape[3], mask.shape[3]
        # ))

        if self.needs_mask:
            mask = self.to_mask(drv, mask)  
                                
        return drv, mask if self.needs_mask else None
        
            
    
        

class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        size_start,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu',
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        
        self.feat_multiplier = 2 if isconcat else 1  # changes here

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        assert size_start in self.channels
        self.size_start = size_start
        
        self.conv1 = MaskedStyledConv(
            self.channels[size_start], 
            int(self.channels[size_start]*self.feat_multiplier), 
            3, 
            style_dim, 
            needs_mask=True,
            blur_kernel=blur_kernel, 
            isconcat=isconcat, 
            device=device
        )
        
        self.to_rgb1 = ToRGB(
            int(self.channels[size_start]*self.feat_multiplier), 
            style_dim, 
            upsample=False, 
            device=device)
        
        log = lambda x: int(math.log(x, 2))
        self.log_size = log(size)
        self.log_size_start = log(size_start)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[size_start]

        for i in range(self.log_size_start + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                MaskedStyledConv(
                    int(in_channel*self.feat_multiplier),
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    needs_mask=True,
                    blur_kernel=blur_kernel,
                    isconcat=isconcat,
                    device=device
                )
            )

            self.convs.append(
                MaskedStyledConv(
                    int(out_channel*self.feat_multiplier), 
                    out_channel, 
                    3, 
                    style_dim,
                    needs_mask=True,
                    blur_kernel=blur_kernel, 
                    isconcat=isconcat, 
                    device=device
                )
            )

            self.to_rgbs.append(
                ToRGB(
                    int(out_channel*self.feat_multiplier), 
                    style_dim, 
                    device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.device = device
        
    def prepare(
        self,
        styles,
        feat,
        mask,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        assert noise is not None

        assert truncation == 1


        assert len(styles) == 1
        inject_index = self.n_latent
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
        
        out = feat
        out, mask = self.conv1(out, latent[:, 0], noise=noise[0], mask = mask)
        
        skip = self.to_rgb1(out, latent[:, 1])

        return out, skip, latent, mask, \
            zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs), \
            

    def mixup(self, out, skip, latent, mask, mix_zip, ):
        
        i = 1
                   
        for conv1, conv2, noise1, noise2, to_rgb in mix_zip:
            # one iteration results
            out, mask = conv1(out, latent[:, i], mask = mask, noise=noise1)
            out, mask = conv2(out, latent[:, i + 1], mask = mask, noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

            yield skip, mask

    def forward(
        self,
        styles,
        feat,
        mask,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        needs_multi_res = False, 
        needs_mask = False
    ):
        
                
        # first results
        out, skip, latent, mask, mix_zip = self.prepare(
            styles,
            feat,
            mask,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=noise,)
        
        # store results
        image_list = [skip] if needs_multi_res else []
        mask_list = [mask] if needs_multi_res and needs_mask else []
        
        for image, mask in self.mixup(out, skip, latent, mask, mix_zip):
            if needs_multi_res:
                image_list.append(image)
                if needs_multi_res:
                    mask_list.append(mask)
        
        return image_list if  needs_multi_res else [image], \
                mask_list if needs_multi_res and needs_mask else [mask], \
                latent if return_latents else None

class FullGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        size_start,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        audio_dim=16*29,
        narrow=1,
        device='cpu'
    ):
        super().__init__()
        
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        
        log = lambda x: int(math.log(x, 2))    
        self.log_size = log(size)
        self.log_size_start = log(size_start)
        
        self.generator = Generator(
            size, 
            style_dim, 
            n_mlp, 
            channel_multiplier=channel_multiplier, 
            blur_kernel=blur_kernel, 
            lr_mlp=lr_mlp, 
            isconcat=isconcat, 
            narrow=narrow, 
            size_start=size_start,
            device=device)
        
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size_start-1)]
        for i in range(self.log_size ,self.log_size_start, -1):
            out_channel = channels[2 ** (i - 1)]
            #conv = [ResBlock(in_channel, out_channel, blur_kernel)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(audio_dim, style_dim, activation='fused_lrelu', device=device))

    def forward(self,
        inputs,
        audio,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):  
        noise = self.image2attr(inputs)
        outs = self.audio2embed(audio)
        #print(outs.shape)
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        outs = self.generator([outs], return_latents, inject_index, truncation, truncation_latent, input_is_latent, noise=noise)
        return outs
    
    def audio2embed(self, audio):
        audio = audio.view(audio.shape[0], -1)
        outs = self.final_linear(audio)
        return outs

    def image2attr(self, 
                   inputs):
        noise = []
        for i in range(self.log_size - self.log_size_start + 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)   
        return noise
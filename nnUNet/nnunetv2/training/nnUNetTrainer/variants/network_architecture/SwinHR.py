# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Type, Union
from monai.networks.blocks.convolutions import Convolution
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from einops import rearrange
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock, UnetrBasicBlock, UnetResBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from monai.networks.layers import get_act_layer
from monai.utils import look_up_option

rearrange, _ = optional_import("einops", name="rearrange")


class SwinHR(nn.Module):
    """Swin Transformer block is of the Swin UNETR model in MONAI"""

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 6, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.swinViT_forward = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        self.swinViT_forward_attention = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.conv_block3 = UnetResBlock(
            spatial_dims,
            16 * feature_size,
            8 * feature_size,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
        )

        self.conv_block2 = UnetResBlock(
            spatial_dims,
            8 * feature_size,
            4 * feature_size,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
        )
        self.conv_block1 = UnetResBlock(
            spatial_dims,
            4 * feature_size,
            2 * feature_size,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
        )
        self.conv_block0 = UnetResBlock(
            spatial_dims,
            2 * feature_size,
            feature_size,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
        )
        self.conv_block = UnetResBlock(
            spatial_dims,
            2 * feature_size,
            feature_size,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
        )
        self.conv_block_out = UnetResBlock(
            spatial_dims,
            2 * feature_size,
            feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5_attention = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4_attention = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3_attention = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2_attention = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1_attention = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.attention_out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(4608, 4)
        # )
        # # type: ignore

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in, x_attention):

        x0, x1, x2, x3, x4, x5 = self.swinViT_forward(x_in)
        x0_attention, x1_attention, x2_attention, x3_attention, x4_attention, x5_attention = self.swinViT_forward_attention(
            x_attention)
        x4_attention_up = self.decoder5_attention(x4_attention, x3_attention.contiguous())
        x3_attention_up = self.decoder4_attention(x4_attention_up, x2_attention.contiguous())
        x2_attention_up = self.decoder3_attention(x3_attention_up, x1_attention.contiguous())
        x1_attention_up = self.decoder2_attention(x2_attention_up, x0_attention.contiguous())
        x0_attention_up = self.decoder1_attention(x1_attention_up, x5_attention.contiguous())
        attention_out = self.attention_out(x0_attention_up)
        x4_up = self.decoder5(x4, self.conv_block3(torch.cat((x3.contiguous(), x4_attention_up), dim=1)))
        x3_up = self.decoder4(x4_up, self.conv_block2(torch.cat((x2.contiguous(), x3_attention_up), dim=1)))
        x2_up = self.decoder3(x3_up, self.conv_block1(torch.cat((x1.contiguous(), x2_attention_up), dim=1)))
        x1_up = self.decoder2(x2_up, self.conv_block0(torch.cat((x0.contiguous(), x1_attention_up), dim=1)))
        x0_up = self.decoder1(x1_up, self.conv_block(torch.cat((x5.contiguous(), x0_attention_up), dim=1)))
        out = self.out(self.conv_block_out(torch.cat((x0_up, x0_attention_up), dim=1)))
        return out


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            dropout_rate: float = 0.0,
            act: Union[Tuple, str] = "GELU",
            dropout_mode="vit",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fn = get_act_layer(act)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.fn(self.linear2(x))
        x = self.fn(self.linear3(x))
        return x


class Stem(nn.Sequential):
    def __init__(self, dimensions, in_channels, out_channels, if_downsample, norm_mode="instance"):
        super().__init__(
            Convolution(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if if_downsample == False else 3,
                strides=1 if if_downsample == False else 2,
                padding=0 if if_downsample == False else 1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=1,
                padding=1,
                groups=out_channels,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            )

        )


class Stem_out1(nn.Sequential):
    def __init__(self, dimensions, in_channels, out_channels, norm_mode="instance"):
        super().__init__(
            Convolution(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=2,
                padding=1,
                groups=out_channels,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,
                is_transposed=True

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            )

        )


class Stem_out2(nn.Sequential):
    def __init__(self, dimensions, in_channels, out_channels, norm_mode="instance"):
        super().__init__(
            Convolution(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,
                is_transposed=True,

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=1,
                padding=1,
                groups=out_channels,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm=norm_mode,

            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                conv_only=True

            )

        )


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

    """

    def __init__(
            self,
            patch_size: Union[Sequence[int], int] = 2,
            in_chans: int = 1,
            embed_dim: int = 48,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj1 = Stem(
            dimensions=spatial_dims,
            in_channels=in_chans,
            out_channels=embed_dim,
            if_downsample=False
        )
        self.proj2 = Stem(
            dimensions=spatial_dims,
            in_channels=embed_dim,
            out_channels=embed_dim,
            if_downsample=True
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()

        _, _, d, h, w = x_shape
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x_proj1 = self.proj1(x)
        x_proj2 = self.proj2(x_proj1)
        if self.norm is not None:
            x_shape = x.size()
            x_proj2 = x_proj2.flatten(2).transpose(1, 2)
            x_proj2 = self.norm(x_proj2)
            d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
            x_proj2 = x_proj2.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)

        return x_proj1, x_proj2


class H_RLK(nn.Module):
    def __init__(
            self,
            dimensions,
            in_channels,
            out_channels,
            large_kernel,
            if_padding,
            strides,
            if_smallkernel: bool,
            norm_mode=nn.LayerNorm,
    ):
        super().__init__()
        self.if_smallkernel = if_smallkernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.large_kernel = large_kernel
        self.small_kernels = Smallk_list(self.large_kernel) if if_smallkernel == True else []
        self.norm1 = norm_mode(self.in_channels)
        self.norm2 = norm_mode(self.out_channels)
        self.conv1 = Convolution(
            spatial_dims=dimensions,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            strides=1,
            adn_ordering="NDA",
            act=("prelu", {"init": 0.2}),
            norm='instance',
        )
        self.Depthwise_large = nn.Sequential(

            Convolution(
                spatial_dims=dimensions,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm='instance',
            ),
            Convolution(
                spatial_dims=dimensions,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.large_kernel,
                padding=[self.large_kernel // 2 if if_padding == 1 else 0],
                groups=self.out_channels,
                strides=strides,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm='instance',

            ),

            Convolution(
                spatial_dims=dimensions,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                strides=1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                norm='instance',

            ),

        )
        self.reparameters_blocks = nn.ModuleList(
            [
                Convolution(
                    spatial_dims=dimensions,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=small_kernel,
                    padding=small_kernel // 2,
                    groups=self.in_channels,
                    strides=strides,
                    adn_ordering="NDA",
                    act=("prelu", {"init": 0.2}),
                    norm='instance',
                )
                for small_kernel in self.small_kernels
            ]
        )

    def forward(self, x):
        if self.if_smallkernel:
            x = rearrange(x, "n c d h w -> n d h w c")
            x = self.norm1(x)
            x = rearrange(x, "n d h w c-> n c d h w")
            x_base = self.Depthwise_large(x)
            for block in self.reparameters_blocks:
                x_base = x_base + block(x)
            x_base = rearrange(x_base, "n c d h w -> n d h w c")
            x_base = self.norm2(x_base)

            x_base = self.norm2(x_base)
            out = rearrange(x_base, "n d h w c-> n c d h w")
            return out
        else:
            x = rearrange(x, "n c d h w -> n d h w c")
            x = self.norm1(x)
            x = rearrange(x, "n d h w c-> n c d h w")
            x_base = self.Depthwise_large(x)
            x_base = self.conv1(x_base)
            x_base = rearrange(x_base, "n c d h w -> n d h w c")
            x_base = self.norm2(x_base)
            out = rearrange(x_base, "n d h w c-> n c d h w")
            return out


def Smallk_list(large_kernel):
    small_list = []
    if large_kernel == 13:
        small_list.append(7)
        small_list.append(5)
    elif large_kernel == 7:
        small_list.append(5)
    return small_list



def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[0], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # 3, b, heads, n, c_per_head
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop,
                       dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_r = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
            downsample: isinstance = None,  # type: ignore
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: downsample layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            x = rearrange(x, "b d h w c -> b c d h w")
            x = self.downsample(x)

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            window_size: Sequence[int],
            patch_size: Sequence[int],
            depths: Sequence[int],
            num_heads: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
            patch_norm: bool = False,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim tself.mlpo embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.patch_embed_attention = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.patch_merging = nn.ModuleList([
            H_RLK(
                dimensions=3,
                in_channels=embed_dim,
                out_channels=2 * embed_dim,
                # large_kernel=3,
                large_kernel=13,
                if_padding=1,
                strides=2,
                if_smallkernel=True,
            ),
            H_RLK(
                dimensions=3,
                in_channels=2 * embed_dim,
                out_channels=4 * embed_dim,
                large_kernel=13,
                if_padding=1,
                strides=2,
                if_smallkernel=True,
            ),
            H_RLK(
                dimensions=3,
                in_channels=4 * embed_dim,
                out_channels=8 * embed_dim,
                large_kernel=7,
                if_padding=1,
                strides=2,
                if_smallkernel=True,
            ),
            H_RLK(
                dimensions=3,
                in_channels=8 * embed_dim,
                out_channels=16 * embed_dim,
                large_kernel=5,
                if_padding=1,
                strides=2,
                if_smallkernel=False,
            )
        ]
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=self.patch_merging[i_layer],
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def forward(self, x):
        x5, x0 = self.patch_embed(x)
        x1 = self.layers1[0](x0.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        return [x0, x1, x2, x3, x4, x5]


class Up_Concat(nn.Module):
    def __init__(
            self,
            in_channels,
    ):
        super().__init__()
        self.conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            strides=1,
            adn_ordering="NDA",
            act=("prelu", {"init": 0.2}),
            norm='instance',
        )

    def forward(self, up_x, res_x):
        concat_layer = torch.concat((up_x, res_x), dim=1)
        x_add = self.conv(concat_layer)
        return x_add
if __name__=="__main__":
    model=SwinHR(img_size=(32,192,192),in_channels=1, out_channels=3)
    # model=H_RLK(dimensions=3, in_channels=3, out_channels=3, large_kernel=13, if_padding=1, strides=1, if_smallkernel=True)
    # print(summary(model, input_size=(2,1,32,192,192), device="cpu"))
    print(model(torch.ones((2,1,32,192,192))).shape)

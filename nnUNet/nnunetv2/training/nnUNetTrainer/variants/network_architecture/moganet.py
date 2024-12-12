import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN3d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN3d':
        return LayerNorm3d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm3d(embed_dims, eps=1e-5)


class LayerNorm3d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"] 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv3d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv3d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv3d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv3d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv3d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv3d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv3d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv3d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv3d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv3d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool3d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x



class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3], x.shape[4])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv3d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )
    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3], x.shape[4])
        return x, out_size
import torch
import torch.nn as nn
from torchinfo import summary
from torch.amp import autocast
# from mamba_ssm import Mamba
import torch.nn.functional as F






class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(concat)
        x = x * sa
        return x

class CBAMResBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMResBlock, self).__init__()
        self.cbam = CBAM(in_channels, reduction, kernel_size)
    
    def forward(self, x):
        out = self.cbam(x)
        out = out + x  # 残差连接
        return out
class MSCHead_nonMoe_skfv3_cbam(nn.Module):
    def __init__(self, in_channels, out_channels, larger_kernel=7):
        super(MSCHead_nonMoe_skfv3_cbam, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv3d(in_channels, in_channels, (larger_kernel, 1, 1), 1, (larger_kernel // 2, 0, 0))
        self.head3 = nn.Conv3d(in_channels, in_channels, (1, larger_kernel, 1), 1, (0, larger_kernel // 2, 0))
        self.head4 = nn.Conv3d(in_channels, in_channels, (1, 1, larger_kernel), 1, (0, 0, larger_kernel // 2))
        self.head5 = nn.Conv3d(in_channels, in_channels, larger_kernel, 1, larger_kernel // 2, groups=in_channels)
        self.sk = SKFusionv3(in_channels, height=5, kernel_sizes=[3,5,7])
        self.cbam_res = CBAMResBlock(in_channels, reduction=4, kernel_size=7)  # 集成残差 CBAM 模块
        self.out = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.head1(x)  # (B, C, D, H, W)
        x2 = self.head2(x)  # (B, C, D, H, W)
        x3 = self.head3(x)  # (B, C, D, H, W)
        x4 = self.head4(x)  # (B, C, D, H, W)
        x5 = self.head5(x)  # (B, C, D, H, W)
        
        # 特征融合
        x = self.sk([x1, x2, x3, x4, x5])        
        # 应用混合注意力机制
        x = self.cbam_res(x)  # (B, 6C, D, H, W)
        
        # 输出卷积
        x = self.out(x)  # (B, out_channels, D, H, W)
        return x
class MSCHeadv6(nn.Module):
    def __init__(self, in_channels, out_channels, larger_kernel=7):
        super(MSCHeadv6, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv3d(in_channels, in_channels, (larger_kernel, 1, 1), 1, (larger_kernel // 2, 0, 0))
        self.head3 = nn.Conv3d(in_channels, in_channels, (1, larger_kernel, 1), 1, (0, larger_kernel // 2, 0))
        self.head4 = nn.Conv3d(in_channels, in_channels, (1, 1, larger_kernel), 1, (0, 0, larger_kernel // 2))
        self.head5 = nn.Conv3d(in_channels, in_channels, larger_kernel, 1, larger_kernel // 2, groups=in_channels)
        
        # MoE 门控网络
        self.moe_gating = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Conv3d(in_channels * 5, in_channels * 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 2, 5, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  # 在专家维度上进行归一化
        )

        self.sk = SKFusionv3(in_channels, height=5, kernel_sizes=[3,5,7])
        self.cbam_res = CBAMResBlock(in_channels * 2, reduction=4, kernel_size=7)  # 集成残差 CBAM 模块
        self.out = nn.Conv3d(in_channels * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.head1(x)  # (B, C, D, H, W)
        x2 = self.head2(x)  # (B, C, D, H, W)
        x3 = self.head3(x)  # (B, C, D, H, W)
        x4 = self.head4(x)  # (B, C, D, H, W)
        x5 = self.head5(x)  # (B, C, D, H, W)
        
        # MoE 门控权重生成
        concatenated_experts = torch.cat([x1, x2, x3, x4, x5], dim=1)  # (B, 5C, D, H, W)
        gating_weights = self.moe_gating(concatenated_experts)  # (B, 5, 1, 1, 1)
        
        # 应用门控权重
        gating_weights = gating_weights.view(gating_weights.size(0), 5, 1, 1, 1,1)  # (B, 5, 1, 1, 1)
        experts = [x1, x2, x3, x4, x5]
        weighted_experts = torch.stack(experts, dim=1) * gating_weights  # (B, 5, C, D, H, W)
        weighted_experts = torch.sum(weighted_experts, dim=1)  # (B, C, D, H, W)
        
        # 特征融合
        x_fused = self.sk([x1, x2, x3, x4, x5])
        x = torch.cat([weighted_experts, x_fused], dim=1)  # (B, 6C, D, H, W)
        
        # 应用混合注意力机制
        x = self.cbam_res(x)  # (B, 6C, D, H, W)
        
        # 输出卷积
        x = self.out(x)  # (B, out_channels, D, H, W)
        return x


class MSCHeadv5(nn.Module):
    def __init__(self, in_channels, out_channels, larger_kenel=7):
        super(MSCHeadv5, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv3d(in_channels, in_channels, (larger_kenel, 1, 1), 1, (larger_kenel // 2, 0, 0))
        self.head3 = nn.Conv3d(in_channels, in_channels, (1, larger_kenel, 1), 1, (0, larger_kenel // 2, 0))
        self.head4 = nn.Conv3d(in_channels, in_channels, (1, 1, larger_kenel), 1, (0, 0, larger_kenel // 2))
        self.head5 = nn.Conv3d(in_channels, in_channels, larger_kenel, 1, larger_kenel // 2, groups=in_channels)
        self.sk = SKFusionv3(in_channels, height=5, kernel_sizes=[3,5,7])
        self.out = nn.Conv3d(in_channels * 6, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        x5 = self.head5(x)
        x = self.sk([x1, x2, x3, x4, x5])
        x = torch.cat([x1, x2, x3, x4, x5, x], dim=1)
        x = self.out(x)
        return x
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False, device_type="cuda")
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class SKFusionv3(nn.Module):
    def __init__(self, dim=1, height=2, reduction=4, kernel_sizes=[3, 5, 7]):
        super(SKFusionv3, self).__init__()
        
        self.height = height
        self.kernel_sizes = kernel_sizes
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 多尺度卷积层，使用不同的卷积核大小
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(1, self.height, kernel_size, stride=1, padding=kernel_size//2)
            for kernel_size in self.kernel_sizes
        ])
        # 组合最大池化和平均池化
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.combined_pool = nn.Sequential(
            nn.Conv1d(len(kernel_sizes)*self.height, self.height, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.height),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, in_feats):
        B, C, D, H, W = in_feats[0].shape

        # 假设 in_feats 是一个列表，长度为 height，每个元素形状为 (B, C, D, H, W)
        in_feats = torch.cat(in_feats, dim=1)  # (B, C * height, D, H, W)
        in_feats = in_feats.view(B, self.height, C, D, H, W)  # (B, height, C, D, H, W)

        # 特征求和
        feats_sum = torch.sum(in_feats, dim=1)  # (B, C, D, H, W)

        # 全局平均池化和全局最大池化
        avg_attn = self.avg_pool(feats_sum)  # (B, C, 1, 1, 1)
        max_attn = self.max_pool(feats_sum)  # (B, C, 1, 1, 1)
        
        # 结合两个池化的特征
        combined_attn = torch.cat([avg_attn, max_attn], dim=1)  # (B, 2C, 1, 1, 1)
        # combined_attn = avg_attn + max_attn
        combined_attn = combined_attn.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # 使用多个不同卷积核大小的 Conv1d 层进行特征变换
        conv_feats = []
        for conv in self.conv1d_layers:
            feat = conv(combined_attn)  # (B, height, 2C, 1)
            feat = F.relu(feat)
            conv_feats.append(feat)
        
        # 将多个卷积层的输出拼接
        conv_feats = torch.cat(conv_feats, dim=1)  # (B, height * len(kernel_sizes), 2C, 1)
        conv_feats = conv_feats.squeeze(-1)  # (B, height * len(kernel_sizes), 2C)
        
        # 通过 1x1 卷积融合不同尺度的特征
        attn = self.combined_pool(conv_feats)  # (B, height, 2C, 1)
        attn = attn.squeeze(-1)  # (B, height, 2C)
        
        # 生成注意力权重
        attn = attn.view(B, self.height, C, 2)  # 假设 2 是因池化方式数量（avg 和 max）
        attn = torch.mean(attn, dim=-1)  # (B, height, C)
        attn = self.softmax(attn).view(B, self.height, C, 1, 1, 1)  # (B, height, C, 1, 1, 1)

        # 特征加权求和
        out = torch.sum(in_feats * attn, dim=1)  # (B, C, D, H, W)
        return out

class MSCHead_moe(nn.Module):
    def __init__(self, in_channels, out_channels, larger_kernel=7):
        super(MSCHead_moe, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv3d(in_channels, in_channels, (larger_kernel, 1, 1), 1, (larger_kernel // 2, 0, 0))
        self.head3 = nn.Conv3d(in_channels, in_channels, (1, larger_kernel, 1), 1, (0, larger_kernel // 2, 0))
        self.head4 = nn.Conv3d(in_channels, in_channels, (1, 1, larger_kernel), 1, (0, 0, larger_kernel // 2))
        self.head5 = nn.Conv3d(in_channels, in_channels, larger_kernel, 1, larger_kernel // 2, groups=in_channels)
        
        # MoE 门控网络
        self.moe_gating = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Conv3d(in_channels * 5, in_channels * 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 2, 5, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  # 在专家维度上进行归一化
        )
        self.out = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.head1(x)  # (B, C, D, H, W)
        x2 = self.head2(x)  # (B, C, D, H, W)
        x3 = self.head3(x)  # (B, C, D, H, W)
        x4 = self.head4(x)  # (B, C, D, H, W)
        x5 = self.head5(x)  # (B, C, D, H, W)
        
        # MoE 门控权重生成
        concatenated_experts = torch.cat([x1, x2, x3, x4, x5], dim=1)  # (B, 5C, D, H, W)
        gating_weights = self.moe_gating(concatenated_experts)  # (B, 5, 1, 1, 1)
        
        # 应用门控权重
        gating_weights = gating_weights.view(gating_weights.size(0), 5, 1, 1, 1,1)  # (B, 5, 1, 1, 1)
        experts = [x1, x2, x3, x4, x5]
        weighted_experts = torch.stack(experts, dim=1) * gating_weights  # (B, 5, C, D, H, W)
        weighted_experts = torch.sum(weighted_experts, dim=1)  # (B, C, D, H, W)
                
        # 输出卷积
        x = self.out(x)  # (B, out_channels, D, H, W)
        return x
class ConvNeXtConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_rate=4):
        super(ConvNeXtConv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=in_channels,
        )
        self.ln = nn.LayerNorm(in_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0)
        self.conv2 = nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ResNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=2,
    ):
        super().__init__()
        self.stride = stride
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels * expand_rate,
                    in_channels * expand_rate,
                    3,
                    stride,
                    1,
                    groups=in_channels,
                ),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.residual = in_channels == out_channels
        self.act = nn.LeakyReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x = self.conv_list[0](x)
        x = self.conv_list[1](x)
        x = self.conv_list[2](x)
        x = x + res if self.residual and self.stride == 1 else x
        return x


class DenseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=4,
        dropout_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels, in_channels * expand_rate, 1, 1, 0
                ),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels + in_channels * expand_rate,
                    out_channels,
                    1,
                    1,
                    0,
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.residual = in_channels == out_channels
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.residual:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = self.conv_list[1](torch.cat([x, x1], dim=1))
        x2 = self.dp_2(x2)
        x = (
            self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
            if self.residual
            else self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        )
        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        stride=2,
        **kwargs,
    ):
        super().__init__()
        assert num_conv >= 1, "num_conv must be greater than or equal to 1"
        self.downsample_avg = nn.AvgPool3d(stride)
        self.downsample_resnext = ResNeXtConv(in_channels, in_channels, stride=stride)
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels,out_channels,**kwargs)
                    if _ == 0
                    else    MogaBlock(
                            embed_dims=out_channels,
                            ffn_ratio=2,
                            drop_rate=0.1,
                            drop_path_rate=0.1,
                            act_type='GELU',
                            norm_type='BN',
                            init_value=1e-5,
                            attn_dw_dilation=[1, 2, 3],
                            attn_channel_split=[1, 3, 4],
                            attn_act_type='SiLU',
                            attn_force_fp32=False,
                    )

                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x):
        x = self.downsample_avg(x) + self.downsample_resnext(x)
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        fusion_mode="add",
        stride=2,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up_transpose = nn.ConvTranspose3d(
            low_channels, high_channels, stride, stride
        )
        # self.up_transpose = nn.Sequential(
        #     nn.ConvTranspose3d(low_channels, high_channels, stride, stride),
        #     nn.InstanceNorm3d(high_channels, affine=True),
        #     nn.LeakyReLU(inplace=True),
        # )
        in_channels = 2 * high_channels if fusion_mode == "cat" else high_channels
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels, out_channels, **kwargs)
                    if _ == 0
                    else conv(out_channels, out_channels, **kwargs)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x_low, x_high):
        x_low = self.up_transpose(x_low)  # + self.up_interploation(x_low)
        x = (
            torch.cat([x_high, x_low], dim=1)
            if self.fusion_mode == "cat"
            else x_low + x_high
        )
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.1):
        super().__init__()
        self.dp = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv3d(in_channels, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.dp(x)
        p = self.conv1(x)
        return p


class MogaNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        conv=DenseConv,
        channels=[32, 64, 128, 256, 512],
        encoder_num_conv=[2, 2, 2, 2],
        decoder_num_conv=[0, 0, 0, 0],
        encoder_expand_rate=[2, 2, 3, 4],
        decoder_expand_rate=[2, 2, 3, 4],
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)],
        dropout_rate_list=[0.025, 0.05, 0.1, 0.1],
        drop_path_rate_list=[0.025, 0.05, 0.1, 0.1],
        deep_supervision=False,
        predict_mode=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        assert len(channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层
        self.encoders.append(DenseConv(in_channels, channels[0], expand_rate=2))
        self.moga_encoders = nn.ModuleList()
        # 创建编码器层
        for i in range(self.depth):
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                ),
            )
            # self.moga_encoders.append(
            #             MogaBlock(
            #                 embed_dims=channels[i + 1],
            #                 ffn_ratio=2,
            #                 drop_rate=0.1,
            #                 drop_path_rate=0.1,
            #                 act_type='GELU',
            #                 norm_type='BN',
            #                 init_value=1e-5,
            #                 attn_dw_dilation=[1, 2, 3],
            #                 attn_channel_split=[1, 3, 4],
            #                 attn_act_type='SiLU',
            #                 attn_force_fp32=False,
            #         )
            # )


        # 创建解码器层
        for i in range(self.depth):
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    conv=conv,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    dropout_rate=dropout_rate_list[self.depth - i - 1],
                    drop_path_rate=drop_path_rate_list[self.depth - i - 1],
                )
            )
        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )
        # self.out[-1] = MSCHeadv6(channels[0], n_classes, 7)
        # self.out[-1] = MSCHead_moe(channels[0], n_classes, 7)
        # self.out[-1]=MSCHead_nonMoe_skfv3_cbam(channels[0], n_classes, 7)

    def forward(self, x):
        encoder_features = []  # 存储编码器输出
        decoder_features = []  # 存储解码器输出

        # 编码过程
        for index,encoder in enumerate(self.encoders):
            # print(encoder)
            x = encoder(x)
            # if index:
            #     x=self.moga_encoders[index-1](x)
            encoder_features.append(x)

        # 解码过程
        x_dec = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        elif self.predict_mode:
            return self.out[-1](decoder_features[-1])
        else:
            return x_dec, self.out[-1](decoder_features[-1])


if __name__ == "__main__":
    model = MogaNet(1, 4).to("cuda:2")
    x = torch.randn(2, 1, 48, 192, 192).to("cuda:2")
    y = model(x)
    print(y[0].shape)
    # summary(model, input_size=(2, 1, 48, 192, 192), device="cuda:4", depth=5)

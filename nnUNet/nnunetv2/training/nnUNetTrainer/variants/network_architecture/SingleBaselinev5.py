# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.distributions as td
from torch.amp import autocast

# Third-party libraries
from mamba_ssm import Mamba
from einops import rearrange, repeat
import timm
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from timm.models.layers import DropPath, trunc_normal_

# Python standard libraries
import time
import math
import copy
from functools import partial
from typing import Optional, Callable
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops



    

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=0.5,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_core_windows
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core_windows(self, x: torch.Tensor, layer=1):
        return self.forward_corev0(x)
        if layer == 1:
            return self.forward_corev0(x)
        downsampled_4 = F.avg_pool2d(x, kernel_size=2, stride=2)
        processed_4 = self.forward_corev0(downsampled_4)
        processed_4 = processed_4.permute(0, 3, 1, 2)
        restored_4 = F.interpolate(processed_4, scale_factor=2, mode='nearest')
        restored_4 = restored_4.permute(0, 2, 3, 1)
        if layer == 2:
            output = (self.forward_corev0(x) + restored_4) / 2.0

        downsampled_8 = F.avg_pool2d(x, kernel_size=4, stride=4)
        processed_8 = self.forward_corev0(downsampled_8)
        processed_8 = processed_8.permute(0, 3, 1, 2)
        restored_8 = F.interpolate(processed_8, scale_factor=4, mode='nearest')
        restored_8 = restored_8.permute(0, 2, 3, 1)

        output = (self.forward_corev0(x) + restored_4 + restored_8) / 3.0
        return output
        # B C H W
        
        num_splits = 2 ** layer
        split_size = x.shape[2] // num_splits  # Assuming H == W and is divisible by 2**layer

        # Use unfold to create windows
        x_unfolded = x.unfold(2, split_size, split_size).unfold(3, split_size, split_size)
        x_unfolded = x_unfolded.contiguous().view(-1, x.size(1), split_size, split_size)

        # Process all splits at once
        processed_splits = self.forward_corev0(x_unfolded)
        processed_splits = processed_splits.permute(0, 3, 1, 2)
        # Reshape to get the splits back into their original positions and then permute to align dimensions
        processed_splits = processed_splits.view(x.size(0), num_splits, num_splits, x.size(1), split_size, split_size)
        processed_splits = processed_splits.permute(0, 3, 1, 4, 2, 5).contiguous()
        processed_splits = processed_splits.view(x.size(0), x.size(1), x.size(2), x.size(3))
        processed_splits = processed_splits.permute(0, 2, 3, 1)

        return processed_splits


        # num_splits = 2 ** layer
        # split_size = x.shape[2] // num_splits  # Assuming H == W and is divisible by 2**layer
        # outputs = []
        # for i in range(num_splits):
        #     row_outputs = []
        #     for j in range(num_splits):
        #         sub_x = x[:, :, i*split_size:(i+1)*split_size, j*split_size:(j+1)*split_size].contiguous()
        #         processed = self.forward_corev0(sub_x)
        #         row_outputs.append(processed)
        #     # Concatenate all column splits for current row
        #     outputs.append(torch.cat(row_outputs, dim=2))
        # # Concatenate all rows
        # final_output = torch.cat(outputs, dim=1)

        return final_output


    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W

        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        B, H, W, C = x.shape


        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        z = z.permute(0, 3, 1, 2)
        
        z = z.permute(0, 2, 3, 1).contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        y = self.forward_core(x, layer)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        layer: int = 1,
        **kwargs,
    ):
        super().__init__()
        factor = 2.0 
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer
        
    def forward(self, input: torch.Tensor):
        input_x = self.down(input)
        input_x = input_x + self.drop_path(self.self_attention(self.ln_1(input_x), self.layer))
        x = self.up(input_x) + input
        return x
class LayerNormBatchFirst(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        ## x.shape = (B, C, D, H, W) ##
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
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
class Spatial3DAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size,
                                    padding=kernel_size//2,
                                    bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, features):
        # Calculate channel-wise statistics
        avg_pooled = torch.mean(features, dim=1, keepdim=True) 
        max_pooled, _ = torch.max(features, dim=1, keepdim=True)
        
        # Concatenate statistics and compute attention weights
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        attention_weights = self.spatial_conv(pooled_features)
        
        return self.activation(attention_weights)

class TripleLine3DFusion(nn.Module):
    def __init__(self, in_channels, kernel_size=5, reduction_factor=2.0):
        super().__init__()
        
        # Calculate reduced dimension
        reduced_channels = int(in_channels // reduction_factor)
        self.reduced_channels = reduced_channels
        
        # Feature dimension reduction
        self.dimension_reducer = nn.Conv3d(in_channels, reduced_channels, 
                                         kernel_size=1, stride=1)
        self.depth_conv = nn.Sequential(
            nn.Conv3d(reduced_channels, reduced_channels,
                    kernel_size=(1, 1, kernel_size),
                    stride=1, 
                    padding=(0, 0, kernel_size//2),
                    groups=reduced_channels
                    ),
            nn.InstanceNorm3d(reduced_channels),
            nn.GELU()
        )

        self.height_conv = nn.Sequential(
            nn.Conv3d(reduced_channels, reduced_channels, 
                    kernel_size=(1, kernel_size, 1),
                    stride=1,
                    padding=(0, kernel_size//2, 0), 
                    groups=reduced_channels
                    ),
            nn.InstanceNorm3d(reduced_channels),
            nn.GELU()
        )

        self.width_conv = nn.Sequential(
            nn.Conv3d(reduced_channels, reduced_channels,
                    kernel_size=(kernel_size, 1, 1),
                    stride=1,
                    padding=(kernel_size//2, 0, 0),
                    groups=reduced_channels
                    ),
            nn.InstanceNorm3d(reduced_channels),
            nn.GELU()
        )
        # Spatial attention module
        self.attention = Spatial3DAttention(kernel_size=5)
        
        # Residual projection if needed
        self.residual_proj = (nn.Conv3d(
                                reduced_channels, 
                                in_channels, 
                                kernel_size=1, stride=1) 
                                if reduced_channels != in_channels 
                                else nn.Identity()
                            )

    def forward(self, input_features):
        # Reduce feature dimensions
        reduced_features = self.dimension_reducer(input_features)
        
        # Apply planar convolutions
        depth_features = self.depth_conv(reduced_features)
        height_features = self.height_conv(reduced_features) 
        width_features = self.width_conv(reduced_features)
        
        # Fuse planar features
        fused_features = depth_features + height_features + width_features
        
        # Apply spatial attention
        attention_weights = self.attention(fused_features)
        attended_features = fused_features * attention_weights
        
        # Residual connection
        residual_output = self.residual_proj(attended_features)
        output = residual_output + input_features
        
        return output
class ConvNeXtConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_rate=4,stride=1):
        super(ConvNeXtConv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
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


class AdaptiveMeanAndStdPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        mean = self.avg_pool(x)
        squared = self.avg_pool(x * x)
        # sqrt(E(X^2) - E(X)^2)
        std = torch.sqrt(torch.clamp(squared - mean * mean, min=1e-10))
        return mean,std
class ParallelConvBranch(nn.Module):
    def __init__(self, in_channel, kernel_sizes):
        super(ParallelConvBranch, self).__init__()
        
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=k,
                    padding=(k - 1) // 2
                ),
                nn.GELU()
            ) for k in kernel_sizes
        ])
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.conv_branches]
        return torch.sigmoid(torch.stack(branch_outputs, dim=0).sum(dim=0))


class GatedMLP(nn.Module):#统合层间关系，并增强相邻层之间的权重
    def __init__(self, in_features, dropout_rate=0.05,rate =2):
        super().__init__()

        self.gate_branch = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features=in_features,out_features=int(in_features*rate)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=int(in_features*rate),out_features=in_features),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.transpose(-1,-2)
        # x = self.conv_branch(x.unsqueeze(0))
        gate_output = self.gate_branch(x)
        return torch.sigmoid(gate_output.unsqueeze(-1).unsqueeze(-1))
class WithoutChannelSliceAttention(nn.Module):
    def __init__(self, num_slice,in_channel,is_cat=True):
        super().__init__()
 
        self.pools = AdaptiveMeanAndStdPool2d(1)
        self.is_cat = is_cat
        self.shared_conv_avg = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                # nn.BatchNorm2d(in_channel),
                nn.GELU()
            )
        self.shared_conv_std = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                # nn.BatchNorm2d(in_channel),
                nn.GELU()
            )
        if self.is_cat:
            self.down = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel,kernel_size=1),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.GELU()
                )
        self.channel_attention_layer = GatedMLP(in_channel)
        self.slice_attention_layer = GatedMLP(num_slice)

    def forward(self, x):
        mean,std = self.pools(x)
        mean = self.shared_conv_avg(mean)
        std = self.shared_conv_std(std)
        if self.is_cat:
            out = torch.cat([mean,std], dim=1)
            out = self.down(out)
        else:
            out = mean + std
        out = out.squeeze()
        # channel_attention = self.channel_attention_layer(out)
        # channel_attention_result= channel_attention * x
        slice_attention = self.slice_attention_layer(out.transpose(1,0)).transpose(1,0)#计算切片注意力
        slice_attention_result = slice_attention * x
        return slice_attention_result + x
class CatSpceialDualAttention(nn.Module):
    def __init__(self, num_slice,in_channel,is_cat=True):
        super().__init__()
 
        self.pools = AdaptiveMeanAndStdPool2d(1)
        self.is_cat = is_cat
        self.shared_conv_avg = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                # nn.BatchNorm2d(in_channel),
                nn.GELU()
            )
        self.shared_conv_std = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                # nn.BatchNorm2d(in_channel),
                nn.GELU()
            )
        if self.is_cat:
            self.down = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel,kernel_size=1),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.GELU()
                )
        self.channel_attention_layer = GatedMLP(in_channel)
        self.slice_attention_layer = GatedMLP(num_slice)

    def forward(self, x):
        mean,std = self.pools(x)
        mean = self.shared_conv_avg(mean)
        std = self.shared_conv_std(std)
        if self.is_cat:
            out = torch.cat([mean,std], dim=1)
            out = self.down(out)
        else:
            out = mean + std
        out = out.squeeze()
        channel_attention = self.channel_attention_layer(out)
        channel_attention_result= channel_attention * x
        slice_attention = self.slice_attention_layer(out.transpose(1,0)).transpose(1,0)#计算切片注意力
        slice_attention_result = slice_attention * x
        return channel_attention_result + slice_attention_result + x
class SpceialDualAttention(nn.Module):
    def __init__(self, num_slice,in_channel,is_cat=True,kernel_size=4):
        super().__init__()
 
        self.pools = AdaptiveMeanAndStdPool2d(1)
        self.is_cat = is_cat
        self.shared_conv_avg = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU()
            )
        self.shared_conv_std = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU()
            )
        if self.is_cat:
            self.down = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel,kernel_size=1),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
        self.channel_attention_layer = GatedMLP(in_channel)
        self.slice_attention_layer = GatedMLP(num_slice)

    def forward(self, x):
        mean,std = self.pools(x)
        mean = self.shared_conv_avg(mean)
        std = self.shared_conv_std(std)
        if self.is_cat:
            out = torch.cat([mean,std], dim=1)
            out = self.down(out)
        else:
            out = mean + std
        out = out.squeeze()
        channel_attention = self.channel_attention_layer(out)
        channel_attention_result= channel_attention * x
        slice_attention = self.slice_attention_layer(out.transpose(1,0)).transpose(1,0)#计算切片注意力
        slice_attention_result = slice_attention * channel_attention_result
        return slice_attention_result + x
class DenseDualAttention(nn.Module):
    def __init__(self, num_slice,in_channel,is_cat=True,kernel_size=4):
        super().__init__()
 
        self.pools = AdaptiveMeanAndStdPool2d(1)
        self.is_cat = is_cat
        self.shared_conv_avg = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU()
            )
        self.shared_conv_std = nn.Sequential(                
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU()
            )
        if self.is_cat:
            self.down = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel,kernel_size=1),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
        self.channel_attention_layer = GatedMLP(in_channel)
        self.slice_attention_layer = GatedMLP(num_slice)

    def forward(self, x):
        mean,std = self.pools(x)
        mean = self.shared_conv_avg(mean)
        std = self.shared_conv_std(std)
        if self.is_cat:
            out = torch.cat([mean,std], dim=1)
            out = self.down(out)
        else:
            out = mean + std
        out = out.squeeze()
        channel_attention = self.channel_attention_layer(out)
        channel_attention_result= channel_attention * x + x 
        slice_attention = self.slice_attention_layer(channel_attention_result.transpose(1,0)).transpose(1,0)#计算切片注意力
        slice_attention_result = slice_attention.squeeze() * channel_attention_result + channel_attention_result
        return slice_attention_result + x
class DualAttention(nn.Module):
    def __init__(self, num_slice,in_channel,is_ds=True,is_cat=False,kernel_size=4):
        super().__init__()
 
        self.pools = AdaptiveMeanAndStdPool2d(kernel_size)
        self.is_cat = is_cat
        if is_ds:#深度可分离卷积
            self.shared_conv_avg = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size, bias=False,groups=in_channel),
                nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU(),
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
            self.shared_conv_std = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size, bias=False,groups=in_channel),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU(),
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
        else:#普通卷积
            self.shared_conv_avg = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, bias=False),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
            self.shared_conv_std = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, bias=False),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.LeakyReLU()
                )
        if self.is_cat:
            self.down = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel,kernel_size=1),
                # nn.BatchNorm2d(in_channel),#存疑
                nn.GELU()
                )
        self.channel_attention_layer = GatedMLP(in_channel)
        self.slice_attention_layer = GatedMLP(num_slice)

    def forward(self, x):
        mean,std = self.pools(x)
        mean = self.shared_conv_avg(mean)
        std = self.shared_conv_std(std)
        if self.is_cat:
            out = torch.cat([mean,std], dim=1)
            out = self.down(out)
        else:
            out = mean + std
        out = out.squeeze()
        channel_attention = self.channel_attention_layer(out)
        channel_attention_result= channel_attention * x
        slice_attention = self.slice_attention_layer(out.transpose(1,0)).transpose(1,0)#计算切片注意力
        slice_attention_result = slice_attention * channel_attention_result
        return slice_attention_result + x

class SliceAttentionModule(nn.Module):
    def __init__(self,in_features,rate=4,uncertainty=False,rank=5):
        super(SliceAttentionModule,self).__init__()
        self.uncertainty=uncertainty
        self.rank=rank
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=torch.amax(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x*att

class DirectionalMamba(nn.Module):
    """处理单个方向上的所有切片"""
    def __init__(self, d_model, d_state=16,num_slice=64,is_vssb=True,is_slice_attention=False):
        super().__init__()
        self.is_vssb = is_vssb
        if self.is_vssb:
            self.mamba = VSSBlock(hidden_dim=d_model, d_state=d_state)
        else:
            self.mamba = Mamba(d_model=d_model, d_state=d_state)

        if is_slice_attention:
            self.slice_attention = WithoutChannelSliceAttention(num_slice,in_channel=d_model)
            # self.slice_attention = DualAttention(num_slice,in_channel=d_model)
            # self.slice_attention = DenseDualAttention(num_slice,in_channel=d_model)n
        else:
            self.slice_attention = nn.Identity()
        
    def forward(self, slices):
        # slices shape: [num_slices, B, C, H, W]
        num_slices, batch_size, channels, height, width = slices.shape
        
        output = torch.empty_like(slices)
        
        for batch_idx in range(batch_size):
            batch_slices = slices.select(1, batch_idx)  # [num_slices, C, H, W]
            
            if self.is_vssb:
                processed = self.mamba(batch_slices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [num_slices, C,H,W]
            else:
                sequence_length = height * width

                batch_slices = batch_slices.reshape(num_slices, channels, sequence_length)
                batch_slices = batch_slices.transpose(1, 2)  # [num_slices, H*W, C]
                
                # Mamba处理
                processed = self.mamba(batch_slices)  # [num_slices, H*W, C]
                
                # 重塑回2D切片形状
                processed = processed.transpose(1, 2).reshape(num_slices, channels, height, width)
            processed = self.slice_attention(processed)
            
            # 直接写入预分配的tensor
            output[:, batch_idx] = processed
            
            del batch_slices, processed
            
        return output


class TriplaneMamba3D(nn.Module):
    def __init__(self, input_channels,num_slices,is_split,is_res=True,is_proj=False):
        super().__init__()
        
        # 投影层
        self.is_proj=is_proj
        if self.is_proj:
            self.proj_in = nn.Conv3d(input_channels, input_channels, 1)
        
        # 三个方向的处理模块
        self.is_res=is_res
        self.is_split=is_split
        if self.is_split:
            self.mamba_x = DirectionalMamba(d_model=input_channels//2,num_slice=num_slices)
            self.mamba_y = DirectionalMamba(d_model=input_channels//4,num_slice=num_slices)
            self.mamba_z = DirectionalMamba(d_model=input_channels//4,num_slice=num_slices)
        else:
            self.mamba_x = DirectionalMamba(d_model=input_channels,num_slice=num_slices)
            self.mamba_y = DirectionalMamba(d_model=input_channels,num_slice=num_slices)
            self.mamba_z = DirectionalMamba(d_model=input_channels,num_slice=num_slices)
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, 1,stride=1, padding=0),
            nn.InstanceNorm3d(input_channels),
            nn.LeakyReLU()
        )
        # self.fusion = CBAMResBlock(input_channels,kernel_size=5)
        
    def forward(self, x):
        # x shape: [B, C, D, H, W]
        if self.is_proj:
            x = self.proj_in(x)
        _, C, _, _, _ = x.shape
        if self.is_split:
            channel_quarter = C // 4
            # 将输入分成四份
            x_feat = x[:, :channel_quarter*2]  # 1/2通道给x方向
            y_feat = x[:, channel_quarter*2:channel_quarter*3]  # 1/4通道给y方向
            z_feat = x[:, channel_quarter*3:]  # 1/4通道给z方向

            feat_list = []
            
            # X方向
            feat_list.append(self.mamba_x(x_feat.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4))
            
            # Y方向
            feat_list.append(self.mamba_y(y_feat.permute(3, 0, 1, 2, 4)).permute(1, 2, 3, 0, 4))
            
            # Z方向
            feat_list.append(self.mamba_z(z_feat.permute(4, 0, 1, 2, 3)).permute(1, 2, 3, 4, 0))
            
            # feat_list.append(res_feat)
            out = self.fusion(torch.cat(feat_list, dim=1))
        else:
            # 逐方向处理并累积到输出
            feat_list = []
            
            # X方向
            feat_list.append(self.mamba_x(x.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4))
            
            # Y方向
            feat_list.append(self.mamba_y(x.permute(3, 0, 1, 2, 4)).permute(1, 2, 3, 0, 4))
            
            # Z方向
            feat_list.append(self.mamba_z(x.permute(4, 0, 1, 2, 3)).permute(1, 2, 3, 4, 0))
            
            # feat_list.append(res_feat)
            out = self.fusion(feat_list[0]+feat_list[1]+feat_list[2])
        if self.is_res:
            return out + x
        else:
            return out



class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=5):
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
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super(CBAMResBlock, self).__init__()
        self.cbam = CBAM(in_channels, reduction, kernel_size)
    
    def forward(self, x):
        out = self.cbam(x)
        out = out + x  # 残差连接
        return out


class ResNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_rate=2,
        kernel_size=3
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
                    kernel_size,
                    stride,
                    kernel_size//2,
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
                # LayerNormBatchFirst(in_channels),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels, in_channels * expand_rate, 1, 1, 0
                ),
                # nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                # nn.LeakyReLU(inplace=True),
                nn.GELU(),
            )
        )
        temp_in_channels = in_channels + in_channels + in_channels * expand_rate
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                # nn.InstanceNorm3d(out_channels, affine=True),
                # nn.LeakyReLU(inplace=True),
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
        patch_size=64,
        is_split=True,
        **kwargs,
    ):
        super().__init__()
        assert num_conv >= 1, "num_conv must be greater than or equal to 1"
        self.downsample_avg = nn.AvgPool3d(stride)
        self.downsample_resnext = ResNeXtConv(in_channels, in_channels, stride=stride)
        self.tmamba=TriplaneMamba3D(input_channels=in_channels,num_slices=patch_size,is_split=is_split)
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

    def forward(self, x):
        x = self.downsample_avg(x) + self.downsample_resnext(x)
        # x = self.tmamba(x)
        x_down = x
        for extractor in self.extractor:
            x = extractor(x)
        # x = self.tmamba(x)
        return x_down,x


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        patch_size=64,
        fusion_mode="add",
        stride=2,
        is_split=True,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up_transpose = nn.ConvTranspose3d(
            high_channels, high_channels, stride, stride
        )
        in_channels = 2 * high_channels if fusion_mode == "cat" else high_channels
        self.extractor = nn.ModuleList(
            [
                (
                    # TriplaneMamba3D(input_channels=high_channels,num_slices=patch_size,is_split=is_split)
                    # conv(low_channels, high_channels)
                    
                    nn.Sequential(
                        nn.Conv3d(low_channels, high_channels, 1, 1, 0),
                        nn.InstanceNorm3d(high_channels, affine=True),
                        nn.LeakyReLU(inplace=True),
                    )
                    
                    if _ == 0
                    else conv(high_channels, high_channels)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x_low, x_high):
        for extractor in self.extractor:
            x_low = extractor(x_low)
        x = (
            torch.cat([x_high, x_low], dim=1)
            if self.fusion_mode == "cat"
            else x_low + x_high
        )
        x = self.up_transpose(x)
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



class DenseConvDown(nn.Module):
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
        self.res = in_channels == out_channels
        self.dense = stride == 1

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                # LayerNormBatchFirst(in_channels),
                # nn.LeakyReLU(inplace=True),
            )
            
        )
        temp_in_channels = in_channels + in_channels if self.dense else in_channels
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, in_channels * expand_rate, 1, 1, 0),
                # nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                # nn.LeakyReLU(inplace=True),
                nn.GELU(),
            )
        )
        temp_in_channels = (
            in_channels + in_channels + in_channels * expand_rate
            if self.dense
            else in_channels + in_channels * expand_rate
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(temp_in_channels, out_channels, 1, 1, 0),
                # nn.InstanceNorm3d(out_channels, affine=True),
                # nn.LeakyReLU(inplace=True),
            )
        )
        self.dp_1 = nn.Dropout(dropout_rate)
        self.dp_2 = nn.Dropout(dropout_rate * 2)
        self.drop_path = (
            True if torch.rand(1) < drop_path_rate and self.training else False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        if self.drop_path and self.dense:
            return res
        x1 = self.conv_list[0](x)
        x1 = self.dp_1(x1)
        x2 = (
            self.conv_list[1](torch.cat([x, x1], dim=1))
            if self.dense
            else self.conv_list[1](x1)
        )
        x2 = self.dp_2(x2)
        if self.dense and self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1)) + res
        elif self.dense and not self.res:
            x = self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        else:
            x = self.conv_list[2](torch.cat([x1, x2], dim=1))
        return x

class SingleBaselinev5(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        conv=DenseConv,
        channels=[2**i for i in range(5, 10)],
        encoder_num_conv=[1, 1, 1, 1],
        decoder_num_conv=[1, 1, 1, 1],
        encoder_expand_rate=[4] * 4,
        decoder_expand_rate=[4] * 4,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)],
        dropout_rate_list=[0.025, 0.05, 0.1, 0.1],
        drop_path_rate_list=[0.025, 0.05, 0.1, 0.1],
        deep_supervision=False,
        predict_mode=False,
        is_skip=False,
        is_split=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        self.is_skip = is_skip
        assert len(channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层
        self.skips = nn.ModuleList()  # 使用 ModuleList 存储SKIP层
        self.encoders.append(DenseConv(in_channels, channels[0]))
        patch_ini=128
        # 创建编码器层
        for i in range(self.depth):
            patch_ini = int(patch_ini/strides[i][0])
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    patch_size=patch_ini,
                    is_split=is_split,
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                ),
            )

        # 创建解码器层
        for i in range(self.depth):
            patch_ini*=strides[self.depth - i - 1][0]
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    # conv=conv,
                    patch_size=patch_ini,
                    is_split=is_split,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    dropout_rate=dropout_rate_list[self.depth - i - 1],
                    drop_path_rate=drop_path_rate_list[self.depth - i - 1],
                )
            )
        for i in range(self.depth):
            if self.is_skip:
                self.skips.append(
                    TripleLine3DFusion(
                        in_channels = channels[self.depth - i],
                        kernel_size=7
                    )
                )
            else:
                self.skips.append(
                    nn.Identity()
                )
        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )
        

    def forward(self, x):
        encoder_features = []  # 存储编码器输出
        decoder_features = []  # 存储解码器输出

        # 编码过程
        for i,encoder in enumerate(self.encoders):
            if i==0:
                x = encoder(x)
                encoder_features.append([x])
            else:
                x_down,x = encoder(x)
                encoder_features.append([x_down,x])

        for i in range(self.depth+1):
            if i == 0:
                x_down,x_dec =  encoder_features[self.depth-i][0],encoder_features[self.depth-i][1]
                x_dec = self.skips[i](x_dec)
                # print(x_dec.shape)
            elif i == self.depth:
                x_dec = self.decoders[i-1](x_dec, x_down)
                decoder_features.append(x_dec)
                # print(x_dec.shape)
            else:
                x_dec = self.decoders[i-1](x_dec, self.skips[i](x_down))
                x_down = encoder_features[self.depth-i][0]
                decoder_features.append(x_dec)
                # print(x_dec.shape)

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        elif self.predict_mode:
            return self.out[-1](decoder_features[-1])
        else:
            return x_dec, self.out[-1](decoder_features[-1])


if __name__ == "__main__":
    model =SingleBaselinev5(1, 2).to("cuda:2")
    # x = torch.randn(2, 1, 128, 128, 128).to("cuda:6")
    # input = torch.ones((2,1,128,128,128))
    # torch.onnx.export(model, input, f='slice.onnx')   #导出 .onnx 文件
    # netron.start('slice.onnx') #展示结构图
    # y = model(x)
    # import time
    # time.sleep(10000)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (1,128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')

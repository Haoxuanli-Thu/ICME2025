import torch
import torch.nn as nn
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
try:
    from SS3D import SS3D, SS3D_v5, SS3D_v6
except:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models.SS3D import SS3D, SS3D_v5, SS3D_v6
class FeedForward(nn.Module):
    def __init__(self, dim, dropout_rate, hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
class VSSBlock3D(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1,
      **kwargs,
      ):
    super().__init__()
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expansion_factor, **kwargs)
    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    return x
  
class VSSBlock3D_v5(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size = 12,
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v5(d_model=hidden_dim, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  orientation=orientation, 
                                  scan_type=scan_type, 
                                  size=size,
                                  **kwargs)

    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x

class VSSBlock3D_v6(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size = 12,
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v6(d_model=hidden_dim, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  orientation=orientation, 
                                  scan_type=scan_type, 
                                  size=size,
                                  **kwargs)

    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x
  
class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        version = 'v5', #None, v5, v6
        expansion_factor = 1,
        scan_type = 'scan',
        orientation_order = None,
        size = 12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        if version is None:
            print('Vanilla VSS')
            self.blocks = nn.ModuleList([
                VSSBlock3D(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    expansion_factor=expansion_factor,
                )
                for i in range(depth)])
        elif version =='v5':
            print('VSS version 5:')
            if orientation_order is None:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v5(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size = size,
                        orientation=i%6, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in range(depth)])
            else:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v5(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size=size,
                        orientation=i%6, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in orientation_order])
            print()
        elif version =='v6':
            print('VSS version 6:')
            if orientation_order is None:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v6(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size = size,
                        orientation=i%8, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in range(depth)])
            else:
                self.blocks = nn.ModuleList([
                    VSSBlock3D_v6(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size=size,
                        orientation=i%8, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in orientation_order])
            print()
        else:
            raise Exception('define a valid VSS version')
            

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
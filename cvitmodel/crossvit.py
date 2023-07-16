import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub

from timm.models.layers import DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, Block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossAttention(nn.Module):
    def __init__(self, dim=4096, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xy):
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)  # B1C -> B1H(C/H) -> BH1(C/H)
        # x = torch.cat((xy[:, 0:1, ...], x[:, 1:, ...]), dim=1)

        k = self.wk(xy).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(xy).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim=4096, num_heads=12, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.0,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() ##正则化
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xy):
        attn = self.attn(self.norm1(xy), self.norm1(x))
        y = x - self.drop_path(attn)
        if self.has_mlp:
            y = y + self.drop_path(self.mlp(self.norm2(y)))
        return y




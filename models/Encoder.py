# -*- coding: UTF-8 -*-
import copy
import torch
import torch.nn as nn
from functools import partial
from torch.nn.modules.container import ModuleList


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, mask=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        # mask [batch_size, num_patches + 1]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # if mask is not None:
        #     mask_ = mask.unsqueeze(1).unsqueeze(-1)
        #     q = q * mask_
        #     k = k * mask_
        #     v = v * mask_
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if mask is not None:
        #     mask_ = mask.unsqueeze(1).unsqueeze(-2)
        #     attn = attn + mask_
        attn = attn.softmax(dim=-1)

        # if mask is not None:
        #     mask_ = mask.unsqueeze(1).unsqueeze(-2)
        #     attn = mask_ * attn
            # attn[attn!=0] = attn[attn!=0].exp()
            # attn = attn/attn.sum(-1, keepdim=True)

        attn = self.attn_drop(attn)

        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(-2)
            attn = mask_ * attn
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, seq_length=60, embed_dim=300, depth=8, num_heads=6, head_dim=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None, cls=True, end=False, pos=True):
        super(TransformerEncoder, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.cls, self.end, self.pos = cls, end, pos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls else None
        self.end_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if end else None
        self.pos_embed = nn.Parameter(torch.zeros(1, int(cls) + seq_length + int(end), embed_dim)) if pos else None
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.head = nn.Linear(embed_dim, head_dim) if head_dim > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02) if self.pos_embed is not None else None
        nn.init.trunc_normal_(self.cls_token, std=0.02) if self.cls else None
        nn.init.trunc_normal_(self.end_token, std=0.02) if self.end else None
        self.apply(_init_weights)

    def forward_features(self, x, mask=None): # [B, L, F]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) if self.cls else None # [1, 1, F] -> [B, 1, F]
        end_token = self.end_token.expand(x.shape[0], -1, -1) if self.end else None
        # x = torch.cat((cls_token, x, end_token), dim=1)  # [B, L + 1, F]
        x = torch.cat([m for m in (cls_token, x, end_token) if m is not None], dim=1)
        if mask is not None:
            cls_mask = torch.ones(x.shape[0], 1).to(mask.device) if self.cls else None
            end_mask = torch.ones(x.shape[0], 1).to(mask.device) if self.end else None
            mask = torch.cat([m for m in (cls_mask, mask, end_mask) if m is not None], dim=-1)
            # mask = torch.cat((cls_mask, mask), dim=-1)
        x = self.pos_drop(x + (self.pos_embed if self.pos else 0))
        for block in self.blocks:
            x = block(x, mask)
        # x = self.blocks(x, mask)
        x = self.norm(x)
        return x #[:, 0]

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask) #
        x = self.head(x)
        return x


def _init_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



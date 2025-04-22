# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import TransformerEncoder
from .Encoder import DropPath, Mlp, partial, _init_weights, Block, _get_clones, Attention

import numpy as np

def torch_norm(x, dim=-1) -> torch.tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)


class MlpBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        return x


class MlpEncoder(nn.Module):
    def __init__(self, embed_dim=300, depth=8, head_dim=0,
                 mlp_ratio=4.0, drop_ratio=0.,
                 drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MlpBlock(dim=embed_dim, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.head = nn.Linear(embed_dim, head_dim) if head_dim > 0 else nn.Identity()
        self.apply(_init_weights)

    def forward_features(self, x): # [B, L, F]
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x) #
        x = self.head(x)
        return x


class MLPFusion(nn.Module):
    def __init__(self,
                 dim,   cross_num=1,# 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.cross_num = cross_num
        self.mlp = _get_clones(nn.Linear(dim, cross_num * dim, bias=qkv_bias), N=cross_num)
        self.proj = _get_clones(nn.Linear(dim, dim), N=cross_num)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) and not (isinstance(xs, torch.Tensor) and len(xs.size()) == 3):
            xs = [xs]
        B, C = xs[0].size()
        xs_, xs__, x_list = [], [], []
        for i in range(self.cross_num):
            x_ = self.mlp[i](xs[i]).view(B, self.cross_num, C).permute(1, 0, 2)
            xs_.append(x_)
        for i in range(self.cross_num):
            x__ = torch.cat([xs_[j][i].unsqueeze(0) for j in range(self.cross_num)], dim=0)
            xs__.append(x__)

        for i in range(self.cross_num):
            x = xs__[i].sum(0).view(B, C)
            x = self.proj_drop(self.proj[i](x))
            x_list.append(x)
        return x_list
    

class AttentionBlock(nn.Module):
    def __init__(self,
                 dim, cross_num,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attention: nn.Module=MLPFusion,):
        super().__init__()
        self.norm1 = _get_clones(norm_layer(dim), N=cross_num)
        self.attn = attention(dim, cross_num, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = _get_clones(norm_layer(dim), N=cross_num)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _get_clones(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio), N=cross_num)


    def forward(self, x):
        norm_xs, x_list = [], []
        for i, xi in enumerate(x):
            norm_x = self.norm1[i](xi)
            norm_xs.append(norm_x)
        xs = self.attn(norm_xs)
        for i, xi in enumerate(xs):
            # attn = xi
            x_ = x[i] + self.drop_path(xi)
            # x_ = x[i] + self.drop_path((self.gate[i](x[i] + attn) + attn )* x[i])
            x_ = x_ + self.drop_path(self.mlp[i](self.norm2[i](x_)))

            x_list.append(x_)
        return x_list


class GeneralAttentionEncoder(nn.Module):
    def __init__(self, cross_num=3, embed_dim=300, depth=8, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None, attention: nn.Module=MLPFusion):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.norm = _get_clones(norm_layer(embed_dim), N=cross_num)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=embed_dim, cross_num=cross_num, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                norm_layer=norm_layer, act_layer=act_layer, attention=attention)
            for i in range(depth)
        ])

        self.apply(_init_weights)

    def forward(self, *x): # [B, L, F]
        norm_xs, x_list = [], []
        for xi in x:
            xi = self.pos_drop(xi)
            norm_xs.append(xi)
        x = self.blocks(x)
        for i, xi in enumerate(x):
            xi = self.norm[i](xi)
            x_list.append(xi)
        return x_list
    

class AttentionEncoder(nn.Module):
    def __init__(self, seq_length, embed_dim=300, depth=8, num_heads=6, head_dim=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.head = nn.Linear(embed_dim, head_dim) if head_dim > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def forward_features(self, x, mask=None): # [B, L, F]
        x = self.pos_drop(x + self.pos_embed)
        # x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask) #
        x = self.head(x)
        return x


def lq2_loss(out, tgt, w=None):
    b, k = out.size()
    T = 1
    out = out / T
    p = F.softmax(out, dim=-1)
    pt = p.argmax(-1)
    q = torch.zeros_like(pt) + 0.5
    q[pt == tgt] = 1
    # q = p[torch.arange(b), tgt].detach()
    loss = (1 - p[torch.arange(b), tgt] ** q) / q
    if w:
        loss = loss * w
    return loss.mean()


def true_class_probability(out, tgt):
    p = F.softmax(out, dim=-1)
    b = out.size()[-2]
    # print(tgt.size())
    if len(out.size()) == 3:
        return p[:, torch.arange(b), tgt].unsqueeze(-1)
    return p[torch.arange(b), tgt]


def pred_correct(preds, labels):
    return (preds.argmax(-1) == labels).float()



# correctness history class
class History(object):
    def __init__(self, n_data):
        self.correctness = torch.zeros(n_data)
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness):
        data_idx = data_idx.cpu()
        self.correctness[data_idx] += correctness.cpu()

    # max correctness update
    def max_correctness_update(self):
        self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)
        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        cum_correctness1 = self.correctness[data_idx1.cpu()]
        cum_correctness2 = self.correctness[data_idx2.cpu()]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)

        # make target pair
        target1 = cum_correctness1
        target2 = cum_correctness2
        # calc target
        greater = (target1 > target2).float()
        less = (target1 < target2).float() * (-1)

        target = greater + less
        target = target.to(data_idx1.device)
        # calc margin
        margin = abs(target1 - target2)
        margin = margin.to(data_idx1.device)

        return target, margin



if __name__ == '__main__':
    torch.sigmoid(torch.randn(1))


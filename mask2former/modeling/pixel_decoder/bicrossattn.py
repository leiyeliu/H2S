import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class BidirectionalCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads,
            context_dim,
            dropout=0.,
            talking_heads=False,
            prenorm=False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.co_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x,
            context,
            mask=None,
            context_mask=None,
            pos=None,
            context_pos=None,
            return_attn=False,
            rel_pos_bias=None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        qk = self.to_qk(x + pos if pos is not None else x)
        v = self.to_v(x)
        context_qk = self.context_to_qk(context + context_pos if context_pos is not None else context)
        context_v = self.context_to_v(context)

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                                           (qk, context_qk, v, context_v))

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        context_out = self.context_to_out(context_out)
        out = x + self.out_dropout(out)
        context_out = context + self.co_dropout(context_out)
        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out
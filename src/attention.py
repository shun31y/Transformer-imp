import torch
import math
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask to ensure that attention is only applied to previous tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0),
        )
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)  # (B, T, num_heads, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, num_heads, T, 3 * head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each is (B, num_heads, T, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, num_heads, head_dim)
        attn_output = attn_output.view(B, T, C)  # (B, T, d_model)

        out = self.out_proj(attn_output)  # (B, T, d_model)
        out = self.resid_dropout(out)
        return out
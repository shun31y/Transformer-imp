from torch import nn
import torch
from .blocks import TransformerBlock
from torch.nn import functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, block_size, num_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        assert T <= self.block_size
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = self.loss(logits, targets)
        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.block_size:]
            logits, _ = self(input_cond)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def loss(self, logits, targets):
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B*T, V)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            return loss
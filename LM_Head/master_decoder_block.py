import torch.nn as nn
from master_multi_head_attention import MasterMultiHeadAttention
from master_norm import MasterLayerNorm
from master_mlp import MasterMLP


class MasterDecoderBlock(nn.Module):
  def __init__(self, embedding_dim, num_heads, context_length):
    super().__init__()

    self.self_attention = MasterMultiHeadAttention(embedding_dim, embedding_dim, context_length, num_heads, dropout_rate=0.5)
    self.norm1 = MasterLayerNorm(embedding_dim)
    self.mlp = MasterMLP(embedding_dim, embedding_dim)
    self.norm2 = MasterLayerNorm(embedding_dim)

  def forward(self, x):
    # --- Self-Attention block ---
    res = x
    x = self.norm1(x)
    x = self.self_attention(x)
    x = x + res

    # --- MLP block ---
    res = x
    x = self.norm2(x)
    x = self.mlp(x)
    x = x + res

    return x

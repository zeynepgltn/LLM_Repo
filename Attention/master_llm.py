import torch
import torch.nn as nn
from master_self_attention import MasterSelfAttention

def get_rotary_position_encoding(input: torch.Tensor, base: 10000, device="cpu"):
    context_length, dimension = input.shape
    assert dimension % 2 == 0, "Dimension must be even"

    half_dimension = dimension // 2
    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (freqs_indices / dimension))
    position = torch.arange(0, context_length, device=device, dtype=torch.float32)

    #broadcast uyumlu hale getirme
    angles = position[:, None] * freqs[None, :]
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    input_even = input[:, :half_dimension]
    input_odd = input[:, half_dimension:]

    input_rotated_even = input_even * cos_angles - input_odd * sin_angles
    input_rotated_odd  = input_odd  * cos_angles + input_even * sin_angles

    input_rotated = torch.empty_like(input)
    input_rotated[:, :half_dimension] = input_rotated_even
    input_rotated[:, half_dimension:] = input_rotated_odd

    return input_rotated

class MasterLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        self.get_pos = get_rotary_position_encoding
    
        self.self_attention = MasterSelfAttention(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)#sözlük anlamları
        x = self.get_pos(x, base=10000, device=x.device) #position anlamları
        x = self.self_attention(x)
        return x
    
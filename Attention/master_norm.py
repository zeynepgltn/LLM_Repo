import torch
import torch.nn as nn

class MasterLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight=nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weight * x_normalized
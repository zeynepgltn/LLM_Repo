import torch.functional as F
import torch.nn as nn
import torch

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (
      1 + torch.tanh(
          torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        )
    )

class MasterMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(embedding_dim, hidden_dim)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = GELU()

        """ self.layers=nn.Sequential(
             nn.Linear(embedding_dim, hidden_dim),
             GELU(),
             nn.Linear(hidden_dim, embedding_dim)
         ) """
        """ x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x """

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.gelu(gate)
        fuse = gate * up
       
        outputs = self.down_proj(fuse)
       
        return outputs

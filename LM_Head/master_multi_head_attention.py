import torch
import torch.nn as nn


class MasterMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate=0):
        super().__init__()

        self.context_length = context_length

        self.multi_head_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.projection = nn.Linear(embedding_dim, output_dim)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = x.shape

        x = x[:, :self.context_length, :]
        attention_mask = self.mask[:seq_len, :seq_len]

        out, _ = self.multi_head_attention(
            x, x, x,
            attn_mask=attention_mask
        )

        out = self.projection(out)
        return out

  
#   self.heads = nn.ModuleList(
#       [UstaCausalAttention(embedding_dim, output_dim, context_length, dropout_rate) for _ in range(num_heads)]
#     )

#     self.projection = nn.Linear(embedding_dim, output_dim)

#   def forward(self, x):
#     attention_outs = []
#     for head in self.heads:
#       head_out = head(x)
#       attention_outs.append(head_out)

#     attention_out = torch.cat(attention_outs, dim=1)

#     return self.projection(attention_out)
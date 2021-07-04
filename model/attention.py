import torch
from util import *
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    q_k = torch.matmul(query, key.transpose(-2, -1))
    scores = q_k / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    output = F.softmax(scores, dim=-1)
    if dropout is not None:
        output = dropout(output)
    
    return torch.matmul(output, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = \
            [l(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Scaled Dot-Product Attention layer
        x = attention(query, key, value, mask, self.dropout)

        # Concatenate the nhead layers of attention outputs
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)

        # Final linear layer to get the final output
        return self.linears[3](x) 


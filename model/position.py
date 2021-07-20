import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_size, ffn_hid_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(emb_size, ffn_hid_dim)
        self.w_2 = nn.Linear(ffn_hid_dim, emb_size)

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))
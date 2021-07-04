from fetch_tokenizers import PAD_IDX
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x.long()) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_embedding = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).reshape(max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        positional_embedding[:, 0::2] = torch.sin(position * div)
        positional_embedding[:, 1::2] = torch.cos(position * div)
        positional_embedding = positional_embedding.unsqueeze(-2)

        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, x):
        x = x + self.positional_embedding[:x.size(0),:]
        return self.dropout(x)
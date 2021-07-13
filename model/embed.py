import math
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


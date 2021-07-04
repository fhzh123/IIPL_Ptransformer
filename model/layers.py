from util import *
import torch.nn as nn
import torch.nn.functional as F

class SubLayer(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayer, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class EnocderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EnocderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.sublayer = clones(SubLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayer(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)
import torch
from ... import util
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features_size))
        self.b_2 = nn.Parameter(torch.zeros(features_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean)  / (std + self.eps) + self.b_2

class SubLayer(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        attn = sublayer(self.norm(x))
        return x + self.dropout(attn[0])

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
        self.sublayer = util.clones(util.SubLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask.reshape((128, 22))))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = util.clones(util.SubLayer(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, tgt_mask, src_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(memory, memory, x, src_mask))
        return self.sublayer[2](x, self.feed_forward)
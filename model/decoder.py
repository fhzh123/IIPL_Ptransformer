import torch.nn as nn
from util import clones

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, attn, feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = embed_size
        self.attn = attn
        self.ff= feedforward
        self.norm = nn.LayerNorm(embed_size, 1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_mask, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
        x = self.attn(
                 tgt, tgt, tgt, 
                 attn_mask = tgt_mask,
                 key_padding_mask = tgt_key_padding_mask
                 )

        x = self.norm(x + self.dropout(x))

        x = self.attn(
                 memory, memory, tgt,
                 attn_mask = memory_mask,
                 key_padding_mask = memory_key_padding_mask
            )

        x = self.norm(x + self.dropout(x))
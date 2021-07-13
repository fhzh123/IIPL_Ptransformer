import torch.nn as nn
from util import clones

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, attn, feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = embed_size
        self.attn = attn
        self.ff =  feedforward
        self.norm = nn.LayerNorm(embed_size, 1e-6)
        self.dropout = dropout

    def forward(self, src, src_mask, key_padding_mask):
        # MultiheadAttention
        x, _ = self.attn(
                        src, src, src, 
                        attn_mask = src_mask, 
                        key_padding_mask = key_padding_mask
                        )
        
        # Sublayer Connection
        x = self.norm(x + self.dropout(x))

        # FeedForward 
        x = self.ff(x)

        # Sublayer Connection
        x = self.norm(x + self.dropout(x))

        return x


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(Encoder ,self).__init__()
        self.layers = clones(layer, n_layers)

    def forward(self, src, src_mask, key_padding_mask=None):
        for layer in self.layers:
            x = layer(
                src=src, 
                src_mask=src_mask, 
                key_padding_mask=key_padding_mask
                )

        return x
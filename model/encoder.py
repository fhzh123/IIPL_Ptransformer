import torch.nn as nn
from util import clones

class EncoderLayer(nn.Module):
    """
    This would be the Encoder I would be using instead of TransformerEncoderLayer.
    Honestly, they are similar in nature and I am still experimenting the two in between.
    """
    def __init__(self, embed_size, attn, feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = embed_size
        self.attn = attn
        self.ff =  feedforward
        self.norm1 = nn.LayerNorm(embed_size, 1e-6)
        self.norm2 = nn.LayerNorm(embed_size, 1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask, src_key_padding_mask):
        # MultiheadAttention
        attn_output, _ = self.attn(
                        src, src, src, 
                        attn_mask = src_mask, 
                        key_padding_mask = src_key_padding_mask
                        )
                        
        
        # x = [src_seq_len, batch, embed_size]
        
        # Sublayer Connection
        x = self.norm1(src + self.dropout(attn_output))

        # FeedForward 
        attn_output = self.ff(x)

        # Sublayer Connection
        x = self.norm2(x + self.dropout(attn_output))

        return x


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(Encoder ,self).__init__()
        
        # Make N layers of encoder layers.
        self.layers = clones(layer, n_layers)

    def forward(self, src, src_mask, src_key_padding_mask=None):

        x = src

        for layer in self.layers:
            x = layer(
                src=x, 
                src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask
                )

        return x
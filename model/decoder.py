import torch.nn as nn
from util import clones

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, attn, feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = embed_size
        self.attn = attn
        self.ff= feedforward
        self.norm1 = nn.LayerNorm(embed_size, 1e-6)
        self.norm2 = nn.LayerNorm(embed_size, 1e-6)
        self.norm3 = nn.LayerNorm(embed_size, 1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, memory, trg_mask, memory_mask, trg_key_padding_mask, memory_key_padding_mask):
        # MultiheadAttention
        attn_output, _ = self.attn(
                 trg, trg, trg, 
                 attn_mask = trg_mask,
                 key_padding_mask = trg_key_padding_mask
                 )
        
        # x = [trg_seq_len, batch, embed_size]

        # Sublayer Connection
        x = self.norm1(trg + self.dropout(attn_output))

        # MultiheadAttention
        attn_output, _= self.attn(
                 x, memory, memory,
                 key_padding_mask = memory_key_padding_mask
            )
        
        # x = [src_seq_len, batch, embed_size]

        # Sublayer Connection
        x = self.norm2(x + self.dropout(attn_output))

        # FeedForward 
        attn = self.ff(x)

        # Sublayer Connection
        x = self.norm3(x + self.dropout(attn))

        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        
        # Make N layers of encoder layers.
        self.layers = clones(layer, N)

    def forward(self, trg, memory, trg_mask, memory_mask, trg_key_padding_mask, memory_key_padding_mask):
        for layer in self.layers:
            trg = layer(
                trg=trg, 
                memory=memory, 
                trg_mask=trg_mask, 
                memory_mask=memory_mask, 
                trg_key_padding_mask=trg_key_padding_mask, 
                memory_key_padding_mask=memory_key_padding_mask
                )
        
        return trg
import torch.nn as nn
from util import clones


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, attn, feedforward, dropout, device=None):
        super(DecoderLayer, self).__init__()
        self.size = embed_size
        self.attn = attn
        self.ff = feedforward
        self.norm1 = nn.LayerNorm(embed_size, 1e-6, device=device)
        self.norm2 = nn.LayerNorm(embed_size, 1e-6, device=device)
        self.norm3 = nn.LayerNorm(embed_size, 1e-6, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # MultiheadAttention
        attn_output, _ = self.attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )

        # x = [tgt_seq_len, batch, embed_size]

        # Sublayer Connection
        x = self.norm1(tgt + self.dropout(attn_output))

        # MultiheadAttention
        attn_output, _ = self.attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )

        # x = [src_seq_len, batch, embed_size]

        # Sublayer Connection
        x = self.norm2(x + self.dropout(attn_output))

        # FeedForward
        attn_output = self.ff(x)

        # Sublayer Connection
        x = self.norm3(x + self.dropout(attn_output))

        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        # Make N layers of encoder layers.
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, 1e-6)

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return self.norm(tgt), memory

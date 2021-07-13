from util import *
from torch.nn import Module
from model.encoder import *
from model.decoder import *


class Encoder_Decoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, N):
        super(Encoder_Decoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, N)
        self.decoder_layers = clones(decoder_layer, N)
        self.encoder_norm = nn.LayerNorm(encoder_layer.embed_size, 1e-6)
        self.decoder_norm = nn.LayerNorm(decoder_layer.embed_size, 1e-6)
        self.N = N

    def forward(self, src, src_mask, key_padding_mask, tgt, 
                memory, memory_mask, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
        for n in range(self.N):
            src = self.encoder_layers[n](src, src_mask, key_padding_mask)
            #tgt = self.decoder_layers[n](tgt, src, src_mask, tgt_mask)
            tgt = self.decoder_layers[n](tgt,memory, memory_mask, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask)
        return tgt
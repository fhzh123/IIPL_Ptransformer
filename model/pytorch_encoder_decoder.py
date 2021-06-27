from ... import util
from torch.nn import Dropout
from torch.nn import Module
from model.layers import *

class Encoder(Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = util.clones(layer, N)
        self.norm = util.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = util.clones(layer, N)
        self.norm = util.LayerNorm(layer.size)

    def forward(self, tgt, memory, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(tgt, memory, tgt_mask, src_mask)
        return self.norm(x)

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, N):
        super(Encoder_Decoder, self).__init__()
        self.encoder_layers = util.clones(encoder_layer, N)
        self.decoder_layers = util.clones(decoder_layer, N)
        self.encoder_norm = util.LayerNorm(encoder_layer.size)
        self.decoder_norm = util.LayerNorm(decoder_layer.size)
        self.N = N

    def forward(self, src, src_mask, tgt, tgt_mask):
        for n in range(self.N):
            encoder_output = self.encoder_norm(self.encoder_layers[n](src, src_mask))
            decoder_output = self.decoder_norm(self.decoder_layers[n](encoder_output, src_mask, tgt, tgt_mask))
        return decoder_output

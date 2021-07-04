from util import *
from torch.nn import Module
from model.layers import *

class Encoder(Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps=1e-6)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, eps=1e-6)

    def forward(self, tgt, memory, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(tgt, memory, src_mask, tgt_mask)
        return self.norm(x)

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, N):
        super(Encoder_Decoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, N)
        self.decoder_layers = clones(decoder_layer, N)
        self.norm = nn.LayerNorm(encoder_layer.size, eps=1e-6)
        self.N = N

    def forward(self, src, src_mask, tgt, tgt_mask):
        for n in range(self.N):
            # print("from encoder_decoder: {}, {}".format(src_mask, tgt_mask))
            src = self.encoder_layers[n](src, src_mask)
            tgt = self.decoder_layers[n](tgt, src, src_mask, tgt_mask)
        return self.norm(tgt)

from util import *
from torch.nn import Module


class Encoder_Decoder(nn.Module):
     def __init__(self, encoder_layer, decoder_layer, N):
         super(Encoder_Decoder, self).__init__()
         self.encoder_layers = clones(encoder_layer, N)
         self.decoder_layers = clones(decoder_layer, N)
         self.N = N

     def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
        #  for idx in range(self.N):
        #      src = self.encoder_layers[n](src, src_mask, src_key_padding_mask)
        #      tgt = self.decoder_layers[n](tgt, src, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        #  return tgt 

       src_dict = {}

       for idx, layer in enumerate(self.encoder_layers):
         src = layer(src, src_mask, src_key_padding_mask)
         src_dict[idx] = src
         
       for idx, layer in enumerate(self.decoder_layers):
         tgt = layer(tgt, src_dict[idx], tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)

       return tgt
      

     def encode(self, src, src_mask):
       src_dict = {}

       for idx, layer in enumerate(self.encoder_layers):
         src = layer(src, src_mask, None)
         src_dict[idx] = src

       return src_dict

     def decode(self, tgt, memory, tgt_mask):

       for idx, layer in enumerate(self.decoder_layers):
         tgt = layer(tgt, memory[idx], tgt_mask, None, None)

       return tgt
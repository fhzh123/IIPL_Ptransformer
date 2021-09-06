from model.position import PositionWiseFeedForward
from util import clones, get_n_stacks
import torch.nn.functional as F
from torch.nn import Module
import torch.nn as nn
import torch

class Encoder_Decoder(Module):
  """
  This class is the paralell transformer that sends information
  from the encoder to decoder every layer without any linear transformation.

  encoder1 -> decoder1
  encoder2 -> decoder2
  encoder3 -> decoder3
  encoder4 -> decoder4
  encoder5 -> decoder5
  encoder6 -> decoder6
  """

  def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num, device=None):
      super(Encoder_Decoder, self).__init__()
      self.encoder_layers = clones(encoder_layer, encoder_layers_num)
      self.decoder_layers = clones(decoder_layer, decoder_layers_num)
      self.encoder_layers_num = encoder_layers_num
      self.decoder_layers_num = decoder_layers_num
      self.norm = nn.LayerNorm(encoder_layer.size, 1e-6)

  def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
    if self.encoder_layers_num > self.decoder_layers_num:
      n = self.encoder_layers_num - self.decoder_layers_num

      for idx in range(n):
        src = self.encoder_layers[idx](src, src_mask, src_key_padding_mask)

      self.encoder_layers = self.encoder_layers[n:]

    src_list = []
    
    for idx, layer in enumerate(self.encoder_layers):
      src = layer(src, src_mask, src_key_padding_mask)
      src_list.append(src)

    for idx, layer in enumerate(self.decoder_layers):
      tgt = layer(tgt, src_list[idx], tgt_mask, None,
                  tgt_key_padding_mask, memory_key_padding_mask)

    return self.norm(tgt),src_list[-1]

  def encode(self, src, src_mask):
    src_list = []

    for layer in self.encoder_layers:
      src = layer(src, src_mask, None)
      src_list.append(src)

    return src_list

  def decode(self, tgt, memory, tgt_mask):

    for idx, layer in enumerate(self.decoder_layers):
      tgt = layer(tgt, memory[idx], tgt_mask, None, None, None)

    return tgt,_



# class Encoder_Decoder_linear(Module):
#   """
#   This class is the paralell transformer that sends information
#   from the encoder to decoder every layer without any linear transformation.

#   encoder1 -Linear1-> decoder1
#   encoder2 -Linear2-> decoder2
#   encoder3 -Linear3-> decoder3
#   encoder4 -Linear4-> decoder4
#   encoder5 -Linear5-> decoder5
#   encoder6 -Linear6-> decoder6
#   """

#   def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num, device=None):
#       super(Encoder_Decoder_linear, self).__init__()
#       self.encoder_layers = clones(encoder_layer, encoder_layers_num)
#       self.decoder_layers = clones(decoder_layer, decoder_layers_num)
#       self.encoder_layers_num = encoder_layers_num
#       self.decoder_layers_num = decoder_layers_num
#       self.w = clones(FeedForward(
#           encoder_layer.size, encoder_layer.size * 4, encoder_layer.size), encoder_layers_num)
#       self.norm = nn.LayerNorm(encoder_layer.size, 1e-6)

#   def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
#     if self.encoder_layers_num > self.decoder_layers_num:
#       n = self.encoder_layers_num - self.decoder_layers_num

#       for idx in range(n):
#         src = self.encoder_layers[idx](src, src_mask, src_key_padding_mask)

#       self.encoder_layers = self.encoder_layers[n:]

#     src_list = []

#     for idx, layer in enumerate(self.encoder_layers):
#       src = self.w[idx](layer(src, src_mask, src_key_padding_mask))
#       src_list.append(src)

#     for idx, layer in enumerate(self.decoder_layers):
#       tgt = layer(tgt, src_list[idx], tgt_mask, None,
#                   tgt_key_padding_mask, memory_key_padding_mask)

#     return self.norm(tgt)

#   def encode(self, src, src_mask):
#     src_list = []

#     for idx, layer in enumerate(self.encoder_layers):
#       src = self.w[idx](layer(src, src_mask, None))
#       src_list.append(src)

#     return src_list

#   def decode(self, tgt, memory, tgt_mask):

#     for idx, layer in enumerate(self.decoder_layers):
#       tgt = layer(tgt, memory[idx], tgt_mask, None, None, None)

#     return tgt


# class Encoder_Decoder_reverse(Module):
#   """
#   This class is the paralell transformer that sends information
#   from the encoder to decoder every layer without any linear transformation.

#   encoder1 --> decoder6
#   encoder2 --> decoder5
#   encoder3 --> decoder4
#   encoder4 --> decoder3
#   encoder5 --> decoder2
#   encoder6 --> decoder1
#   """

#   def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num, device=None):
#       super(Encoder_Decoder_reverse, self).__init__()
#       self.encoder_layers = clones(encoder_layer, encoder_layers_num)
#       self.decoder_layers = clones(decoder_layer, decoder_layers_num)
#       self.encoder_layers_num = encoder_layers_num
#       self.decoder_layers_num = decoder_layers_num
#       self.norm = nn.LayerNorm(encoder_layer.size, 1e-6)

#   def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
#     if self.encoder_layers_num > self.decoder_layers_num:
#       n = self.encoder_layers_num - self.decoder_layers_num

#       for idx in range(n):
#         src = self.encoder_layers[idx](src, src_mask, src_key_padding_mask)

#       self.encoder_layers = self.encoder_layers[n:]

#     src_list = []

#     for idx, layer in enumerate(self.encoder_layers):
#       src = layer(src, src_mask, src_key_padding_mask)
#       src_list.append(src)

#     for idx in reversed(range(len(self.decoder_layers))):
#       encoder_idx = len(self.decoder_layers) - idx - 1
#       tgt = self.decoder_layers[idx](
#           tgt, src_list[encoder_idx], tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)

#     return self.norm(tgt)

#   def encode(self, src, src_mask):
#     src_list = []

#     for layer in self.encoder_layers:
#       src = layer(src, src_mask, None)
#       src_list.append(src)

#     return src_list

#   def decode(self, tgt, memory, tgt_mask):

#     for idx in reversed(range(len(self.decoder_layers))):
#       encoder_idx = len(self.decoder_layers) - idx - 1
#       tgt = self.decoder_layers[idx](
#           tgt, memory[encoder_idx], tgt_mask, None, None, None)

#     return self.norm(tgt)


# class Encoder_Decoder_reverse_linear(nn.Module):
#   """
#   This class is the paralell transformer that sends information
#   from the encoder to decoder every layer without any linear transformation.

#   encoder1 -Linear-> decoder6
#   encoder2 -Linear-> decoder5
#   encoder3 -Linear-> decoder4
#   encoder4 -Linear-> decoder3
#   encoder5 -Linear-> decoder2
#   encoder6 -Linear-> decoder1
#   """

#   def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num, device=None):
#       super(Encoder_Decoder_reverse_linear, self).__init__()
#       self.encoder_layers = clones(encoder_layer, encoder_layers_num)
#       self.decoder_layers = clones(decoder_layer, decoder_layers_num)
#       self.encoder_layers_num = encoder_layers_num
#       self.decoder_layers_num = decoder_layers_num
#       self.w = clones(FeedForward(
#           encoder_layer.size, encoder_layer.size * 4, encoder_layer.size), encoder_layers_num)
#       self.norm = nn.LayerNorm(encoder_layer.size, 1e-6)

#   def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
#     if self.encoder_layers_num > self.decoder_layers_num:
#       n = self.encoder_layers_num - self.decoder_layers_num

#       for idx in range(n):
#         src = self.encoder_layers[idx](src, src_mask, src_key_padding_mask)

#       self.encoder_layers = self.encoder_layers[n:]

#     src_list = []

#     for idx, layer in enumerate(self.encoder_layers):
#       src = self.w[idx](layer(src, src_mask, src_key_padding_mask))
#       src_list.append(src)

#     for idx in reversed(range(len(self.decoder_layers))):
#       encoder_idx = len(self.decoder_layers) - idx - 1
#       tgt = self.decoder_layers[idx](
#           tgt, src_list[encoder_idx], tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)

#     return self.norm(tgt)

#   def encode(self, src, src_mask):
#     src_list = []

#     for idx, layer in enumerate(self.encoder_layers):
#       src = self.w[idx](layer(src, src_mask, None))
#       src_list.append(src)

#     return src_list

#   def decode(self, tgt, memory, tgt_mask):

#     for idx in reversed(range(len(self.decoder_layers))):
#       encoder_idx = len(self.decoder_layers) - idx - 1
#       tgt = self.decoder_layers[idx](
#           tgt, memory[encoder_idx], tgt_mask, None, None, None)

#     return self.norm(tgt)


# class Encoder_Decoder_concat(Module):
#   """
#   This class is the paralell transformer that sends information
#   from a group of encoders to a group of decoders without any linear transformation.

#   encoder1                      decoder1
#            concat-feedforward1->
#   encoder2                      decoder2

#   encoder3                      decoder3
#            concat-feedforward2->
#   encoder4                      decoder4

#   encoder5                      decoder5
#            concat-feedforward3->
#   encoder6                      decoder6
#   """

#   def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num, device=None):
#     super(Encoder_Decoder_concat, self).__init__()
#     self.encoder_layers_num = encoder_layers_num
#     self.decoder_layers_num = decoder_layers_num

#     self.remaining_encoders = None
#     self.remaining_decoders = None

#     if self.encoder_layers_num > self.decoder_layers_num:
#       remaining_encoder_num = self.encoder_layers_num - self.decoder_layers_num
#       self.encoder_layers_num -= remaining_encoder_num
#       self.remaining_encoders = clones(encoder_layer, remaining_encoder_num)
#     elif self.encoder_layers_num < self.decoder_layers_num:
#       remaining_decoder_num = self.decoder_layers_num - self.encoder_layers_num
#       self.decoder_layers_num -= remaining_decoder_num
#       self.remaining_decoders = clones(decoder_layer, remaining_decoder_num)

#     self.encoder_layer_stacks = get_n_stacks(
#         self.encoder_layers_num, encoder_layer)
#     self.decoder_layer_stacks = get_n_stacks(
#         self.decoder_layers_num, decoder_layer)

#     self.w = nn.ModuleList([])
#     for stack in self.encoder_layer_stacks:
#       self.w.append(
#           FeedForward(len(stack)*encoder_layer.size, len(stack) *
#                       encoder_layer.size*2, encoder_layer.size, device)
#       )

#     self.norm = nn.LayerNorm(encoder_layer.size, 1e-6, device=device)

#     self.device = device

#   def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
#     if self.remaining_encoders:
#       for encoder in self.remaining_encoders:
#         src = encoder(src, src_mask, src_key_padding_mask)

#     idx = 0

#     for encoder_stack, decoder_stack in zip(self.encoder_layer_stacks, self.decoder_layer_stacks):

#       src_list = []

#       for encoder in encoder_stack:

#         src = encoder(src, src_mask, src_key_padding_mask)
#         src_list.append(src)

#       src = torch.cat(src_list, -1)

#       src = self.norm(self.w[idx](src))

#       for decoder in decoder_stack:
#         tgt = decoder(tgt, src, tgt_mask, None,
#                       tgt_key_padding_mask, memory_key_padding_mask)

#       idx += 1

#     if self.remaining_decoders:
#       for decoder in self.remaining_decoders:
#         tgt = decoder(tgt, src, tgt_mask, None,
#                       tgt_key_padding_mask, memory_key_padding_mask)

#     return self.norm(tgt)

#   def encode(self, src, src_mask):
#     if self.remaining_encoders:
#       for encoder in self.remaining_encoders:
#         src = encoder(src, src_mask, None)

#     output_list = []

#     for idx, encoder_stack in enumerate(self.encoder_layer_stacks):
#       src_list = []
#       for encoder in encoder_stack:
#         src = encoder(src, src_mask, None)
#         src_list.append(src)

#       src = self.norm(self.w[idx](torch.cat(src_list, -1)).to(self.device))

#       output_list.append(src)

#     return output_list

#   def decode(self, tgt, memory, tgt_mask):
#     for idx, decoder_stack in enumerate(self.decoder_layer_stacks):
#       for decoder in decoder_stack:
#         tgt = decoder(tgt, memory[idx], tgt_mask, None, None, None)

#     if self.remaining_decoders:
#       for decoder in self.remaining_decoders:
#         tgt = decoder(tgt, memory[-1], tgt_mask, None, None, None)

#     return tgt


# class FeedForward(nn.Module):
#   def __init__(self, input_size, hidden_size, output_size, device=None):
#     super(FeedForward, self).__init__()
#     self.w_1 = nn.Linear(input_size, hidden_size, device=device)
#     self.w_2 = nn.Linear(hidden_size, output_size, device=device)
#     self.norm = nn.LayerNorm(output_size, 1e-6, device=device)

#   def forward(self, x):
#     return self.norm(self.w_2(F.gelu(self.w_1(x))))

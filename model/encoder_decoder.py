from util import clones, get_n_stacks
from torch.nn import Module
import torch.nn as nn
import torch


class Encoder_Decoder_mk1(Module):
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

  def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num):
      super(Encoder_Decoder_mk1, self).__init__()
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

    return self.norm(tgt)

  def encode(self, src, src_mask):
    src_list = []

    for layer in self.encoder_layers:
      src = layer(src, src_mask, None)
      src_list.append(src)

    return src_list

  def decode(self, tgt, memory, tgt_mask):

    for idx, layer in enumerate(self.decoder_layers):
      tgt = layer(tgt, memory[idx], tgt_mask, None, None, None)

    return tgt


class Encoder_Decoder_mk2(Module):
  """
  This class is the paralell transformer that sends information
  from a group of encoders to a group of decoders without any linear transformation.

  encoder1                      decoder1
           concat-feedforward->
  encoder2                      decoder2

  encoder3                      decoder3
           concat-feedforward->
  encoder4                      decoder4

  encoder5                      decoder5
           concat-feedforward->
  encoder6                      decoder6
  """

  def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num):
    super(Encoder_Decoder_mk2, self).__init__()
    self.encoder_layers_num = encoder_layers_num
    self.decoder_layers_num = decoder_layers_num

    self.remaining_encoders = None
    self.remaining_decoders = None

    if self.encoder_layers_num > self.decoder_layers_num:
      remaining_encoder_num = self.encoder_layers_num - self.decoder_layers_num
      self.encoder_layers_num -= remaining_encoder_num
      self.remaining_encoders = clones(encoder_layer, remaining_encoder_num)
    elif self.encoder_layers_num < self.decoder_layers_num:
      remaining_decoder_num = self.decoder_layers_num - self.encoder_layers_num
      self.decoder_layers_num -= remaining_decoder_num
      self.remaining_decoders = clones(decoder_layer, remaining_decoder_num)

    self.encoder_layer_stacks = get_n_stacks(
        self.encoder_layers_num, encoder_layer)
    self.decoder_layer_stacks = get_n_stacks(
        self.decoder_layers_num, decoder_layer)

    self.w = nn.ModuleList([])
    for stack in self.encoder_layer_stacks:
      self.w.append(
          nn.Linear(len(stack) * encoder_layer.size, encoder_layer.size)
      )

    self.norm = nn.LayerNorm(encoder_layer.size, 1e-6)

  def forward(self, src, src_mask, src_key_padding_mask, tgt, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask):
    if self.remaining_encoders:
      for encoder in self.remaining_encoders:
        src = encoder(src, src_mask, src_key_padding_mask)

    idx = 0

    for encoder_stack, decoder_stack in zip(self.encoder_layer_stacks, self.decoder_layer_stacks):

      src_list = []

      for encoder in encoder_stack:
        src = encoder(src, src_mask, src_key_padding_mask)
        src_list.append(src)

      src = torch.cat(src_list, -1)

      src = self.norm(self.w[idx](src))

      for decoder in decoder_stack:
        tgt = decoder(tgt, src, tgt_mask, None,
                      tgt_key_padding_mask, memory_key_padding_mask)

      idx += 1

    if self.remaining_decoders:
      for decoder in self.remaining_decoders:
        tgt = decoder(tgt, src, tgt_mask, None,
                      tgt_key_padding_mask, memory_key_padding_mask)

    return self.norm(tgt)

  def encode(self, src, src_mask, device):
    if self.remaining_encoders:
      for encoder in self.remaining_encoders:
        src = encoder(src, src_mask)

    output_list = []

    for idx, encoder_stack in enumerate(self.encoder_layer_stacks):
      src_list = []
      for encoder in encoder_stack:
        src = encoder(src, src_mask)
        src_list.append(src)

      output_list.append(self.norm(self.w[idx](src).to(device)))

    return output_list

  def decode(self, tgt, memory, tgt_mask):
    for idx, decoder_stack in enumerate(self.decoder_layer_stacks):
      for decoder in decoder_stack:
        tgt = decoder(tgt, memory[idx], tgt_mask)

    if self.remaining_decoders:
      for decoder in self.remaining_decoders:
        tgt = decoder(tgt, memory[-1], tgt_mask)

    return tgt


class Encoder_Decoder_mk3(Module):
  """
  This class is the paralell transformer that sends information
  from the encoder to decoder every layer without any linear transformation.

  encoder1 -Linear-> decoder1
  encoder2 -Linear-> decoder2
  encoder3 -Linear-> decoder3
  encoder4 -Linear-> decoder4
  encoder5 -Linear-> decoder5
  encoder6 -Linear-> decoder6
  """

  def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num):
      super(Encoder_Decoder_mk1, self).__init__()
      self.encoder_layers = clones(encoder_layer, encoder_layers_num)
      self.decoder_layers = clones(decoder_layer, decoder_layers_num)
      self.encoder_layers_num = encoder_layers_num
      self.decoder_layers_num = decoder_layers_num
      self.w = clones(nn.Linear(encoder_layer.size), encoder_layers_num)
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
      src_list.append(self.w[idx](src))

    for idx, layer in enumerate(self.decoder_layers):
      tgt = layer(tgt, src_list[idx], tgt_mask, None,
                  tgt_key_padding_mask, memory_key_padding_mask)

    return self.norm(tgt)

  def encode(self, src, src_mask):
    src_list = []

    for layer in self.encoder_layers:
      src = layer(src, src_mask, None)
      src_list.append(src)

    return src_list

  def decode(self, tgt, memory, tgt_mask):

    for idx, layer in enumerate(self.decoder_layers):
      tgt = layer(tgt, memory[idx], tgt_mask, None, None, None)

    return tgt

import enum
from util import clones
from torch.nn import Module

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

  def forward(self, src, src_mask, src_key_padding_mask, trg, memory_key_padding_mask, trg_mask, trg_key_padding_mask):
    if self.encoder_layers_num > self.decoder_layers_num:
      n = self.encoder_layers_num - self.decoder_layers_num

      for idx in range(n-1):
        src = self.encoder_layers[idx](src, src_mask, src_key_padding_mask)
      
      self.encoder_layers = self.encoder_layers[n:]

    src_list = []

    for idx, layer in enumerate(self.encoder_layers):
      src = layer(src, src_mask, src_key_padding_mask)
      src_list.append(src)
        
    for idx, layer in enumerate(self.decoder_layers):
      trg = layer(trg, src_list[idx], trg_mask, None, trg_key_padding_mask, memory_key_padding_mask)

    return trg
    
  def encode(self, src, src_mask):
    src_list = []

    for idx, layer in enumerate(self.encoder_layers):
      src = layer(src, src_mask, None)
      src_list[idx] = src

    return src_list

  def decode(self, trg, memory, trg_mask):
    
    for idx, layer in enumerate(self.decoder_layers):
      trg = layer(trg, memory[idx], trg_mask, None, None)

    return trg

class Encoder_Decoder_mk2(Module):
  """
  This class is the paralell transformer that sends information
  from a group of encoders to a group of decoders without any linear transformation.

  encoder1 -> decoder1
  encoder2 -> decoder2

  encoder3 -> decoder3
  encoder4 -> decoder4

  encoder5 -> decoder5
  encoder6 -> decoder6
  """
  def __init__(self, encoder_layer, decoder_layer, encoder_layers_num, decoder_layers_num):
    super(Encoder_Decoder_mk2, self).__init__()
    self.encoder_layers_num = encoder_layers_num
    self.decoder_layers_num = decoder_layers_num

    self.encoder_layer_stacks = []
    self.decoder_layer_stacks = []

    self.n_encoder_stack = encoder_layers_num // 3

    if encoder_layers_num % 3 == 0:
      for _ in range(3):
        self.encoder_layer_stacks.append(
            clones(encoder_layer, self.n_encoder_stack)
        )
    else:
      for _ in range(2):
        self.encoder_layer_stacks.append(
            clones(encoder_layer, self.n_encoder_stack)
        )
      self.encoder_layer_stacks.append(
          clones(encoder_layer, encoder_layers_num % 3)
      )

    n_decoder_stack = decoder_layers_num // 3
    if decoder_layers_num % 3 == 0:
      for _ in range(3):
        self.decoder_layer_stacks.append(
            clones(decoder_layer, n_decoder_stack)
        )
    else:
      for _ in range(2):
        self.decoder_layer_stacks.append(
            clones(decoder_layer, n_decoder_stack)
        )
      self.encoder_layer_stacks.append(
          clones(decoder_layer, decoder_layers_num % 3)
      )

  def forward(self, src, src_mask, src_key_padding_mask, trg, memory_key_padding_mask, trg_mask, trg_key_padding_mask):
    if self.encoder_layers_num > self.decoder_layers_num:
      n = self.encoder_layers_num - self.decoder_layers_num
    
    if len(self.n_encoder_stack) <= n:
      for idx in range(n):
        src = self.encoder_layer_stacks[0][idx](src, src_mask, src_key_padding_mask)

      self.encoder_layer_stacks[0] = self.encoder_layer_stacks[0][n:]

    src_list = []

    outputs = []

    for stack in self.encoder_layer_stacks:
      for layer in stack:
        src = layer(src, src_mask, src_key_padding_mask)
        outputs.append(src)

      src_list.append(src)

    for idx, stack in enumerate(self.decoder_layer_stacks):
      curr_srcs = src_list[idx]
      for idx, layer in enumerate(stack):
        trg = layer(trg, curr_srcs[idx], trg_mask, None, trg_key_padding_mask, memory_key_padding_mask)

    return trg

  def encode(self, src, src_mask):
    src_list = []

    outputs = []

    for stack in self.encoder_layer_stacks:
      for layer in stack:
        src = layer(src, src_mask)
        outputs.append(src)

      src_list.append(src)

  def decode(self, trg, memory, trg_mask):

    for idx, stack in enumerate(self.decoder_layer_stacks):
      curr_srcs = memory[idx]
      for idx, layer in enumerate(stack):
        trg = layer(trg, curr_srcs[idx], trg_mask, None, None, None)

    return trg


class Encoder_Decoder_mk3(Module):
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

    self.encoder_layer_stacks = []
    self.decoder_layer_stacks = []

    self.n_encoder_stack = encoder_layers_num // 3

    if encoder_layers_num % 3 == 0:
      for _ in range(3):
        self.encoder_layer_stacks.append(
            clones(encoder_layer, self.n_encoder_stack)
        )
    else:
      for _ in range(2):
        self.encoder_layer_stacks.append(
            clones(encoder_layer, self.n_encoder_stack)
        )
      self.encoder_layer_stacks.append(
          clones(encoder_layer, encoder_layers_num % 3)
      )

    n_decoder_stack = decoder_layers_num // 3
    if decoder_layers_num % 3 == 0:
      for _ in range(3):
        self.decoder_layer_stacks.append(
            clones(decoder_layer, n_decoder_stack)
        )
    else:
      for _ in range(2):
        self.decoder_layer_stacks.append(
            clones(decoder_layer, n_decoder_stack)
        )
      self.encoder_layer_stacks.append(
          clones(decoder_layer, decoder_layers_num % 3)
      )

  def forward(self, src, src_mask, src_key_padding_mask, trg, memory_key_padding_mask, trg_mask, trg_key_padding_mask):
    if self.encoder_layers_num > self.decoder_layers_num:
      n = self.encoder_layers_num - self.decoder_layers_num

    if len(self.n_encoder_stack) <= n:
      for idx in range(n):
        src = self.encoder_layer_stacks[0][idx](
            src, src_mask, src_key_padding_mask)

      self.encoder_layer_stacks[0] = self.encoder_layer_stacks[0][n:]

    src_list = []

    outputs = []

    for idx, stack in enumerate(self.encoder_layer_stacks):
      for layer in stack:
        src = layer(src, src_mask, src_key_padding_mask)
        outputs.append(src)

      src_list.append(src)

    for idx, stack in enumerate(self.decoder_layer_stacks):
      curr_srcs = src_list[idx]
      for idx, layer in enumerate(stack):
        trg = layer(trg, curr_srcs[idx], trg_mask, None,
                    trg_key_padding_mask, memory_key_padding_mask)

    return trg

  def encode(self, src, src_mask):
    src_list = []

    outputs = []

    for idx, stack in enumerate(self.encoder_layer_stacks):
      for layer in stack:
        src = layer(src, src_mask)
        outputs.append(src)

      src_list.append(src)

  def decode(self, trg, memory, trg_mask):

    for idx, stack in enumerate(self.decoder_layer_stacks):
      curr_srcs = memory[idx]
      for idx, layer in enumerate(stack):
        trg = layer(trg, curr_srcs[idx], trg_mask, None, None, None)

    return trg

def get_n_stacks(n_layers):
  assert n_layers > 2, "Must have at least more than 2 layers."
  if n_layers == 3:
    return 1

  
  
  
from util import *
from torch.nn import Module


class Encoder_Decoder(nn.Module):
  def __init__(self, encoder_layer, decoder_layer, N):
      super(Encoder_Decoder, self).__init__()
      self.encoder_layers = clones(encoder_layer, N)
      self.decoder_layers = clones(decoder_layer, N)
      self.N = N

  def forward(self, src, src_mask, src_key_padding_mask, trg, memory_key_padding_mask, trg_mask, trg_key_padding_mask):
    #  for idx in range(self.N):
    #      src = self.encoder_layers[n](src, src_mask, src_key_padding_mask)
    #      trg = self.decoder_layers[n](trg, src, trg_mask, trg_key_padding_mask, memory_key_padding_mask)
    #  return trg 

    #TODO: 원래는 하나씩 인코더에서 디코더로 넘겨주는건데 6개면 2개씩?(예시.) concat해서 넘겨주어서 진행하는 방식으로, 기준을 어떻게 잡는건지 궁금한데.
    # 참고논문: Transformer XL
    # TODO: 저희가 알아야 될거: 인코더나 디코더를 묶는 기준, 어디까지가 구조적인 정보이고, 어디까지가 의미적인 정보인가? 
      # 아이디어1. randomly wired network처럼 가장 optimal한 방법을 알아서 찾아가는 방법?
      # 아이디어2. 

    src_dict = {}

    for idx, layer in enumerate(self.encoder_layers):
      src = layer(src, src_mask, src_key_padding_mask)
      src_dict[idx] = src
        
    for idx, layer in enumerate(self.decoder_layers):
      trg = layer(trg, src_dict[idx], trg_mask, None, trg_key_padding_mask, memory_key_padding_mask)

    return trg
    

  def encode(self, src, src_mask):
    src_dict = {}

    for idx, layer in enumerate(self.encoder_layers):
      src = layer(src, src_mask, None)
      src_dict[idx] = src

    return src_dict

  def decode(self, trg, memory, trg_mask):

    for idx, layer in enumerate(self.decoder_layers):
      trg = layer(trg, memory[idx], trg_mask, None, None)

    return trg

# class temp(nn.Module):
#   def __init__(self, encoder_layer, decoder_layer, N):
#     super(temp, self).__init__()
#     self.encoder_layers = clones(encoder_layer, N)
#     self.decoder_layers = clones(decoder_layer, N)
#     self.N = N
#     self.div = N // 3
  
#   def forward(self, src, src_mask, src_key_padding_mask, trg, memory_key_padding_mask, trg_mask, trg_key_padding_mask):
#     src_dict = {}
#     src_dicts = []
#     count = 0

#     for idx, layer in enumerate(self.encoder_layers):
#       src = layer(src, src_mask, src_key_padding_mask)
      
#       if count <= 1:
#         src_dict[idx] = src
#       else:
#         src_dicts[]

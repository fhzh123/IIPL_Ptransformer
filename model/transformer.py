import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embed import Embedding
from model.attention import MultiHeadAttention
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.encoder_decoder import Encoder_Decoder_mk1
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from model.position import PositionWiseFeedForward, PositionalEncoding

class IIPL_Transformer(nn.Module):
  def __init__(self,
               num_encoder_layers,
               num_decoder_layers,
               emb_size,
               nhead,
               src_vocab_size,
               trg_vocab_size,
               dim_feedforward = 2048,
               dropout: float = 0.1):
    super(IIPL_Transformer, self).__init__()
    self.src_tok_emb = Embedding(src_vocab_size, emb_size)
    self.trg_tok_emb = Embedding(trg_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
    self.generator = nn.Linear(emb_size, trg_vocab_size)
    # self.attn = MultiHeadAttention(emb_size, nhead, dropout)
    self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
    # self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
    self.encoder = Encoder(
      TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_encoder_layers
      # EncoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout), num_encoder_layers
      )
    self.decoder = Decoder(
      TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_decoder_layers
      # DecoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout), num_decoder_layers
      )

  def forward(self,
              src,
              trg,
              src_mask,
              trg_mask,
              src_key_padding_mask,
              trg_key_padding_mask,
              memory_key_padding_mask):
    outs = self.decode(trg, self.encode(src, src_mask, src_key_padding_mask), trg_mask, None, trg_key_padding_mask, memory_key_padding_mask)
    return F.log_softmax(self.generator(outs), dim=-1)
        
  def encode(self, src, src_mask, src_key_padding_mask=None):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    return self.encoder(src_emb, src_mask, src_key_padding_mask)

  def decode(self, trg, src, trg_mask, memory_mask=None, trg_key_padding_mask=None, memory_key_padding_mask=None):
    trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
    return self.decoder(trg_emb, src, trg_mask, memory_mask, trg_key_padding_mask, memory_key_padding_mask)

class IIPL_P_Transformer(nn.Module):
  def __init__(self,
               num_encoder_layers,
               num_decoder_layers,
               emb_size,
               nhead,
               src_vocab_size,
               trg_vocab_size,
               dim_feedforward = 512,
               dropout: float = 0.1):
    super(IIPL_P_Transformer, self).__init__()
    self.src_tok_emb = Embedding(src_vocab_size, emb_size)
    self.trg_tok_emb = Embedding(trg_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
    self.generator = nn.Linear(emb_size, trg_vocab_size)
    self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
    self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
    self.encoder_decoder = Encoder_Decoder_mk1(
      # TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),
      # TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),
      EncoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout),
      DecoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout),
      num_decoder_layers
      )
    
  def forward(self,
              src,
              trg,
              src_mask,
              trg_mask,
              src_key_padding_mask,
              trg_key_padding_mask,
              memory_key_padding_mask):

    outs = self.encode_decode(
        src=src,
        trg=trg,
        src_mask=src_mask,
        trg_mask=trg_mask,
        src_key_padding_mask=src_key_padding_mask,
        trg_key_padding_mask=trg_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask
        )

    return F.log_softmax(self.generator(outs), dim=-1)

  def encode_decode(self, src, trg, src_mask, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
    return self.encoder_decoder(
                        src=src_emb,
                        trg=trg_emb,
                        src_mask=src_mask,
                        trg_mask=trg_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        trg_key_padding_mask=trg_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask
                        )

  def encode(self, src, src_mask):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    return self.encoder_decoder.encode(src_emb, src_mask)

  def decode(self, trg, src, trg_mask):
    trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
    return self.encoder_decoder.decode(trg_emb, src, trg_mask)

def build_model(num_layers,
                emb_size,
                nhead,
                src_vocab_size,
                trg_vocab_size,
                dim_feedforward=2048,
                dropout=0.1,
                variation=False,
                load=False,
                device=None):
  if variation:
    print("\nBuilding P-Transformer..")
    model = IIPL_P_Transformer(
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        nhead=nhead,
        emb_size=emb_size,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
  else:
    print("\nBuilding Original Transformer")
    model = IIPL_Transformer(
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        nhead=nhead,
        emb_size=emb_size,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
  if load:
    model.load_state_dict(torch.load('./data/checkpoints/checkpoint.pth', map_location=device))
  else:
    for p in model.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  model = model.to(device)

  return model


  
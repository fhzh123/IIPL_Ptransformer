import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.position import PositionWiseFeedForward
from model.embed import Embedding, PositionalEncoding
from model.encoder_decoder import Encoder_Decoder

class IIPL_Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout,
                 device=None):
      super(IIPL_Transformer, self).__init__()
      self.generator = nn.Linear(emb_size, tgt_vocab_size)
      self.src_tok_emb = Embedding(src_vocab_size, emb_size)
      self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
      # self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
      self.attn = MultiHeadAttention(emb_size, nhead, dropout)
      self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
      self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
      # self.encoder = TransformerEncoder(
      self.encoder = Encoder(
          # TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_encoder_layers
          EncoderLayer(emb_size, copy.deepcopy(self.attn),copy.deepcopy(self.ff), dropout), num_encoder_layers
        )
      # self.decoder = TransformerDecoder(
      self.decoder = Decoder(
          # TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_decoder_layers
          DecoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout), num_decoder_layers
      )

    def forward(self,
                src,
                tgt,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask):
      outs, mem = self.decode(tgt, self.encode(src, src_mask, src_key_padding_mask),
                          tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)
      return F.log_softmax(self.generator(outs), dim=-1), mem

    def encode(self, src, src_mask, src_key_padding_mask=None):
      src_emb = self.positional_encoding(self.src_tok_emb(src))
      return self.encoder(src_emb, src_mask, src_key_padding_mask)

    def decode(self, tgt, src, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
      tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
      return self.decoder(tgt_emb, src, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

class IIPL_P_Transformer(nn.Module):
  def __init__(
              self,
              num_encoder_layers,
              num_decoder_layers,
                emb_size,
                nhead,
                src_vocab_size,
                tgt_vocab_size,
                dim_feedforward,
                dropout,
                device=None):
      super(IIPL_P_Transformer, self).__init__()
      self.generator = nn.Linear(emb_size, tgt_vocab_size)
      self.src_tok_emb = Embedding(src_vocab_size, emb_size)
      self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
      # self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
      self.attn = MultiHeadAttention(emb_size, nhead, dropout)
      self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
      self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
        
      self.encoder_decoder = Encoder_Decoder(
      # TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_encoder_layers
      EncoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout),
      # TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout, activation='gelu'),num_decoder_layers
      DecoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), dropout),
      num_encoder_layers, num_decoder_layers
      )

  def forward(self,
              src,
              tgt,
              src_mask,
              tgt_mask,
              src_key_padding_mask,
              tgt_key_padding_mask,
              memory_key_padding_mask):
      outs,encoder_out = self.encoder_decoder(self.positional_encoding(self.src_tok_emb(src)), src_mask, src_key_padding_mask, self.positional_encoding(self.tgt_tok_emb(tgt)), memory_key_padding_mask, tgt_mask, tgt_key_padding_mask)
      return F.log_softmax(self.generator(outs), dim=-1),encoder_out

  def encode(self, src, src_mask, src_key_padding_mask=None, device=None):
      src_emb = self.positional_encoding(self.src_tok_emb(src))
      return self.encoder_decoder.encode(src_emb, src_mask)

  def decode(self, tgt, src, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
      tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
      return self.encoder_decoder.decode(tgt_emb, src, tgt_mask)

  
def build_model(num_layers,
                emb_size,
                nhead,
                src_vocab_size,
                tgt_vocab_size,
                dim_feedforward=2048,
                dropout=0.1,
                isP=False,
                device=None):
  if isP:
    print("\nBuilding P-Transformer..")
    model = IIPL_P_Transformer(
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        nhead=nhead,
        emb_size=emb_size,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
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
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  model = model.to(device)

  return model


  

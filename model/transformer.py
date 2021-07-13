import copy
import torch.nn as nn
from torch.nn import Transformer
from model.embed import Embedding
from model.position import PositionWiseFeedForward, PositionalEncoding
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer

class IIPL_Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(IIPL_Transformer, self).__init__()
        self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
        self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            EncoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), copy.deepcopy(self.dropout)), num_encoder_layers
        )
        self.decoder = Decoder(
            DecoderLayer(emb_size, copy.deepcopy(self.attn), copy.deepcopy(self.ff), copy.deepcopy(self.dropout)), num_decoder_layers
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src,
                tgt,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        outs = self.decode(
                        tgt=tgt, 
                        memory=self.encode(
                                    src=src, 
                                    src_mask=src_mask, 
                                    src_padding_mask=src_padding_mask
                                    ), 
                        memory_mask=None, 
                        memory_key_padding_mask=memory_key_padding_mask, 
                        tgt_mask=tgt_mask, 
                        tgt_padding_mask=tgt_padding_mask
                        )
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.encoder(
                        src=src_emb, 
                        src_mask=src_mask, 
                        src_padding_mask=src_padding_mask
                        )

    def decode(self, tgt, memory, memory_mask, memory_key_padding_mask, tgt_mask, tgt_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.decoder(
                        tgt=tgt_emb, 
                        memory=memory, 
                        memory_mask=None, 
                        memory_key_padding_mask=memory_key_padding_mask, 
                        tgt_mask=tgt_mask, 
                        tgt_padding_mask=tgt_padding_mask
                        )

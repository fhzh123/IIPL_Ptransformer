from model.embed import *
from torch.nn import Transformer
from model.position import *
from model.encoder import *
from model.decoder import *
import copy


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward = 512,
                 dropout = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.attn = nn.MultiheadAttention(emb_size, nhead, dropout)
        self.ff = PositionWiseFeedForward(emb_size, dim_feedforward)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_encoder=Encoder(EncoderLayer(
                                                            emb_size,
                                                            copy.deepcopy(self.attn),
                                                            copy.deepcopy(self.ff),
                                                            dropout
                                            ), 
                                       num_encoder_layers),
                                       custom_decoder=Decoder(DecoderLayer(
                                                            emb_size,
                                                            copy.deepcopy(self.attn),
                                                            self.deepcopy(self.ff),
                                                            dropout
                                            ),
                                        num_decoder_layers)
                                       )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
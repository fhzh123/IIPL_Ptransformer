from catalogue import create
from torch import nn
import torch.nn as nn
from copy import deepcopy as dc
from model.pytorch_encoder_decoder import *
from model.attention import *
from model.position import *
from util import *

class o_transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, device):
        super(o_transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.device = device

    def forward(self, src, tgt):
        src_mask, tgt_mask = create_mask(
                                    src=src, 
                                    tgt=tgt, 
                                    device=self.device)

        return self.generator(
            self.decode(
                memory=self.encode(src, src_mask), 
                src_mask=src_mask, 
                tgt=tgt, 
                tgt_mask=tgt_mask)
            )

    def encode(self, src, src_mask):
        return self.encoder(
            x=self.src_embed(src), 
            src_mask=src_mask
            )

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(
            x=self.tgt_embed(tgt), 
            memory=memory, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask)

class p_transformer(nn.Module):
    def __init__(self, encoder_decoder, src_embed, tgt_embed, generator, device):
        super(p_transformer, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.device = device 

    def forward(self, src, tgt):
        src_mask, tgt_mask = create_mask(
                            src=src, 
                            tgt=tgt, 
                            device=self.device)

        return self.generator(
            self.encode_decode(
                src=src, 
                src_mask=src_mask, 
                tgt=tgt, 
                tgt_mask=tgt_mask
                )
            )

    def encode_decode(self, src, src_mask, tgt, tgt_mask):
        return self.encoder_decoder(
            src=self.src_embed(src), 
            src_mask=src_mask, 
            tgt=self.tgt_embed(tgt), 
            tgt_mask=tgt_mask
            )


def build_model(vocabs, nhead, d_model, d_ff, N, device, dropout=0.1, variation=False, load=False):
    # attn = MultiHeadAttention(nhead, d_model, dropout)
    attn = nn.MultiheadAttention(d_model, nhead, dropout, device=device)
    feedforward = PositionWiseFeedForward(d_model, d_ff)
    position = PositionalEncoding(d_model, dropout)
    generator = nn.Linear(d_model, len(vocabs['tgt_lang']))
    if not variation:
        model = o_transformer(Encoder(
                                EncoderLayer(d_model, dc(attn), dc(feedforward), dropout), N
                                ), 
                              Decoder(
                                  DecoderLayer(d_model, dc(attn), dc(attn), dc(feedforward), dropout), N
                                  ),
                              nn.Sequential(
                                  Embeddings(d_model, len(vocabs['src_lang'])), 
                                  dc(position)
                                  ),
                              nn.Sequential(
                                  Embeddings(d_model, len(vocabs['tgt_lang'])), 
                                  dc(position)
                                  ),
                              generator, 
                              device
                              )
    else:
        model = p_transformer(Encoder_Decoder(
                                              EncoderLayer(d_model, dc(attn), dc(feedforward), dropout), 
                                              DecoderLayer(d_model, dc(attn), dc(attn), dc(feedforward), dropout), 
                                              N
                                              ),
                              nn.Sequential(
                                  Embeddings(d_model, len(vocabs['src_lang'])), 
                                  dc(position)
                                  ),
                              nn.Sequential(
                                  Embeddings(d_model, len(vocabs['tgt_lang'])), 
                                  dc(position)
                                  ),
                              generator, 
                              device
                              )

    if load:
        state_dict = torch.load('checkpoints/new_script_checkpoint_inf2.pth')
        model.load_state_dict(state_dict)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model.to(device)

    return model
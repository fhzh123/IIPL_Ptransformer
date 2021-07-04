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
        src = src.transpose(0,1)

        # src = [batch size, src len]

        src_mask = make_src_mask(src)

        # src_mask = [batch_size, 1, 1, src len]

        tgt = tgt.transpose(0,1)

        # tgt = [batch size, trg len]

        tgt_mask = make_trg_mask(tgt, self.device)

        # tgt_mask = [batch_size, 1, trg len, trg len]

        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class p_transformer(nn.Module):
    def __init__(self, encoder_decoder, src_embed, tgt_embed, generator, device):
        super(p_transformer, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.device = device 

    def forward(self, src, tgt):

        src = src.transpose(0,1)

        # src = [batch size, src len]

        src_mask = make_src_mask(src)

        # src_mask = [batch_size, 1, 1, src len]

        tgt = tgt.transpose(0,1)

        # tgt = [batch size, trg len]

        tgt_mask = make_trg_mask(tgt, self.device)

        # tgt_mask = [batch_size, 1, trg len, trg len]

        return self.generator(self.encode_decode(src=src, src_mask=src_mask, tgt=tgt, tgt_mask=tgt_mask))

    def encode_decode(self, src, src_mask, tgt, tgt_mask):
        return self.encoder_decoder(src=self.src_embed(src), src_mask=src_mask, tgt=self.tgt_embed(tgt), tgt_mask=tgt_mask)

def make_src_mask(src):
    
    #src = [batch size, src len]
    
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    #src_mask = [batch size, 1, 1, src len]
    # print("src_mask: {}".format(src_mask.shape))

    return src_mask

def make_trg_mask(trg, device):

    #trg = [batch size, trg len]
    
    trg_pad_mask = (trg != PAD_IDX).unsqueeze(1).unsqueeze(2)

    #trg_pad_mask = [batch size, 1, 1, trg len]
    
    trg_len = trg.shape[1]
    
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()

    #trg_sub_mask = [trg len, trg len]
        
    trg_mask = trg_pad_mask & trg_sub_mask

    #trg_mask = [batch size, 1, trg len, trg len]
    # print("trg_mask: {}".format(trg_mask.shape))
    
    return trg_mask

def build_model(vocabs, nhead, d_model, d_ff, N, device, dropout=0.1, variation=False, load=False):
    attn = MultiHeadAttention(nhead, d_model, dropout)
    feedforward = PositionWiseFeedForward(d_model, d_ff)
    position = PositionalEncoding(d_model, dropout)
    if not variation:
        model = o_transformer(Encoder(EnocderLayer(d_model, dc(attn), dc(feedforward), dropout), N), 
                              Decoder(DecoderLayer(d_model, dc(attn), dc(attn), dc(feedforward), dropout), N),
                              nn.Sequential(Embeddings(d_model, len(vocabs['src_lang'])), dc(position)),
                              nn.Sequential(Embeddings(d_model, len(vocabs['tgt_lang'])), dc(position)),
                              Generator(d_model, len(vocabs['tgt_lang'])), device=device
                              )
    else:
        model = p_transformer(Encoder_Decoder(EnocderLayer(d_model, dc(attn), dc(feedforward), dropout), DecoderLayer(d_model, dc(attn), dc(attn), dc(feedforward), dropout), N),
                              nn.Sequential(Embeddings(d_model, len(vocabs['src_lang'])), dc(position)),
                              nn.Sequential(Embeddings(d_model, len(vocabs['tgt_lang'])), dc(position)),
                              Generator(d_model, len(vocabs['tgt_lang'])), device=device
                              )

    if load:
        state_dict = torch.load('checkpoints/script_checkpoint_inf.pth')
        model.load_state_dict(state_dict)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model.to(device)

    return model
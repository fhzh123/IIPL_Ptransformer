import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def clones(module, N):
    # Creates deep copies of the modules N times and store it in a list.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_src_mask(src):
    
    #src = [batch size, src len]
    
    src_mask = (src == PAD_IDX)

    #src_mask = [batch size, 1, 1, src len]

    return src_mask

def make_trg_mask(trg, device):

    #trg = [batch size, trg len]
    
    trg_pad_mask = (trg == PAD_IDX)

    #trg_pad_mask = [batch size, 1, 1, trg len]
    
    # trg_len = trg.shape[1]
    
    # trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()

    # #trg_sub_mask = [trg len, trg len]
        
    # trg_mask = trg_pad_mask & trg_sub_mask

    #trg_mask = [batch size, 1, trg len, trg len]
    
    return trg_pad_mask

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones((sz, sz), device=device) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0,1)
    return src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model, src_sentence, vocabs, text_transform, device):
    model.eval()
    src = text_transform['src_lang'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX, device=device).flatten()
    return " ".join(vocabs['tgt_lang'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos> ", "").replace(" <eos>", "")

def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60)
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec    
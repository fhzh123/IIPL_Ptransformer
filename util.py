import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
UNK_IDX = 3
BOS_IDX = 1
EOS_IDX = 2

def generate_square_subsequent_mask(tgt_seq_len, device):
    mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device)) == 1).transpose(0, 1)
    """ 
    mask = [1, 0, 0
            1, 1, 0
            1, 1, 1] obviously the mask dimension would be tgt_seq_len * tgt_seq_len
    """
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    """
    mask = [1, -inf, -inf
            1,    1, -inf
            1,    1,    1] dimension stays the same.
    """
    return mask

def create_mask(src, tgt, device):
    # src = [src_len, batch]
    # tgt = [tgt_len, batch]
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    # src_mask = [src_len, src_len]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device = device)
    # tgt_mask = [tgt_len, tgt_len]

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # src_padding_mask == src.shape.transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    # tgt_padding_mask == tgt.shape.transpose(0,1)

    """
    src_mask and tgt_mask enables the transformer to attend to the position less than i given current position i
    src_padding_mask and tgt_padding_mask helps transformer ignore the padding indexes.
    """

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def clones(module, N):
    # returns N deepcopies of the input module
    return nn.ModuleList( [ copy.deepcopy(module) for _ in range(N) ] )
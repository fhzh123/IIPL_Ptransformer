import copy
import torch
import random
import numpy as np
import torch.nn as nn

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def generate_square_subsequent_mask(sz, device):
    """ 
    mask = [1, 0, 0
            1, 1, 0
            1, 1, 1] obviously the mask dimension would be tgt_seq_len * tgt_seq_len
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)

    """
    mask = [1, -inf, -inf
            1,    1, -inf
            1,    1,    1] dimension stays the same.
    """
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def create_mask(src, tgt, device):
    """
    src_mask and tgt_mask enables the transformer to attend to the position less than i given current position i
    src_padding_mask and tgt_padding_mask helps transformer ignore the padding indexes.
    """

    # src = [src_len, batch]
    src_seq_len = src.shape[0]

    # tgt = [tgt_len, batch]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    # src_padding_mask == src.shape.transpose(0,1)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)

    # tgt_padding_mask == tgt.shape.transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def clones(module, N):
    # returns N deepcopies of the input module
    return nn.ModuleList( [ copy.deepcopy(module) for _ in range(N) ] )

def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60) - minutes
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec   


def get_n_stacks(n_layers, layer, partition=3):

  partitions = np.full(partition, n_layers//partition)
  partitions[: n_layers % partition] += 1

  partitions = list(partitions)

  stacks = []

  for part in partitions:
    stacks.append(clones(layer, part))

  return stacks

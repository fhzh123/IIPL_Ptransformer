import gc
import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from tokenizers import Tokenizer

UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

def generate_square_subsequent_mask(tgt_seq_len, device):
    mask = (torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device)) == 1).transpose(0, 1)
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

def divide_sentences(sentences):
    train, val, test = {}, {}, {}

    for ln in ['src_lang', 'tgt_lang']:
        temp = sentences[ln]
        random.shuffle(temp)
        train_len = int(len(temp)*0.8)
        val_len = int(len(temp)*0.1)+train_len
        test_len =  int(len(temp)*0.1)+val_len+train_len
        tmp_train,tmp_val,tmp_test = temp[0:train_len], temp[train_len:val_len], temp[val_len:test_len]

        train[ln], val[ln], test[ln] = tmp_train, tmp_val, tmp_test

    print("\ntrain data length: {}".format(len(train['src_lang'])))
    print("validation data length: {}".format(len(val['src_lang'])))
    print("test data length: {}\n".format(len(test['src_lang'])))

    return train, val, test

def get_vocab_size():
    gc.disable()
    de_tokenizer = Tokenizer.from_file(os.path.join("./data/preprocessed", 'de_tokenizer.json'))
    en_tokenizer = Tokenizer.from_file(os.path.join("./data/preprocessed", 'en_tokenizer.json'))
    gc.enable()

    return de_tokenizer.get_vocab_size(), en_tokenizer.get_vocab_size()

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

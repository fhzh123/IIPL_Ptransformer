import gc
import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

PAD_IDX = 3
UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

def generate_square_subsequent_mask(trg_seq_len, device):
    mask = (torch.triu(torch.ones(trg_seq_len, trg_seq_len, device=device)) == 1).transpose(0, 1)
    """ 
    mask = [1, 0, 0
            1, 1, 0
            1, 1, 1] obviously the mask dimension would be trg_seq_len * trg_seq_len
    """
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    """
    mask = [1, -inf, -inf
            1,    1, -inf
            1,    1,    1] dimension stays the same.
    """
    return mask

def create_mask(src, trg, device):
    # src = [src_len, batch]
    # trg = [trg_len, batch]
    src_seq_len = src.size(0)
    trg_seq_len = trg.size(0)

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    # src_mask = [src_len, src_len]


    trg_mask = generate_square_subsequent_mask(trg_seq_len,device = device)
    # trg_mask = [trg_len, trg_len]

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # src_padding_mask == src.shape.transpose(0,1)
    trg_padding_mask = (trg == PAD_IDX).transpose(0, 1)
    # trg_padding_mask == trg.shape.transpose(0,1)

    """
    src_mask and trg_mask enables the transformer to attend to the position less than i given current position i
    src_padding_mask and trg_padding_mask helps transformer ignore the padding indexes.
    """

    return src_mask, trg_mask, src_padding_mask, trg_padding_mask

def clones(module, N):
    # returns N deepcopies of the input module
    return nn.ModuleList( [ copy.deepcopy(module) for _ in range(N) ] )

def divide_sentences(sentences):
    train, val, test = {}, {}, {}

    for ln in ['src_lang', 'trg_lang']:
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
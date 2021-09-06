import os
import torch
import random
import pickle
import pandas as pd
from datasets import load_dataset


data_path = "./data/wmt16/"

def get_sentences_wmt16():
    
    # 1) train data load

    with open(os.path.join(data_path, 'train.de'), 'r') as f:
        train_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'train.en'), 'r') as f:
        train_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    # 2) Valid data load
    with open(os.path.join(data_path, 'val.de'), 'r') as f:
        valid_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'val.en'), 'r') as f:
        valid_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    # 3) Test data load
    with open(os.path.join(data_path, 'test.de'), 'r') as f:
        test_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'test.en'), 'r') as f:
        test_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    train = {'src_lang' :  train_src_sequences , 'tgt_lang' : train_trg_sequences}
    val = {'src_lang' :  valid_src_sequences , 'tgt_lang' : valid_trg_sequences}
    test = {'src_lang' :  test_src_sequences , 'tgt_lang' : test_trg_sequences}

    print("\ntrain data length: {}".format(len(train['src_lang'])))
    print("validation data length: {}".format(len(val['src_lang'])))
    print("test data length: {}\n".format(len(test['src_lang'])))

    return train, val, test


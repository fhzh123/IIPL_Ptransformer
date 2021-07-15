# Import Modules
import os
import json
import time
import pickle
import argparse
import numpy as np
import sentencepiece as spm
from glob import glob
from tqdm import tqdm
from collections import Counter

def preprocess(data_path="./data/wmt16", preprocess_path="./data/preprocessed", sentencepiece_model = 'bpe', src_vocab_size = 8000, trg_vocab_size = 8000, pad_id = 0, unk_id = 3, bos_id = 1, eos_id = 2 ):

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load...')
    # 1) Train data load
    with open(os.path.join(data_path, 'train.de'), 'r') as f:
        train_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'train.en'), 'r') as f:
        train_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    # 2) Valid data load
    with open(os.path.join(data_path, 'val.de'), 'r') as f:
        valid_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'val.en'), 'r') as f:
        valid_trg_sequences = [x.replace('\n', '') for x in f.readlines()]

    # 2) Test data load
    with open(os.path.join(data_path, 'test.de'), 'r') as f:
        test_src_sequences = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(data_path, 'test.en'), 'r') as f:
        test_trg_sequences = [x.replace('\n', '') for x in f.readlines()]



    # 3) Path setting
    if not os.path.exists(preprocess_path):
        os.mkdir(preprocess_path)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    print('SentencePiece Training')
    start_time = time.time()

    # 1) Source lanugage
    # Make text to train vocab
    with open(f'{preprocess_path}/src.txt', 'w') as f:
        for text in train_src_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={preprocess_path}/src.txt --model_prefix={preprocess_path}/m_src_{src_vocab_size} '
        f'--vocab_size={src_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={pad_id} --unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id} '
        f'--model_type={sentencepiece_model} --max_sentence_length=10000000')

    src_vocab = list()
    with open(f'{preprocess_path}/m_src_{src_vocab_size}.vocab') as f:
        for line in f:
            src_vocab.append(line[:-1].split('\t')[0])

    src_word2id = {w: i for i, w in enumerate(src_vocab)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{preprocess_path}/m_src_{src_vocab_size}.model')

    train_src_indices = tuple(
        [bos_id] + spm_src.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [eos_id] for text in train_src_sequences
    )
    valid_src_indices = tuple(
        [bos_id] + spm_src.encode(text, out_type=int) + [eos_id] for text in valid_src_sequences
    )
    test_src_indices = tuple(
        [bos_id] + spm_src.encode(text, out_type=int) + [eos_id] for text in test_src_sequences
    )

    # 2) Target lanugage
    # Make text to train vocab
    with open(f'{preprocess_path}/trg.txt', 'w') as f:
        for text in train_trg_sequences:
            f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={preprocess_path}/trg.txt --model_prefix={preprocess_path}/m_trg_{trg_vocab_size} '
        f'--vocab_size={trg_vocab_size} --character_coverage=0.9995 --split_by_whitespace=true '
        f'--pad_id={pad_id} --unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id} '
        f'--model_type={sentencepiece_model} --max_sentence_length=10000000')

    trg_vocab = list()
    with open(f'{preprocess_path}/m_trg_{trg_vocab_size}.vocab') as f:
        for line in f:
            trg_vocab.append(line[:-1].split('\t')[0])

    trg_word2id = {w: i for i, w in enumerate(trg_vocab)}

    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{preprocess_path}/m_trg_{trg_vocab_size}.model')

    train_trg_indices = tuple(
        [bos_id] + spm_trg.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [eos_id] for text in train_trg_sequences
    )
    valid_trg_indices = tuple(
        [bos_id] + spm_trg.encode(text, out_type=int) + [eos_id] for text in valid_trg_sequences
    )
    test_trg_indices = tuple(
        [bos_id] + spm_trg.encode(text, out_type=int) + [eos_id] for text in test_trg_sequences
    )

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(preprocess_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_src_indices': train_src_indices,
            'valid_src_indices': valid_src_indices,
            'train_trg_indices': train_trg_indices,
            'valid_trg_indices': valid_trg_indices,
            'src_word2id': src_word2id,
            'trg_word2id': trg_word2id
        }, f)

    with open(os.path.join(preprocess_path, 'test_processed.pkl'), 'wb') as f:
        pickle.dump({
            'test_src_indices': test_src_indices,
            'test_trg_indices': test_trg_indices,
            'src_word2id': src_word2id,
            'trg_word2id': trg_word2id
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')


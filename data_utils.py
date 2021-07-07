import os
import torch
import random
import pickle
from util import *
import pandas as pd
from typing import List
from fetch_tokenizers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def convert_to_csv(root, files):
  count = 1
  csv_files = []
  for filepath in files:
    if filepath[-4:] == "xlsx":
      xls = pd.read_excel(os.path.join(root, filepath), index_col=None)
      destination = 'data/korean_to_english' + str(count) + '.csv'
      xls.to_csv(destination, encoding='utf-8', index=False)
      csv_files.append(destination)
  
  return csv_files

def convert_to_sentences(csv_files, load):
  dataframes = [ pd.read_csv(filepath) for filepath in csv_files ]
  
  kor_sentences = []
  eng_sentences = []
  if not load:
    for data in dataframes:
      for index, sent in data.iterrows():
        _, kor, eng = sent
        kor_sentences.append(kor)
        eng_sentences.append(eng)
    with open('pickle_files/kor_sent.pkl', 'wb') as kor_out:
      pickle.dump(kor_sentences, kor_out)
    with open('pickle_files/eng_sent.pkl', 'wb') as eng_out:
      pickle.dump(eng_sentences, eng_out)
  else:
    file_kor = open('pickle_files/kor_sent.pkl', 'rb')
    kor_sentences = pickle.load(file_kor)
    file_eng = open('pickle_files/eng_sent.pkl', 'rb')
    eng_sentences = pickle.load(file_eng)

  for kor, eng in zip(kor_sentences[:5], eng_sentences[:5]):
      print(f'[KOR]: {kor}')
      print(f'[ENG]: {eng}\n')
  print(f'[KOR LEN]: {len(kor_sentences)}')
  print(f'[ENG LEN]: {len(eng_sentences)}\n')
  
  return kor_sentences, eng_sentences

def get_kor_eng_sentences(file_path="data", load=False):
  root = "data"
  files = os.listdir("./data")
  csv_files = convert_to_csv(root, files)
  return convert_to_sentences(csv_files, load)


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

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
          txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_text_transform(tokens, vocabs):
  text_transform = {}
  for ln in ['src_lang', 'tgt_lang']:
      text_transform[ln] = sequential_transforms(tokens[ln], 
                                                vocabs[ln], 
                                                tensor_transform)
  return text_transform


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def data_process(sentences, vocabs, tokens):
  output = []
  text_transform = get_text_transform(tokens, vocabs)
  for kor, eng in zip(sentences['src_lang'], sentences['tgt_lang']):
    ko_tensor = text_transform['src_lang'](kor.rstrip("\n"))
    en_tensor = text_transform['tgt_lang'](eng.rstrip("\n"))
    output.append((ko_tensor, en_tensor))
  return output

def get_train_iter(sentences, tokens, vocabs, batch_size):
  train_data = data_process(sentences, vocabs, tokens)
  return DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def get_test_iter(sentences, tokens, vocabs, batch_size):
  test_data = data_process(sentences, vocabs, tokens)
  return DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
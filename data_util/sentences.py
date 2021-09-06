import os
import torch
import random
import pickle
import pandas as pd
from datasets import load_dataset


data_path = "./data/wmt14/wmt14.pkl"

def get_sentences(max_len=300):
    if(os.path.isfile(data_path)):
        print("\nload {}".format(data_path))
        with open(data_path, 'rb') as f:
            dicts = pickle.load(f)
    else:
        print("no pickle")
        dataset = load_dataset('wmt14', 'de-en')

        data = pd.DataFrame(dataset['train']['translation']+dataset['validation']
                            ['translation']+dataset['test']['translation'])
        data.loc[:, 'de_len'] = data.loc[:, 'de'].apply(len)
        data.loc[:, 'en_len'] = data.loc[:, 'en'].apply(len)
        data, dump = data.loc[(data.de_len <= max_len) & (data.en_len <= max_len), ['de', 'en']], data.loc[(data.de_len > max_len) & (data.en_len > max_len), ['de', 'en']]
        del dump
        

        src_list = list(data.loc[:, 'de'])
        tgt_list = list(data.loc[:, 'en'])
        del data
        dicts = {'src_lang': src_list, 'tgt_lang': tgt_list}
        with open(data_path, 'wb') as f:
            pickle.dump(dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nsave {}".format(data_path))

    return divide_sentences(dicts)


def divide_sentences(sentences):
    train, val, test = {}, {}, {}

    for ln in ['src_lang', 'tgt_lang']:
        temp = sentences[ln]
        random.shuffle(temp)
        
        #train_len = int(len(temp)*0.8)
        #val_len = int(len(temp)*0.1)+train_len
        #test_len = int(len(temp))

        train_len = int(400000)
        val_len = int(50000)+train_len
        test_len = int(460000)


        tmp_train, tmp_val, tmp_test = temp[0:train_len], temp[train_len:val_len], temp[val_len:test_len]

        train[ln], val[ln], test[ln] = tmp_train, tmp_val, tmp_test

    print("\ntrain data length: {}".format(len(train['src_lang'])))
    print("validation data length: {}".format(len(val['src_lang'])))
    print("test data length: {}\n".format(len(test['src_lang'])))

    return train, val, test


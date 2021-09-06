import torch
import os
import pickle
from util import BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'src_lang'
TGT_LANGUAGE = 'tgt_lang'

token_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

#vocab_path = "./data/wmt14/vocab_transform.pkl"
vocab_path = "./data/wmt16/vocab_transform.pkl"

def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def get_vocabs(train_iter):
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    if(os.path.isfile(vocab_path)):
        print("    load vocab_transform")
        with open(vocab_path , 'rb') as f:
            vocab_transform = pickle.load(f)
    else :
        vocab_transform = {}

        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                            min_freq=5,
                                                            specials=special_symbols,
                                                            special_first=True)

        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln].set_default_index(UNK_IDX)
        
        with open(vocab_path , 'wb') as f:
            pickle.dump(vocab_transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("    save vocab_transform")

    return vocab_transform


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids, dtype=torch.int64),
                      torch.tensor([EOS_IDX])))


def get_text_transform(vocab_transform):
    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                # Numericalization
                                                vocab_transform[ln],
                                                tensor_transform)  # Add BOS/EOS and create tensor

    return text_transform

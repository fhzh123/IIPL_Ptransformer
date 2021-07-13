import torch
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def get_tokens():
    tokens = {}

    tokens['src_lang'] = get_tokenizer('spacy', language='de_core_news_sm')
    tokens['tgt_lang'] = get_tokenizer('spacy', language='en_core_web_sm')

    return tokens

def get_vocabs(tokens):
    vocabs = {}

    train = Multi30k(split='train')

    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in ['src_lang', 'tgt_lang']:
        vocabs[ln] = build_vocab_from_iterator(
                                    yield_tokens(train, tokens, ln),
                                    min_freq=1,
                                    specials=special_symbols,
                                    special_first=True
                                    ).set_default_index(UNK_IDX)

    return vocabs

def yield_tokens(data, tokens, language):
    lang_index = {'src_lang': 0, 'tgt_lang': 1}

    for data_sample in data:
        yield tokens[language](data_sample[lang_index[language]])


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
def get_text_transform(tokens, vocabs):
    text_transform = {}

    for ln in ['src_lang', 'tgt_lang']:
        text_transform[ln] = sequential_transforms(
                                            tokens[ln],
                                            vocabs[ln],
                                            tensor_transform
                                                   )

    return text_transform
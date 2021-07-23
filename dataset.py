import torch
from torch._C import dtype
from torch.utils.data.dataset import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
from torch.nn.utils.rnn import pad_sequence

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer(
    'spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer(
    'spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter, language: str):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<bos>', '<eos>', '<pad>']

src_train = []
tgt_train = []

with open(os.path.join("./data/preprocessed", 'src_train.txt'), 'r') as f:
	data_ = f.readlines()
	for text in data_:
		src_train.append(text)
	del data_

with open(os.path.join("./data/preprocessed", 'tgt_train.txt'), 'r') as f:
	data_ = f.readlines()
	for text in data_:
		tgt_train.append(text)
	del data_

vocab_transform = {}

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(zip(src_train, tgt_train), ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               # Numericalization
                                               vocab_transform[ln],
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

class CustomDataset(Dataset):
	def __init__(self, src_list, tgt_list, de_tokenizer, en_tokenizer):
		self.src_list = src_list
		self.tgt_list = tgt_list

		self.de_tokenizer = de_tokenizer
		self.en_tokenizer = en_tokenizer

		self.num_data = len(self.src_list)

	def __getitem__(self, index):
		src = self.src_list[index]
		tgt = self.tgt_list[index]

		src_tensor = torch.tensor(self.de_tokenizer.encode(src).ids, dtype=torch.int64)
		tgt_tensor = torch.tensor(self.en_tokenizer.encode(tgt).ids, dtype=torch.int64)
		
		return tuple((src_tensor, tgt_tensor))
		
	def __len__(self):
		return self.num_data

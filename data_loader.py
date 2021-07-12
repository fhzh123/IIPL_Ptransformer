import torch
from vocabs import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def sequential_transforms(*transforms):
	def func(txt_input):
		for transform in transforms:
			txt_input = transform(txt_input)
		return txt_input
	return func

def tensor_transform(token_ids):
	return torch.cat((
		torch.tensor([SOS_IDX]),
		torch.tensor(token_ids),
		torch.tensor([EOS_IDX])
		))

def get_text_transform(tokens, vocabs):
	text_transform = {}
	for ln in ['src_lang', 'tgt_lang']:
		text_transform[ln] = sequential_transforms(
											tokens[ln], 
											vocabs[ln], 
											tensor_transform
											)
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
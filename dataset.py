import torch
from util import PAD_IDX
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
	def __init__(self, src_list, trg_list, de_tokenizer, en_tokenizer):
		self.src_list = src_list
		self.trg_list = trg_list

		self.de_tokenizer = de_tokenizer
		self.en_tokenizer = en_tokenizer

		self.num_data = len(self.src_list)

	def __getitem__(self, index):
		src = self.src_list[index]
		trg = self.trg_list[index]

		src_tensor = torch.tensor(self.de_tokenizer.encode(src).ids)
		trg_tensor = torch.tensor(self.en_tokenizer.encode(trg).ids)
		
		return tuple((src_tensor, trg_tensor))
		
	def __len__(self):
		return self.num_data

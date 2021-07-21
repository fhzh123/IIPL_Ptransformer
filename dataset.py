import torch
from torch.utils.data.dataset import Dataset

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

		src_tensor = torch.tensor(self.de_tokenizer.encode(src).ids)
		tgt_tensor = torch.tensor(self.en_tokenizer.encode(tgt).ids)
		
		return tuple((src_tensor, tgt_tensor))
		
	def __len__(self):
		return self.num_data

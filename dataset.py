import torch
from util import PAD_IDX
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
	def __init__(self, src_list, trg_list, src_tokenizer, trg_tokenizer):
		self.encoded_src = src_tokenizer.encode_batch(src_list)
		self.encoded_trg = trg_tokenizer.encode_batch(trg_list)

		self.num_data = len(src_list)

	def __getitem__(self, index):
		src = self.encoded_src[index]
		trg = self.encoded_trg[index]

		src_tensor = torch.tensor(src.ids)
		trg_tensor = torch.tensor(trg.ids)

		return tuple((src_tensor, trg_tensor))

	def __len__(self):
		return self.num_data
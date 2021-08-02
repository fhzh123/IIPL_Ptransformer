from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
	def __init__(self, src_list, tgt_list):
		self.src_list = src_list
		self.tgt_list = tgt_list

		self.num_data = len(self.src_list)

	def __getitem__(self, index):
		src = self.src_list[index]
		tgt = self.tgt_list[index]

		return tuple((src, tgt))

	def __len__(self):
		return self.num_data

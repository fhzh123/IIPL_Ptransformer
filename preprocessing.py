# Import Modules
import os
import time
from tokenizers import Tokenizer
from util import divide_sentences
from tokenizers.models import BPE
from matplotlib import pyplot as plt
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def preprocess(data_path="./data/wmt16", preprocess_path="./data/preprocessed"):

	#===================================#
	#============Data Load==============#
	#===================================#

	file_list = os.listdir(data_path)

	de_list = [de for de in file_list if de[-2:] == "de"]
	en_list = [en for en in file_list if en[-2:] == "en"]

	src_list = []
	tgt_list = []

	for de in de_list:
		with open(os.path.join(data_path, de), 'r') as f:
			for text in f.readlines():
				stripped_text = text.rstrip("\n")
				
				# if src_max_len < len(stripped_text):
				# 	src_max_len = len(stripped_text)
				if len(stripped_text) <= 300:
					src_list.append(stripped_text)

	for en in en_list:
		with open(os.path.join(data_path, en), 'r') as f:
			for text in f.readlines():
				stripped_text = text.rstrip("\n")
				
				# if tgt_max_len < len(stripped_text):
				# 	tgt_max_len = len(stripped_text)

				if len(stripped_text) <= 300:
					tgt_list.append(stripped_text)

	train, val, test = divide_sentences({'src_lang': src_list, 'tgt_lang': tgt_list})

	# 3) Path setting
	if not os.path.exists(preprocess_path):
		os.mkdir(preprocess_path)

	#===================================#
	#==============Saving===============#
	#===================================#

	# Save train sentences
	with open(f'{preprocess_path}/src_train.txt', 'w') as f:
		for text in train['src_lang']:
			f.write(f'{text}\n')

	with open(f'{preprocess_path}/tgt_train.txt', 'w') as f:
		for text in train['tgt_lang']:
			f.write(f'{text}\n')

	# Save valid sentences
	with open(f'{preprocess_path}/src_val.txt', 'w') as f:
		for text in val['src_lang']:
			f.write(f'{text}\n')

	with open(f'{preprocess_path}/tgt_val.txt', 'w') as f:
		for text in val['tgt_lang']:
			f.write(f'{text}\n')

	# Save test sentences
	with open(f'{preprocess_path}/src_test.txt', 'w') as f:
		for text in test['src_lang']:
			f.write(f'{text}\n')

	with open(f'{preprocess_path}/tgt_test.txt', 'w') as f:
		for text in test['tgt_lang']:
			f.write(f'{text}\n')

	#===================================#
	#==========DE->EN Tokenizer=========#
	#===================================#

	# 1) BPE Tokenizer
			#  - For DE->EN translation, shared tokenizer was used,

	print('BPE Training')
	start_time = time.time()

	de_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
	en_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

	de_trainer = BpeTrainer(
			special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[MASK]"],
			vocab_size=37000, min_frequency=3
	)
	en_trainer = BpeTrainer(
			special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[MASK]"],
			vocab_size=37000, min_frequency=3
	)

	de_tokenizer.pre_tokenizer = Whitespace()
	en_tokenizer.pre_tokenizer = Whitespace()

	de_tokenizer.train_from_iterator(src_list, de_trainer)
	en_tokenizer.train_from_iterator(tgt_list, en_trainer)

	de_tokenizer.post_processor = TemplateProcessing(
			single="[BOS] $A [EOS]",
			pair="[BOS] $A [EOS] $B:1 [EOS]:1",
			special_tokens=[
					("[BOS]", de_tokenizer.token_to_id("[BOS]")),
					("[EOS]", de_tokenizer.token_to_id("[EOS]")),
			],
	)
	en_tokenizer.post_processor = TemplateProcessing(
			single="[BOS] $A [EOS]",
			pair="[BOS] $A [EOS] $B:1 [EOS]:1",
			special_tokens=[
					("[BOS]", en_tokenizer.token_to_id("[BOS]")),
					("[EOS]", en_tokenizer.token_to_id("[EOS]")),
			],
	)

	de_tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=300)
	en_tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=300)

	de_tokenizer.save(f'{preprocess_path}/de_tokenizer.json')
	en_tokenizer.save(f'{preprocess_path}/en_tokenizer.json')

	print("EXAMPLE:\n DE: {}\n EN: {}\n".format(de_tokenizer.encode(
			src_list[0]).tokens, en_tokenizer.encode(tgt_list[0]).tokens))

	print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

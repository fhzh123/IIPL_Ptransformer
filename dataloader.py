# Import modules
import os
import gc
import pickle
# Import PyTorch
from dataset import CustomDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader


min_len=4
src_max_len=50
trg_max_len=50
num_workers=4
batch_size=32
num_epochs=100
lr=5e-5
w_decay=1e-5


def get_dataloader():

    # 1) Data open
    gc.disable()
    with open(os.path.join("./data/preprocessed", 'processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_src_indices = data_['train_src_indices']
        valid_src_indices = data_['valid_src_indices']
        train_trg_indices = data_['train_trg_indices']
        valid_trg_indices = data_['valid_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_
    gc.enable()

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(train_src_indices, train_trg_indices, 
                            min_len=min_len, src_max_len=src_max_len, trg_max_len=trg_max_len),
        'valid': CustomDataset(valid_src_indices, valid_trg_indices,
                            min_len=min_len, src_max_len=src_max_len, trg_max_len=trg_max_len),
    }

    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_workers)
    }

    return dataloader_dict, src_vocab_num, trg_vocab_num


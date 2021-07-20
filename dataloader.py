# Import modules
import os
import gc
import torch
from tokenizers import Tokenizer
# Import PyTorch
from dataset import CustomDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size, num_workers=4):
    
    src_train, trg_train = [], []
    src_val, trg_val = [], []

    # 1) Data open
    gc.disable()
    with open(os.path.join("./data/preprocessed", 'src_train.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            src_train.append(text)
        del data_
    
    with open(os.path.join("./data/preprocessed", 'trg_train.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            trg_train.append(text)
        del data_

    with open(os.path.join("./data/preprocessed", 'src_val.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            src_val.append(text)
        del data_

    with open(os.path.join("./data/preprocessed", 'trg_val.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            trg_val.append(text)
        del data_

    de_tokenizer = Tokenizer.from_file(os.path.join("./data/preprocessed", 'de_tokenizer.json'))
    en_tokenizer = Tokenizer.from_file(os.path.join("./data/preprocessed", 'en_tokenizer.json'))
    gc.enable()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_train, trg_train, de_tokenizer, en_tokenizer),
        'valid': CustomDataset(src_val, trg_val, de_tokenizer, en_tokenizer)
    }

    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=batch_size, shuffle=True, pin_memory=False,
                            num_workers=num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=batch_size, shuffle=False, pin_memory=False,
                            num_workers=num_workers)
    }

    return dataloader_dict

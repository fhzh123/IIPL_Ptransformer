import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch import optim
from my_optim import ScheduledOptim
import nltk.translate.bleu_score as bs
from preprocessing import preprocess
from dataloader import get_dataloader
from model.transformer import build_model
from util import epoch_time, PAD_IDX, create_mask
from bleu import get_bleu
from tqdm import tqdm

SEED = 970308

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, num_epoch, lr,
                 emb_size, nhead, ffn_hid_dim, batch_size,
                 n_layers, dropout, load, variation):
        super(Trainer, self).__init__()
        
        self.params = {
                       'num_epoch': num_epoch,
                       'emb_size': emb_size,
                       'nhead': nhead,
                       'ffn_hid_dim': ffn_hid_dim,
                       'batch_size': batch_size,
                       'n_layers': n_layers,
                       'dropout': dropout,
                       'lr': lr,
                       }

        preprocess()

        self.dataloader, self.params['src_vocab_size'], self.params['tgt_vocab_size'] = get_dataloader()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = build_model(
                            self.params['n_layers'], self.params['emb_size'], self.params['nhead'], 
                            self.params['src_vocab_size'], self.params['tgt_vocab_size'], 
                            self.params['ffn_hid_dim'], self.params['dropout'], False,
                            load, self.device
                            )

        # self.optimizer = ScheduledOptim(
        #     optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=5e-5),
        #     warmup_steps=4000,
        #     hidden_dim=self.params['ffn_hid_dim']
        # )
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=5e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30,80], gamma=0.1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def learn(self):
        print("\nbegin training...")

        for epoch in range(self.params['num_epoch']):
            start_time = time.time()

            epoch_loss = train_loop(self.dataloader['train'], self.model, self.optimizer, self.criterion, self.device)

            self.scheduler.step()

            val_loss = val_loop(self.dataloader['valid'], self.model, self.criterion, self.device)

            end_time = time.time()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])
        
            print("Epoch: {} out of {}".format(epoch+1, self.params['num_epoch']))
            print("Train_loss: {} - Val_loss: {} - Epoch time: {}m {}s - Time left for training: {}m {}s"\
            .format(round(epoch_loss, 3), round(val_loss, 3), minutes, seconds, time_left_min, time_left_sec))

        torch.save(self.model.state_dict(), './data/checkpoints/checkpoint.pth')
        torch.save(self.model, './data/checkpoints/checkpoint.pt')

        get_bleu()

def train_loop(train_iter, model, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(train_iter, desc = 'training...'):
        src = src.to(device)
        tgt = tgt.to(device)
        
        src = src.transpose(0,1) # [length, batch]
        tgt = tgt.transpose(0,1) # [length, batch]
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, device)

        logits = model(
                  src=src, 
                  tgt=tgt_input, 
                  src_mask=src_mask, 
                  tgt_mask=tgt_mask,
                  src_key_padding_mask=src_key_padding_mask, 
                  tgt_key_padding_mask=tgt_key_padding_mask, 
                  memory_key_padding_mask=src_key_padding_mask
                  )

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        
        
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_iter)

def val_loop(val_iter, model, criterion, device):
    model.eval()
    losses = 0

    for src, tgt in tqdm(val_iter, desc = 'validation...'):
        src = src.to(device)
        tgt = tgt.to(device)
        
        src = src.transpose(0,1) # [length, batch]
        tgt = tgt.transpose(0,1) # [length, batch]
        
        
        tgt_input = tgt[:-1, :]
        

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, device)

        logits = model(
                  src=src, 
                  tgt=tgt_input, 
                  src_mask=src_mask, 
                  tgt_mask=tgt_mask,
                  src_key_padding_mask=src_key_padding_mask, 
                  tgt_key_padding_mask=tgt_key_padding_mask, 
                  memory_key_padding_mask=src_key_padding_mask
                  )

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_iter)

def test_loop(test_iter, model, criterion, device):
    model.eval()
    test_loss = 0

    for src, tgt in tqdm(test_iter, desc = 'test'):
        src = src.to(device)
        tgt = tgt.to(device)
        
        src = src.transpose(0,1) # [length, batch]
        tgt = tgt.transpose(0,1) # [length, batch]
        
        tgt_input = tgt[:-1, :]
        

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, device)

        logits = model(
                  src=src, 
                  tgt=tgt_input, 
                  src_mask=src_mask, 
                  tgt_mask=tgt_mask,
                  src_key_padding_mask=src_key_padding_mask, 
                  tgt_key_padding_mask=tgt_key_padding_mask, 
                  memory_key_padding_mask=src_key_padding_mask
                  )

        tgt_out = tgt[1:, :]
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  
        test_loss += loss.item()
    test_loss /= len(test_iter)

    print("Test Loss: {}".format(round(test_loss, 3)))
import time
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from bleu import get_bleu
from my_optim import ScheduledOptim
from torch.utils.data import DataLoader
from model.transformer import build_model
from data_util.dataset import CustomDataset
from torch.nn.utils.rnn import pad_sequence
from data_util.sentences import get_sentences
from util import PAD_IDX, create_mask, epoch_time
from data_util.transforms import get_vocabs, get_text_transform

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, num_epoch, lr,
                 emb_size, nhead, ffn_hid_dim, batch_size,
                 n_layers, dropout, load, isP, variation):
        super(Trainer, self).__init__()

        # self.src_language = src_language
        # self.tgt_language = tgt_language

        self. params = {
            'num_epoch': num_epoch,
            'emb_size': emb_size,
            'nhead': nhead,
            'ffn_hid_dim': ffn_hid_dim,
            'batch_size': batch_size,
            'n_layers': n_layers,
            'dropout': dropout,
            'lr': lr,
        }

        train, val, test = get_sentences()

        self.train_iter = CustomDataset(train['src_lang'], train['tgt_lang'])
        self.val_iter = CustomDataset(val['src_lang'], val['tgt_lang'])
        self.test_iter = CustomDataset(test['src_lang'], test['tgt_lang'])

        print('get_vocab & text_transform start')
        self.vocabs = get_vocabs(self.train_iter)
        self.text_transforms = get_text_transform(self.vocabs)

        self.params['src_vocab_size'], self.params['tgt_vocab_size'] = len(self.vocabs['src_lang']), len(self.vocabs['tgt_lang'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = build_model(
                            self.params['n_layers'], self.params['emb_size'], self.params['nhead'],
                            self.params['src_vocab_size'], self.params['tgt_vocab_size'],
                            self.params['ffn_hid_dim'], self.params['dropout'], isP, variation,
                            self.device
                            )

        self.scheduler = ScheduledOptim(
            torch.optim.Adam(self.model.parameters(),lr=0.0001, betas=(0.9, 0.98), eps=1e-9),
            warmup_steps=4000,
            hidden_dim=self.params['ffn_hid_dim']
        )
        
        self.start_epoch = 0
        self.variation = variation
        if load:
            PATH = f'./data/checkpoints/{self.variation}_checkpoint.pth.tar'
            checkpoint = torch.load(PATH)
            self.start_epoch = checkpoint['epoch']+1
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transforms['src_lang'](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transforms['tgt_lang'](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def learn(self):
        print("\nbegin training...")

        for epoch in range(self.start_epoch, self.params['num_epoch']+self.start_epoch):
            start_time = time.time()

            train_iter = DataLoader(self.train_iter, self.params['batch_size'], True, collate_fn=self.collate_fn)
            epoch_loss = train_loop(train_iter, self.model, self.scheduler, self.criterion, self.device)

            val_iter = DataLoader(self.val_iter, self.params['batch_size'], False, collate_fn=self.collate_fn)
            val_loss = val_loop(val_iter,self.model, self.criterion, self.device)

            end_time = time.time()

            minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])
        
            print("Epoch: {} out of {}".format(epoch+1, self.params['num_epoch']+self.start_epoch))
            print("Train_loss: {} - Val_loss: {} - Epoch time: {}m {}s - Time left for training: {}m {}s"\
            .format(round(epoch_loss, 3), round(val_loss, 3), minutes, seconds, time_left_min, time_left_sec))

            torch.save({'epoch' : epoch+1,
                        'model' : self.model.state_dict(),
                        'optimizer' : self.scheduler.optimizer.state_dict(),
                        'scheduler' : self.scheduler.state_dict()
                        }, f'./data/checkpoints/{self.variation}_checkpoint.pth.tar')
            torch.save(self.model, f'./data/checkpoints/{self.variation}_checkpoint.pt')

        get_bleu(self.model, self.test_iter, self.vocabs, self.text_transforms, self.device)
  
def train_loop(train_iter, model, scheduler, criterion, device):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(train_iter, desc = 'training...'):
        # [length, batch]
        src = src.to(device)
        
        # [length, batch]
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        scheduler.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_iter)

def val_loop(val_iter, model, criterion, device):
    model.eval()
    losses = 0

    for src, tgt in tqdm(val_iter, desc = 'validation...'):
        # [length, batch]
        src = src.to(device)
        
        # [length, batch]
        tgt = tgt.to(device)
        
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_iter)

import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# custom
from bleu import get_bleu
from my_optim import ScheduledOptim
from model.transformer import build_model
from data_util.dataset import CustomDataset
from data_util.sentences import get_sentences
from data_util.sentences_wmt16 import get_sentences_wmt16
from data_util.transforms import get_vocabs, get_text_transform
from util import PAD_IDX, create_mask, epoch_time

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

        #train, val, test = get_sentences()
        train, val, test = get_sentences_wmt16()

        self.train_iter = CustomDataset(train['src_lang'], train['tgt_lang'])
        self.val_iter = CustomDataset(val['src_lang'], val['tgt_lang'])
        self.test_iter = CustomDataset(test['src_lang'], test['tgt_lang'])

        print(self.train_iter[0])

        print('get_vocab & text_transform start')
        self.vocabs = get_vocabs(self.train_iter)
        self.text_transforms = get_text_transform(self.vocabs)
    
        self.params['src_vocab_size'], self.params['tgt_vocab_size'] = len(self.vocabs['src_lang']), len(self.vocabs['tgt_lang'])

        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

        self.model = build_model(
                            self.params['n_layers'], self.params['emb_size'], self.params['nhead'],
                            self.params['src_vocab_size'], self.params['tgt_vocab_size'],
                            self.params['ffn_hid_dim'], self.params['dropout'], isP, self.device
                            )

        self.scheduler = ScheduledOptim(
            torch.optim.Adam(self.model.parameters(),lr=0.0001, betas=(0.9, 0.98), eps=1e-9),
            warmup_steps=4000,
            hidden_dim=self.params['ffn_hid_dim']
        )
        
        self.start_epoch = 0
        self.variation = variation

        if self.variation == 'puzzle' :
            print("puzzle")
        else:
            print("None")

        if load:
            PATH = f"./data/checkpoints/{self.variation}_checkpoint.pth.tar"
            checkpoint = torch.load(PATH)
            self.start_epoch = checkpoint['epoch']+1
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.re_loss = L1_Loss

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            
            src_batch.append(self.text_transforms['src_lang'](src_sample.rstrip("\n")))
            
            tgt_batch.append(self.text_transforms['tgt_lang'](tgt_sample.rstrip("\n")))
            
        
        src_batch = custom_padding(src_batch, padding_value=PAD_IDX,batch_size = len(src_batch))
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def learn(self):
        print("\nbegin training...")
        
        for epoch in range(self.start_epoch, self.params['num_epoch']+self.start_epoch):

            start_time = time.time()
           
            train_iter = DataLoader(self.train_iter, self.params['batch_size'], True, collate_fn=self.collate_fn)
            epoch_loss = train_loop(train_iter, self.model, self.scheduler, self.criterion, self.device,self.variation, re_loss_fn=self.re_loss)

            val_iter = DataLoader(self.val_iter, self.params['batch_size'], False, collate_fn=self.collate_fn)
            val_loss = val_loop(val_iter,self.model, self.criterion, self.device, self.variation, re_loss_fn=self.re_loss)

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

def train_loop(train_iter, model, scheduler, criterion, device ,variation, re_loss_fn = None):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(train_iter, desc = 'training...'):

        #인코더 아웃풋 full, pre, post

        # full , pre ; post
        # 위 둘을 비교하는 로스
        
        # [length, batch]
        src = src.to(device)
        
        # [length, batch]
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
    
        logits, full_encode = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        re_loss = 0

        if variation == 'puzzle':
            bound = int(src.size(0)/2)
            _,seq_pre_encode = model(src[:bound,:], tgt_input, src_mask[:bound,:bound], tgt_mask, src_padding_mask[:,:bound], tgt_padding_mask, src_padding_mask[:,:bound])
            _,seq_post_encode = model(src[bound:,:], tgt_input, src_mask[bound:,bound:], tgt_mask, src_padding_mask[:,bound:], tgt_padding_mask, src_padding_mask[:,bound:])

            re_encode = torch.cat([seq_pre_encode,seq_post_encode],dim=0)

            del _
            del seq_pre_encode
            del seq_post_encode

            re_loss = re_loss_fn(full_encode,re_encode).mean()

        scheduler.zero_grad()

        tgt_out = tgt[1:, :]
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) + re_loss
        loss.backward()
        
        scheduler.step()
        epoch_loss += loss.item() 

    return epoch_loss / len(train_iter)

def val_loop(val_iter, model, criterion, device ,variation, re_loss_fn = None):
    model.eval()
    losses = 0

    for src, tgt in tqdm(val_iter, desc = 'validation...'):
        # [length, batch]
        src = src.to(device)
        
        # [length, batch]
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
        
        logits, full_encode = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        re_loss = 0

        if variation == 'puzzle':
            bound = int(src.size(0)/2)
            _,seq_pre_encode = model(src[:bound,:], tgt_input, src_mask[:bound,:bound], tgt_mask, src_padding_mask[:,:bound], tgt_padding_mask, src_padding_mask[:,:bound])
            _,seq_post_encode = model(src[bound:,:], tgt_input, src_mask[bound:,bound:], tgt_mask, src_padding_mask[:,bound:], tgt_padding_mask, src_padding_mask[:,bound:])

            re_encode = torch.cat([seq_pre_encode,seq_post_encode],dim=0)

            del _
            del seq_pre_encode
            del seq_post_encode
            
            re_loss = re_loss_fn(full_encode,re_encode).mean()

        tgt_out = tgt[1:, :]
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) + re_loss
        losses += loss.item()

    return losses / len(val_iter)


def custom_padding(seq,padding_value = PAD_IDX,batch_size=16):
    pre = []
    post = []

    for x in seq:
        pre.append(x[:int(len(x)/2)])
        post.append(x[int(len(x)/2):])


    max = 0
    for i in post:
        if max<len(i):
            max = len(i)        

    post_x = np.full((batch_size,max),1)
    for idx,i in enumerate(post):
        post_x[idx,:len(i)]=np.asarray(i)


    max = 0
    for i in pre:
        if max<len(i):
            max = len(i)        

    pre_x = np.full((batch_size,max),1)
    for idx,i in enumerate(pre):
        pre_x[idx,-len(i):]=np.asarray(i)

    post_seq = torch.tensor(post_x)
    pre_seq = torch.tensor(pre_x)

    full_seq = torch.cat([pre_seq,post_seq],dim=1)
    
    return full_seq.transpose(1,0)

def L1_Loss(A_tensors, B_tensors):
    
    return torch.abs(A_tensors - B_tensors)
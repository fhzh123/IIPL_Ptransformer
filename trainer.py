import time
import random
from util import *
from data_utils import *
import torch.optim as optim
from model.transformer import *
import nltk.translate.bleu_score as bs

SEED = 981126

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, file_path, num_epoch, lr,
                 emb_size, nhead, ffn_hid_dim, batch_size, 
                 n_layers, dropout, token_type, load, variation):
        self.params = {'num_epoch': num_epoch,
                       'emb_size': emb_size,
                       'nhead': nhead,
                       'ffn_hid_dim': ffn_hid_dim,
                       'batch_size': batch_size,
                       'n_layers': n_layers,
                       'dropout': dropout,
                       'lr': lr,
                       }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kor, eng = get_kor_eng_sentences(file_path)
        self.sentences = {'src_lang': kor, 'tgt_lang': eng}
        self.tokens = get_tokens(self.sentences, token_type)
        self.vocabs = build_vocabs(self.sentences, self.tokens)
        train_sentences, val_sentences, self.test_sentences = divide_sentences(self.sentences)
        self.train_iter = get_train_iter(train_sentences, self.tokens, self.vocabs, self.params['batch_size'])
        self.val_iter = get_test_iter(val_sentences, self.tokens, self.vocabs, self.params['batch_size'])
        self.test_iter = get_test_iter(self.test_sentences, self.tokens, self.vocabs, self.params['batch_size'])


        self.params['src_vocab_size'] = len(self.vocabs['src_lang'])
        self.params['tgt_vocab_size'] = len(self.vocabs['tgt_lang'])
  
        self.transformer = build_model(vocabs=self.vocabs, nhead=self.params['nhead'], N=self.params['n_layers'],
                                       d_model= self.params['emb_size'], d_ff=self.params['ffn_hid_dim'],
                                       device=self.device, dropout=self.params['dropout'], load=load, variation=variation)
        
        # self.optimizer = ScheduledOptim(
        #     optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=self.params['lr']),
        #     warmup_steps=4000,
        #     hidden_dim=self.params['ffn_hid_dim']
        # )
        self.optimizer = optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=self.params['lr'])

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def train(self):
        print("\nbegin training...")

        for epoch in range(self.params['num_epoch']):
            start_time = time.time()

            epoch_loss = train_loop(self.train_iter, self.transformer, self.optimizer, self.criterion, self.device)
            val_loss = val_loop(self.val_iter, self.transformer, self.criterion, self.device)

            end_time = time.time()

            if (epoch + 1) % 5 == 0:
                test(self.test_iter, self.transformer, self.criterion, self.device)

            # if (epoch + 1) % 10 == 0:
            #     get_bleu(self.test_sentences, self.transformer, self.vocabs, self.text_transform, self.device)

            minutes, seconds, time_left_min, time_left_sec = epoch_time(end_time-start_time, epoch, self.params['num_epoch'])
            
            print("Epoch: {} out of {}".format(epoch+1, self.params['num_epoch']))
            print("Train_loss: {} - Val_loss: {} - Epoch time: {}m {}s - Time left for training: {}m {}s"\
            .format(round(epoch_loss, 3), round(val_loss, 3), minutes, seconds, time_left_min, time_left_sec))

        torch.save(self.transformer.state_dict(), 'checkpoints/new_script_checkpoint_inf2.pth')
        torch.save(self.transformer, 'checkpoints/new_script_checkpoint_mod2.pt')

def train_loop(train_iter, model, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    for src, tgt in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1,:]

        optimizer.zero_grad()

        logits = model(src=src, tgt=tgt_input)

        output = logits.contiguous().reshape(-1, logits.shape[-1])
        target = tgt[1:,:].contiguous().reshape(-1)
        
        loss = criterion(output, target)                    
        loss.backward()
        
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_iter)

def val_loop(val_iter, model, criterion, device):
    model.eval()
    val_loss = 0

    for src, tgt in val_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        logits = model(src=src, tgt=tgt_input)

        output = logits.contiguous().reshape(-1, logits.shape[-1])
        target = tgt[1:,:].contiguous().reshape(-1)
        
        loss = criterion(output, target)   
        val_loss += loss.item()

    return val_loss / len(val_iter)

def test(test_iter, model, criterion, device):
    model.eval()
    test_loss = 0

    for src, tgt in test_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        logits = model(src=src, tgt=tgt_input)

        output = logits.contiguous().reshape(-1, logits.shape[-1])
        target = tgt[1:,:].contiguous().reshape(-1)
        
        loss = criterion(output, target)   
        test_loss += loss.item()
    test_loss /= len(test_iter)

    print("Test Loss: {}".format(round(test_loss, 3)))

def get_bleu(sentences, model, vocabs, text_transform, device):
    bleu_scores = 0
    chencherry = bs.SmoothingFunction()

    count = 0
    for ko, eng in zip(sentences['src_lang'], sentences['tgt_lang']):
        candidate = translate(model, ko, vocabs, text_transform, device).split()
        ref = eng.split()

        count += 1
        bleu_scores += bs.sentence_bleu([ref], candidate, smoothing_function=chencherry.method2) 
        print(bleu_scores)
        print(count)

    print('BLEU score -> {}'.format(bleu_scores/len(sentences['src_lang'])))


class ScheduledOptim:
    def __init__(self, optimizer, warmup_steps, hidden_dim):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.step_num = 0
        self.warmup_steps = warmup_steps
    
    def step(self):
        self.step_num += 1
        lr = self.init_lr * self.get_scale()
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr
            
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_scale(self):
        return np.min([
            np.power(self.step_num, -0.5),
            self.step_num * np.power(self.warmup_steps, -1.5)
        ])
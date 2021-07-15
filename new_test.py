# Import modules
import os
import gc
import time
import pickle
import sentencepiece as spm
import nltk.translate.bleu_score as bs
from util import BOS_IDX, EOS_IDX, generate_square_subsequent_mask

# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import CustomDataset



def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        for key in memory.keys():
          memory[key].to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(F.gelu(out[:, -1]))
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def get_bleu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join("./data/preprocessed", 'test_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        test_src_indices = data_['test_src_indices']
        test_trg_indices = data_['test_trg_indices']
        src_word2id = data_['src_word2id']
        trg_word2id = data_['trg_word2id']
        trg_id2word = {v: k for k, v in trg_word2id.items()}
        src_vocab_num = len(src_word2id)
        trg_vocab_num = len(trg_word2id)
        del data_

    test_dataset = CustomDataset(test_src_indices, test_trg_indices,
                                 min_len=4, src_max_len=50, trg_max_len=50)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=32, shuffle=False,
                                 pin_memory=True, num_workers=4)

    model = torch.load("./data/checkpoints/checkpoint.pt")
    model = model.eval()
    spm_trg = spm.SentencePieceProcessor()
    spm_trg.Load(f'{"./data/preprocessed"}/m_trg_{8000}.model')

    bleu_scores = 0

    candidate_token = []
    reference_token = []
    for de, eng in test_dataloader.dataset:
        ref = "".join([trg_id2word[ix] for ix in eng.cpu().numpy()]).replace("<s>", "").replace("</s>", "").replace("<pad>","").replace("<unk>", "").split('▁')[1:]
        reference_token.append(ref)
      
        num_tokens = de.shape[0]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)

        tgt_tokens = greedy_decode(
            model,  de, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device).flatten()
        candidate = "".join([trg_id2word[ix] for ix in tgt_tokens.cpu().numpy()]).replace("<s>", "").replace("</s>", "").replace("unk", "")

        candidate = candidate.split('▁')[1:]
          
        candidate_token.append(candidate)

    bleu_scores = bs.corpus_bleu(reference_token, candidate_token)
    print('BLEU score -> {}'.format(bleu_scores))
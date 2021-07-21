# Import modules
import os
from tokenizers import Tokenizer
import nltk.translate.bleu_score as bs
from util import BOS_IDX, EOS_IDX, generate_square_subsequent_mask
from tqdm import tqdm
# Import PyTorch
import torch
from torch.nn import functional as F

def get_bleu():
    model = torch.load("./data/checkpoints/checkpoint.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chencherry = bs.SmoothingFunction()
    bleu_scores = 0
    src_list, trg_list = [], []

    with open(os.path.join("./data/preprocessed", 'src_test.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            src_list.append(text)
        del data_

    with open(os.path.join("./data/preprocessed", 'trg_test.txt'), 'r') as f:
        data_ = f.readlines()
        for text in data_:
            trg_list.append(text)
        del data_

    for de, en in zip(src_list, trg_list):
        candidate = translate(model, de, device).split()
        reference = [en.split()]
        print(candidate, reference)

        bleu_scores += bs.sentence_bleu(reference, candidate, smoothing_function=chencherry.method2)

    print('BLEU score -> {}'.format(bleu_scores/len(src_list)))

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = F.log_softmax(model.generator(out[:, -1]), dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model, src_sentence, device):
    model.eval()
    de_tokenizer = Tokenizer.from_file(os.path.join(
        "./data/preprocessed", 'de_tokenizer.json'))
    en_tokenizer = Tokenizer.from_file(os.path.join(
        "./data/preprocessed", 'en_tokenizer.json'))

    src = torch.tensor(de_tokenizer.encode(src_sentence).ids).view(-1,1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device).flatten()
    return "".join(en_tokenizer.decode(list(tgt_tokens.cpu().numpy())))


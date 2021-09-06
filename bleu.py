import os
import torch
from typing import List
from torch.nn import functional as F
import nltk.translate.bleu_score as bs
from util import BOS_IDX, EOS_IDX, generate_square_subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    if isinstance(memory, List):
        for mem in memory:
            mem = mem.to(device)
    else:
        memory = memory.to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device = device)).type(torch.bool).to(device)
        out,_ = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src_sentence, vocabs, text_transform, device):
    model.eval()

    src = text_transform['src_lang'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device = device).flatten()
    return " ".join(vocabs['tgt_lang'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<unk>", "")


def get_bleu(model, test_iter, vocabs, text_transform, device):
    bleu_scores = 0
    chencherry = bs.SmoothingFunction()

    count = 0

    for de, en in test_iter:
        candidate = translate(model, de, vocabs, text_transform, device).split()
        reference = [en.split()]
        if reference[0][-1][-1] == ".":
            reference[0][-1].replace(".", "")
            reference[0].append(".")
        if count <= 10:
            print("cadidate :\n     ",candidate, "\n","reference :\n     ", reference)
            count += 1

        bleu_scores += bs.sentence_bleu(reference, candidate,
                                        smoothing_function=chencherry.method2)

    print('BLEU score -> {}'.format(bleu_scores/len(test_iter)))

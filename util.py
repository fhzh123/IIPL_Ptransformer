from vocabs import PAD_IDX, EOS_IDX, SOS_IDX
import torch.nn as nn
import torch
import copy

def clones(module, N):
    return nn.ModuleList([ copy.deepcopy(module) for _ in range(N) ])

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones((1, sz), device=device) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0,1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def epoch_time(time, curr_epoch, total_epochs):
    minutes = int(time / 60)
    seconds = int(time % 60)

    epoch_left = total_epochs - curr_epoch
    time_left = epoch_left * time
    time_left_min = int(time_left / 60) - minutes
    time_left_sec = int(time_left % 60)

    return minutes, seconds, time_left_min, time_left_sec    
    
def greedy_decode(model, src, src_mask, max_len, start_symbol, device, gen):
    src = src.to(device)
    src_mask = src_mask.to(device)
    trg = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        tgt_mask = (generate_square_subsequent_mask(trg.size(0), device)
                    .type(torch.bool)).to(device)
        # encoded = model.encode(src=src, src_mask=src_mask)
        with torch.no_grad():
            out = model.encode_decode(src = src,
                                      src_mask = src_mask,
                                      tgt = trg,
                                      tgt_mask = tgt_mask
                                      )
            # out = model.decode(x=trg, memory=encoded, src_mask=src_mask, tgt_mask=tgt_mask)
            out = out.transpose(0, 1)
            prob = gen(out[:,-1])
            # prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

        trg = torch.cat([trg,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return trg


# actual function to translate input sentence into target language
def translate(model, text_transform, src_sentence, vocabs, device):
    model.eval()
    gen = nn.Linear(512, len(vocabs['tgt_lang'])).to(device)
    src = text_transform['src_lang'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(1, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX, device=device, gen=gen).flatten()
    return " ".join(vocabs['tgt_lang'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")
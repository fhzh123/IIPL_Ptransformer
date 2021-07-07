import torch
import argparse
from util import *
from data_utils import *
from checkpoints import *
from model.transformer import *

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('checkpoints/new_script_checkpoint_mod2.pt')

    model.to(device)
    
    kor, eng = get_kor_eng_sentences("Default")
    sentences = {'src_lang': kor, 'tgt_lang': eng}
    tokens = get_tokens(sentences, 1)
    vocabs = build_vocabs(sentences, tokens)
    text_transform = get_text_transform(tokens, vocabs)

    output_sentence = translate(
        model = model, 
        vocabs = vocabs,
        text_transform = text_transform,
        src_sentence = args.input,
        device = device
        )

    print(output_sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate')
    parser.add_argument('--input', type=str, default="코카콜라는 맛있습니다.")
    args = parser.parse_args()
    main(args)

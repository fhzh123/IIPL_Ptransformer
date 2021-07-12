import pickle
import torchtext.vocab as tt

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def build_vocabs(sentences, tokens):
  vocabs = {}
  
  special_symbols =['<unk>', '<pad>', '<bos>', '<eos>']

  for ln in ['src_lang', 'tgt_lang']:
      vocabs[ln] = tt.build_vocab_from_iterator(iterator=yield_tokens(sentences, ln, tokens),
                                                min_freq=3,
                                                specials=special_symbols,
                                                special_first=True
                                                )
  return vocabs
    

def yield_tokens(data_iter, language, tokens):
    data_iter = data_iter[language]

    for data in data_iter:
        yield tokens[language](data)

def pickle_vocabs(vocabs):
  kor = vocabs['src_lang']
  eng = vocabs['tgt_lang']
  
  with open('./data/pickles/kor_vocab.pickle', 'wb') as kor_file:
    pickle.dump(kor, kor_file)
    
  with open('./data/pickles/eng_vocab.pickle', 'wb') as eng_file:
    pickle.dump(eng, eng_file)
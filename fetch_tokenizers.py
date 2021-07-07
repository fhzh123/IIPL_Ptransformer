from util import *
from pickles import *
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from torchtext.data.utils import get_tokenizer
from tokenizers import SentencePieceBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from torchtext.vocab import build_vocab_from_iterator

def get_ko_words(sentences):
  word_extractor = WordExtractor(min_frequency=5,
      min_cohesion_forward=0.05, 
      min_right_branching_entropy=0.0
  )
  word_extractor.train(sentences) # list of str
  words = word_extractor.extract()
  return words

def get_ko_tokenizer(sentences, load):
  if not load:
    words = get_ko_words(sentences)
    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    pickle_tokenizer(cohesion_score)
  else:
    kor_pikcle = open('pickle_files/tokenizer.pickle', 'rb')
    cohesion_score = pickle.load(kor_pikcle)

  return LTokenizer(scores=cohesion_score)

def get_BPE_tokenizer(sentences, vocab_size=20000, min_frequency=3):
  tokenizer = SentencePieceBPETokenizer()
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train_from_iterator(sentences, 
                               vocab_size=vocab_size,
                               min_frequency=min_frequency,
                               special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]'])

  return tokenizer

def build_vocabs(sentences, tokens):
  vocabs = {}

  special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

  for ln in ['src_lang', 'tgt_lang']:
    vocabs[ln] = build_vocab_from_iterator(iterator=yield_tokens(sentences, ln, tokens),
                                           min_freq=1,
                                           specials=special_symbols,
                                           special_first=True)

  for ln in ['src_lang', 'tgt_lang']:
    vocabs[ln].set_default_index(UNK_IDX)

  return vocabs
  
def yield_tokens(data_iter, language, tokens):
  data_iter = data_iter[language]

  for data_sample in data_iter:
      yield tokens[language](data_sample)

def get_tokens(sentences, token_type, load):
  tokens = {}
  if token_type == 2:
    ko_token = get_BPE_tokenizer(sentences['src_lang'])
    en_token = get_BPE_tokenizer(sentences['tgt_lang']) 
  else:
    ko_token = get_ko_tokenizer(sentences['src_lang'], load)
    en_token = get_tokenizer('spacy', language='en_core_web_sm')

  tokens = {'src_lang': ko_token, 'tgt_lang': en_token}
  return tokens

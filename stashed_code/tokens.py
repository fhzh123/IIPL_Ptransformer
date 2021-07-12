import pickle
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from torchtext.data.utils import get_tokenizer

def get_ko_tokenizer(sentences, load):
  if not load:
    words = get_ko_words(sentences)
    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    pickle_tokenizer(cohesion_score)
  else:
    kor_pikcle = open('./data/pickles/tokenizer.pickle', 'rb')
    cohesion_score = pickle.load(kor_pikcle)

  return LTokenizer(scores=cohesion_score)

def get_ko_words(sentences):
  word_extractor = WordExtractor(min_frequency=5,
      min_cohesion_forward=0.05, 
      min_right_branching_entropy=0.0
  )
  word_extractor.train(sentences)
  words = word_extractor.extract()
  return words

def pickle_tokenizer(cohesion_scores):
    with open('./data/pickles/tokenizer.pickle', 'wb') as pickle_out:
      pickle.dump(cohesion_scores, pickle_out)
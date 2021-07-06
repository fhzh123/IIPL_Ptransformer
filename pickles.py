import pickle

def pickle_tokenizer(cohesion_scores):
    with open('pickle_files/tokenizer.pickle', 'wb') as pickle_out:
      pickle.dump(cohesion_scores, pickle_out)

def pickle_vocabs(vocabs):
  kor = vocabs['src_lang']
  eng = vocabs['tgt_lang']
  
  with open('pickle_files/kor.pickle', 'wb') as kor_file:
    pickle.dump(kor, kor_file)
    
  with open('pickle_files/eng.pickle', 'wb') as eng_file:
    pickle.dump(eng, eng_file)
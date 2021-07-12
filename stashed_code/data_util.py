# import os
# import random
# import pandas as pd
# from tokens import *
# from vocabs import *
# from data_loader import *

# class Data:
#     def __init__(self, load, batch_size):
#         super(Data, self).__init__()
#         self.root = "./data/excels"
#         self.load = load
#         self.train, self.val, self.test = self.get_sentences()
#         self.tokens = self.get_tokens()
#         self.vocabs = self.get_vocabs()
#         self.train_iter = get_train_iter(
#                                          self.train,
#                                          self.tokens,
#                                          self.vocabs,
#                                          batch_size
#                                          )
#         self.val_iter = get_test_iter(
#                                      self.val,
#                                      self.tokens,
#                                      self.vocabs,
#                                      batch_size
#                                      )

#         self.test_iter = get_test_iter(
#                                      self.test,
#                                      self.tokens,
#                                      self.vocabs,
#                                      batch_size
#                                      )

#     def get_sentences(self):
#         csv_files = None
#         if not self.load:
#             files = os.listdir("./data/excels")
#             csv_files = self.convert_to_csv(files)
            
#         sentences = self.convert_to_sentences(csv_files)
#         return sentences

#     def convert_to_csv(self, files):
#         count = 1
#         csv_files = []
#         for filepath in files:
#             if filepath[-4:] == "xlsx":
#                 xls = pd.read_excel(os.path.join(self.root, filepath), index_col=None)
#                 destination = 'data/csvs/korean_to_english' + str(count) + '.csv'
#                 xls.to_csv(destination, encoding='utf-8', index=False)
#                 csv_files.append(destination)
        
#         return csv_files

#     def convert_to_sentences(self, csv_files=None):
#         kor_sentences, eng_sentences = [], []

#         if not self.load:
#             dataframes = [ pd.read_csv(filepath) for filepath in csv_files ]

#             for data in dataframes:
#                 for index, sent in data.iterrows():
#                     _, kor, eng = sent
#                     kor_sentences.append(kor)
#                     eng_sentences.append(eng)
#             with open('./data/pickles/kor_sent.pickle', 'wb') as kor_out:
#                 pickle.dump(kor_sentences, kor_out)
#             with open('./data/pickles/eng_sent.pickle', 'wb') as eng_out:
#                 pickle.dump(eng_sentences, eng_out)
#         else:
#             file_kor = open('./data/pickles/kor_sent.pkl', 'rb')
#             kor_sentences = pickle.load(file_kor)
#             file_eng = open('./data/pickles/eng_sent.pkl', 'rb')
#             eng_sentences = pickle.load(file_eng)
            
#         for kor, eng in zip(kor_sentences[:5], eng_sentences[:5]):
#             print(f'[KOR]: {kor}')
#             print(f'[ENG]: {eng}\n')

#         print(f'[KOR LEN]: {len(kor_sentences)}')
#         print(f'[ENG LEN]: {len(eng_sentences)}\n')

#         sentences = {'src_lang': kor_sentences,
#                      'tgt_lang': eng_sentences}

#         return self.divide_sentences(sentences)

            
#     def divide_sentences(self, sentences):
#         train, val, test = {}, {}, {}
    
#         for ln in ['src_lang', 'tgt_lang']:
#             temp = sentences[ln]
#             random.shuffle(temp)
#             train_len = int(len(temp)*0.8)
#             val_len = int(len(temp)*0.1)+train_len
#             test_len =  int(len(temp)*0.1)+val_len+train_len
#             tmp_train,tmp_val,tmp_test = temp[0:train_len], temp[train_len:val_len], temp[val_len:test_len]

#             train[ln], val[ln], test[ln] = tmp_train, tmp_val, tmp_test

#         print("\ntrain data length: {}".format(len(train['src_lang'])))
#         print("validation data length: {}".format(len(val['src_lang'])))
#         print("test data length: {}\n".format(len(test['src_lang'])))

#         return train, val, test
    
#     def get_tokens(self):
#         tokens = {}

#         tokens['src_lang'] = get_ko_tokenizer(self.train['src_lang'], self.load)
#         tokens['tgt_lang'] = get_tokenizer('spacy', language='en_core_web_sm')

#         return tokens

#     def get_vocabs(self):

#         vocabs = build_vocabs(self.train, self.tokens)
#         pickle_vocabs(vocabs)

#         return vocabs


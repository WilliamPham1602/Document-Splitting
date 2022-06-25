from bs4 import BeautifulSoup
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
import random

random.seed(42)

## length of the vectorized feature for each document, we set it to 300, 512, 1024,... but found that 512 worked best.
feature_length = 512

## clean the text by removing some redundant text such as spaces and changing all text from capital to lower cases
def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'\\n', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

## this is for debugging
def print_complaint(df, index):
    example = df[df.index == index][["labels", "text"]].values[0]
    if len(example) > 0:
        print(example[1])
        print('labels:', example[0])

## preparing the dataset by reading the preprocessed csv files which contain filenames and extracted text
## then apply the clean text function and save it to a new column "text_processed"
def prepare_data_set(data_dir):
    df_0 = pd.read_csv(data_dir + '0.0_0502.csv')
    df_1 = pd.read_csv(data_dir + '1.0_0502.csv')
    frames = [df_0, df_1]
    df = pd.concat(frames)
    df.reset_index()
    df.fillna('', inplace=True)
    df['text_processed'] = df['text'].apply(cleanText)
    print('length df: ', len(df))
    return df


## Split the original dataset to train set and test set
def split_train_test(dataframe, test_size=0.3, rd_state=42):
    train, test = train_test_split(dataframe, test_size=test_size, random_state=rd_state)
    return train, test

## Text tokenization
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

## Class to train the doc2vec model
class doc2vec_Trainer(object):

    def __init__(self, train_opt):
        self.model_name = train_opt.model
        self.data_dir = train_opt.dataset_dir
        self.model_save_dir = train_opt.model_dir
        self.vector_size = train_opt.vector_size
        self.epochs = train_opt.epochs
        self.alpha = train_opt.alpha
        self.distributed = train_opt.distributed

        # prepare dataset
        df = prepare_data_set(self.data_dir)
        # split dataset to train set and test set
        train = df
        # tokenize train set
        self.train_tagged = train.apply(
            lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)

        cores = multiprocessing.cpu_count()

        # we have two choices, using distributed memory (dm) or distributed bag of word (dw), here we train both of them to get 2 models 
        # by replacing the distributed argument in doc2vec_train.py file.
        if self.distributed == 'dm':
            self.d2v_Model = Doc2Vec(dm=1, dm_mean=1, vector_size=self.vector_size, window=10, negative=5, min_count=1, workers=cores,
                                alpha=0.065, min_alpha=0.065)
        else:
            self.d2v_Model = Doc2Vec(dm=0, vector_size=self.vector_size, negative=5, hs=0, min_count=2, sample=0, workers=cores)

        self.d2v_Model.build_vocab([x for x in tqdm(self.train_tagged.values)])


    def start_train(self):
        for _ in range(self.epochs):
            self.d2v_Model.train(utils.shuffle([x for x in tqdm(self.train_tagged.values)]),
                                 total_examples=len(self.train_tagged.values), epochs=1)
            self.d2v_Model.alpha -= self.alpha
            self.d2v_Model.min_alpha = self.d2v_Model.alpha

        self.save_model(self.model_save_dir)
        print('Training finish, model was saved at ./models/')

    def save_model(self, save_dir):
        fname = save_dir + 'pdf_split_d2v_{}_{}_{}_0502.mod'.format(self.model_name, self.vector_size, self.distributed)
        self.d2v_Model.save(fname)






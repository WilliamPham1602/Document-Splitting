import re
import nltk
import pickle
import pypdfium2
import pytesseract
import numpy as np
import pandas as pd

from PIL import Image
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from stop_words import get_stop_words
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def tag_page(prediction):
    """
    :param prediction: classify prediction array: e.g [1, 0, 0, 1, 0, 0, 1, 0, 1]
    :return: tag page: e.g. [3, 3, 2, 1]
    """
    tag = np.split(prediction, np.argwhere(prediction == 1).flatten())
    tag = [len(tag[i]) for i in range(len(tag)) if len(tag[i])]
    tag = np.array(tag)

    return tag

def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'\\n', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

class Prediction():
    def __init__(self, pdf_name, vectorizer_type='gensim', model_type='log'):
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.type = type
        self.pdf_name = pdf_name
        self.output = []
        self.feature_length = 2048

        self.model_dbow = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_db.mod".format(self.feature_length))
        self.model_dmm = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_dm.mod".format(self.feature_length))
        self.vectorizer_gensim = ConcatenatedDoc2Vec([self.model_dbow, self.model_dmm])
        self.model_lstm = None
        self.model_bert = None
        with open('models/logreg/logreg_model_2048__dbow_dm_concate.sav', 'rb') as m:
            self.model_log = pickle.load(m)
    
    def run_batch(self):
        result = []
        for image, _ in pypdfium2.render_pdf_topil(self.pdf_name):
            result.append(self.run_single(image))
        gold_label = tag_page(np.array(result))
        return gold_label


    def run_single(self, image):
        text = str(((pytesseract.image_to_string(image))))
        text = cleanText(text)
        text_list = tokenize_text(text)

        if self.vectorizer_type == 'gensim':
            vectors = self.vectorizer_gensim.infer_vector(text_list)
        else:
            pass

        if self.model_type == 'log':
            return list(self.model_log.predict([vectors]))[0]
        elif self.model_type == 'bert':
            pass
        else:
            pass


if __name__ == '__main__':
    prediction = Prediction('corpus1/TrainTestSet/Trainset/data/868212__concatenated.pdf')
    print(prediction.run_batch())
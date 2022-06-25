import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import random
random.seed(42)

feature_length = 512

model_dbow = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_db.mod".format(feature_length))
model_dmm = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_dm.mod".format(feature_length))

# Concatenate model
model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

# def get_vectors(model, tagged_docs):
#     sents = tagged_docs.values
#     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
#     return targets, regressors


def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'\\n', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    # text = text.replace('x', '')
    return text

def print_complaint(df, index):
    example = df[df.index == index][["labels", "text"]].values[0]
    if len(example) > 0:
        print(example[1])
        print('labels:', example[0])

test_df0 = pd.read_csv('./data/0.0_0519.csv')
test_df1 = pd.read_csv('./data/1.0_0519.csv')
df_test = pd.concat([test_df0, test_df1])

df_0 = pd.read_csv('./data/0.0_ocred_075.csv')
df_1 = pd.read_csv('./data/1.0_ocred_075.csv')

frames = [df_0, df_1]

df = pd.concat(frames)
df.reset_index()
df.head()

df.fillna('', inplace=True)
df.head()

df_test.reset_index()
df_test.head()

df_test.fillna('', inplace=True)
df.head()

df['text_processed'] = df['text'].apply(cleanText)
df_test['text_processed'] = df_test['text'].apply(cleanText)

from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

stop_words = get_stop_words('dutch')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 5), stop_words=stop_words, max_features=20)
features = tfidf.fit_transform(df['text_processed']).toarray()
labels = df['labels']
features_test = tfidf.fit_transform(df_test['text_processed']).toarray()
labels_test = df_test['labels']

X_train, _, y_train, _ = train_test_split(features, labels, test_size=0.2, random_state=42)
_, X_test, _, y_test = train_test_split(features_test, labels_test, test_size=0.2, random_state=42)


a = 1.0
b = len(y_train[y_train==0.0]) / len(y_train[y_train==1.0])
weights = {0.0:a, 1.0:b}

logreg = LogisticRegression(n_jobs=1, C=1e5, multi_class='ovr', class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

filename = './models/logreg/logreg_tfidf_model_dbow_dm_concate_75.sav'.format(feature_length)
pickle.dump(logreg, open(filename, 'wb'))

from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

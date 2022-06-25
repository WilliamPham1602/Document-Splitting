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

feature_length = 1024

model_dbow = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_{}_db.mod".format(feature_length))
model_dmm = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_{}_dm.mod".format(feature_length))

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
    text = text.replace('x', '')
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

train, _ = train_test_split(df, test_size=0.3, random_state=42, stratify=df.labels)

_, test = train_test_split(df_test, test_size=0.3, random_state=42, stratify=df_test.labels)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)


y_train, X_train = vec_for_learning(model, train_tagged)
y_test, X_test = vec_for_learning(model, test_tagged)


logreg = LogisticRegression(n_jobs=1, C=1e5, multi_class='multinomial')
logreg.fit(X_train, y_train)
# with open('models/logreg/logreg_model_1024__dbow_dm_concate.sav', 'rb') as m:
#     logreg = pickle.load(m)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

filename = './models/logreg/logreg_model_{}_dbow_dm_concate_75.sav'.format(feature_length)
pickle.dump(logreg, open(filename, 'wb'))

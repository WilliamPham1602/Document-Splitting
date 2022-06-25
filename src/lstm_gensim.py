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

import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 

import random
random.seed(42)

feature_length = 1024

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
    text = text.replace('x', '')
    return text

def print_complaint(df, index):
    example = df[df.index == index][["labels", "text"]].values[0]
    if len(example) > 0:
        print(example[1])
        print('labels:', example[0])

df_0 = pd.read_csv('./data/0.0_ocred.csv')
df_1 = pd.read_csv('./data/1.0_ocred.csv')

frames = [df_0, df_1]

df = pd.concat(frames)
df.reset_index()
df.head()

df.fillna('', inplace=True)
df.head()

df['text_processed'] = df['text'].apply(cleanText)

train, test = train_test_split(df, test_size=0.3, random_state=42)

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


# logreg = LogisticRegression(n_jobs=1, C=1e5, multi_class='multinomial')
# logreg.fit(X_train, y_train)

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train)).type(torch.LongTensor)
y_test_tensors = Variable(torch.Tensor(y_test)).type(torch.LongTensor)

#reshaping to rows, timestamps, features

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))


X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

num_epochs = 5000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 2048 #number of features
hidden_size = 256 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 2 #number of output classes

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
criterion = torch.nn.CrossEntropyLoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
    
    loss.backward() #calculates the loss of the loss function
    
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

lstm1.eval()
import numpy as np

y_predict = lstm1(X_test_tensors_final).data.numpy()
y_pred = np.argmax(y_predict, axis=-1)


from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

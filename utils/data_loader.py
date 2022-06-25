from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from transformers import BertTokenizer
from transformers import AutoTokenizer
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import nltk

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased", revision="v1")

model_dbow = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_db.mod")
model_dmm = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_dm.mod")
feature_length = 2048
d2v_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

class load_data_from_tfidf(Dataset):
    def __init__(self, x_data, y_data, transform=None):

        self.x_data = x_data
        self.y_data = np.array(y_data)
        self.transform = transform

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        inputs, labels = self.to_array(idx)
        if self.transform is not None:
            data_loader = {'X': inputs, 'y': labels}
            data_loader = self.transform(data_loader)
            inputs, labels = data_loader['X'], data_loader['y']
        # return data_loader
        return inputs, labels

    def to_array(self, idx):
        x_array = self.x_data[idx].toarray()
        # x_array = np.squeeze(x_array, axis=0)
        x_array = torch.from_numpy(x_array).to(torch.float32)

        y_array = self.y_data[idx]
        return x_array, y_array

class load_data_lstm_v2(Dataset):
    def __init__(self, text, label, path, transform=None):
        self.text = text
        self.label = label
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img, label = self.get_image(idx)
        gensim_id, bert_mask = self.get_gensim(idx), 0
        data_loader = {'img': img, 'id': gensim_id, 'mask': bert_mask, 'lb': label}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def get_image(self, idx):
        file_name = self.path[idx]
        file_name = file_name.split('/')[-1]
        label = self.label[idx]
        path = '../data/images/test/{}'.format(file_name)
        img = cv2.imread(path)
        return img, label

    def get_gensim(self, idx):
        text = self.text[idx]
        text = self.tokenize_text(text)
        vector = d2v_model.infer_vector(text)
        return vector

    def tokenize_text(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

class load_data_bert(Dataset):
    def __init__(self, text, label, path, transform=None):
        self.text = text
        self.label = label
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img, label = self.get_image(idx)
        bert_id, bert_mask = self.get_bert(idx)
        data_loader = {'img': img, 'id': bert_id, 'mask': bert_mask, 'lb': label}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def get_image(self, idx):
        file_name = self.path[idx]
        file_name = file_name.split('/')[-1]
        label = self.label[idx]
        path = '../data/images/test/{}'.format(file_name)
        img = cv2.imread(path)
        return img, label

    def get_bert(self, idx):
        text = self.text[idx]
        bert_input = tokenizer(text, padding='max_length', max_length=256,
                               truncation=True, return_tensors="pt")
        id = bert_input['input_ids']
        mask = bert_input['attention_mask']
        return id, mask

class load_data_vgg_lstm(Dataset):
    def __init__(self, vec, label, path, transform=None):
        self.vec = vec
        self.label = label
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img, label = self.get_image(idx)
        gensim_id, bert_mask = self.vec[idx], 0
        data_loader = {'img': img, 'id': gensim_id, 'mask': bert_mask, 'lb': label}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def get_image(self, idx):
        file_name = self.path[idx]
        file_name = file_name.split('/')[-1]
        label = self.label[idx]
        path = '../data_legacy/{}'.format(file_name)
        img = cv2.imread(path)
        return img, label


class load_data_with_path(Dataset):
    def __init__(self, data, label, fn):

        self.x_data = data
        self.y_data = label
        self.path = fn
        # self.d2v = d2v_model

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        X = self.x_data[idx]
        y = self.y_data[idx]
        file_name = self.path[idx]
        file_name = file_name.split('/')[-1]
        data_loader = {'X': X, 'y': y, 'fn': file_name}
        return data_loader


from bs4 import BeautifulSoup
import re
import nltk
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from stop_words import get_stop_words
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from networks.mlp_custom import MLP
from networks.lstm import LSTMNet
from networks.vgg_bert_custom import vgg_bert
from networks.vgg_lstm import Vgg_Lstm
from networks.effb0_bert import Effb0_Bert
from tqdm.autonotebook import tqdm
import traceback
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
from utils.loss import FocalLoss, SharpnessAwareMinimization
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from utils.data_loader import load_data_from_tfidf, load_data_bert, load_data_vgg_lstm, load_data_lstm_v2
from utils.transform import resize_data, Resize_img, Normalizer, vision_aug

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from torchvision import transforms
import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
# from nlpaug.util import Action
import json
from networks.vgg_only import Vgg_Only
from networks.lstm_only import LSTM_Only
from networks.bert_only import Bert_Only

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'\\n', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def calculate_accuracy(y_pred, y):
    y_pred = F.softmax(y_pred, dim=1)
    top_pred = y_pred.argmax(1, keepdim=True)
    y = y.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def calculate_confusion_maxtrix(y_pred, y, n_cls):
    cm = np.zeros((n_cls, n_cls))
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred.argmax(1, keepdim=True).long().cpu().numpy()
    y = y.argmax(1, keepdim=True)
    y = y.long().cpu().numpy()
    # y = np.expand_dims(y, axis=1)
    for i in range(y.shape[0]):
        cm[y[i, 0], y_pred[i, 0]] += 1
    return cm

def augment_text(df, aug_w2v, samples=1500, pr=0.2, label = 1):
    # aug_w2v.aug_p = pr
    new_text = []
    new_path = []

    ##selecting the minority class samples
    df_n = df[df.labels == label].reset_index(drop=True)

    ## data augmentation loop
    for i in tqdm(np.random.randint(0, len(df_n), samples)):
        text = df_n.iloc[i]['text_processed']
        augmented_text = aug_w2v.augment(text)
        new_text.append(augmented_text)
        path = df_n.iloc[i]['paths']
        new_path.append(path)

    ## dataframe
    new = pd.DataFrame({'text_processed': new_text, 'labels': label, 'paths': new_path})
    df = shuffle(df.append(new).reset_index(drop=True))
    return df

def over_sampling(X_train, y_train):
    difference = sum((y_train==0)*1) - sum((y_train==1)*1)
    indices = torch.where(y_train==1)[0]
    rand_subsample = torch.randint(0, len(indices), (difference,))
    X_train, y_train = torch.cat((X_train, X_train[indices[rand_subsample]])), torch.cat((y_train, y_train[indices[rand_subsample]]))
    return X_train, y_train

class mlp_Trainer(object):

    def __init__(self, train_opt):
        # Init
        self.model_name = train_opt.model
        self.data_dir = train_opt.dataset_dir
        self.model_save_dir = train_opt.save_dir + '{}_{}/'.format(train_opt.loss, train_opt.project)
        self.test_size = train_opt.test_split
        self.batch_size = train_opt.batch_size
        self.epochs = train_opt.epochs
        self.lr = train_opt.l_r
        self.device = train_opt.device
        self.num_cls = train_opt.num_cls

        self.logs = self.model_save_dir + 'logs/'
        os.makedirs(self.logs, exist_ok=True)
        self.writer = SummaryWriter(self.logs)

        # Save train params
        with open('{}/train_params.txt'.format(self.model_save_dir), 'w') as f:
            json.dump(train_opt.__dict__, f, indent=2)

        # using tfidf
        if train_opt.model=='tfidf':
            # Load dataset
            df_0 = pd.read_csv('./data/0.0_0502.csv')
            df_1 = pd.read_csv('./data/1.0_0502.csv')
            frames = [df_0, df_1]
            df = pd.concat(frames)
            df.reset_index()
            df.head()
            df.fillna('', inplace=True)
            df.head()
            df['text_processed'] = df['text'].apply(cleanText)
            stop_words = get_stop_words('dutch')

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 5), stop_words=stop_words)
            # # features = tfidf.fit_transform(df['text_processed']).toarray()
            features = tfidf.fit_transform(df['text_processed'])
            # a = tfidf.get_feature_names_out()

            # vectorizer = HashingVectorizer(n_features=train_opt.ft_size, norm='l2', encoding='latin-1', ngram_range=(1, 5), stop_words=stop_words)
            # features = vectorizer.fit_transform(df['text_processed'])

            labels = df['labels']
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

            _transforms = [
                resize_data(size=train_opt.ft_size)
            ]
            train_dataset = load_data_from_tfidf(x_data=X_train, y_data=y_train, transform=transforms.Compose(_transforms))
            valid_dataset = load_data_from_tfidf(x_data=X_test, y_data=y_test, transform=transforms.Compose(_transforms))

            self.in_features = train_opt.ft_size

        elif train_opt.model=='bert' or train_opt.model=='vgg_lstm' or train_opt.model=='vgg_only' or train_opt.model=='lstm_only' or train_opt.model=='bert_only':
            df_0 = pd.read_csv('./data/0.0_ocred_075.csv')
            df_1 = pd.read_csv('./data/1.0_ocred_075.csv')
            frames = [df_0, df_1]
            df = pd.concat(frames)
            df.reset_index()
            df.head()
            df.fillna('', inplace=True)
            df.head()
            df['text_processed'] = df['text'].apply(cleanText)

            train, test = train_test_split(df, test_size=0.3, random_state=42)

            if train_opt.augment:
                aug1 = naw.RandomWordAug()
                aug2 = naw.SplitAug()
                aug3 = nac.RandomCharAug(action="swap")
                aug4 = nac.RandomCharAug(action="delete")
                train = augment_text(df=train, aug_w2v=aug1, samples=2000)
                train = augment_text(df=train, aug_w2v=aug2, samples=2000)
                train = augment_text(df=train, aug_w2v=aug3, samples=2000)
                train = augment_text(df=train, aug_w2v=aug4, samples=2000)

            # train['vecs'] = train['text_processed'].apply(text2vector)
            # test['vecs'] = test['text_processed'].apply(text2vector)

            tr_transform = [
                # vision_aug(rota=train_opt.rotate, crop=0),
                Resize_img(img_size=224),
                Normalizer()
            ]

            if train_opt.model == 'bert' or train_opt.model == 'effb0' or train_opt.model == 'bert_only':
                path_train = train['paths'].tolist()
                text_train = train['text_processed'].tolist()
                label_train = train['labels'].tolist()

                path_test = test['paths'].tolist()
                text_test = test['text_processed'].tolist()
                label_test = test['labels'].tolist()

                train_dataset = load_data_bert(text=text_train, label=label_train,
                                               path=path_train, transform=transforms.Compose(tr_transform))
                valid_dataset = load_data_bert(text=text_test, label=label_test,
                                               path=path_test, transform=transforms.Compose(tr_transform))
            else:
                path_train = train['paths'].tolist()
                text_train = train['text_processed'].tolist()
                label_train = train['labels'].tolist()

                path_test = test['paths'].tolist()
                text_test = test['text_processed'].tolist()
                label_test = test['labels'].tolist()

                train_dataset = load_data_lstm_v2(text=text_train, label=label_train,
                                               path=path_train, transform=transforms.Compose(tr_transform))
                valid_dataset = load_data_lstm_v2(text=text_test, label=label_test,
                                               path=path_test, transform=transforms.Compose(tr_transform))
                # model_dbow = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_db.mod")
                # model_dmm = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_dm.mod")
                # feature_length = 2048
                # self.in_features = feature_length
                # self.d2v_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
                #
                # train_tagged = train.apply(
                #     lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)
                # test_tagged = test.apply(
                #     lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)
                #
                # y_train, X_train = vec_for_learning(self.d2v_model, train_tagged)
                # y_test, X_test = vec_for_learning(self.d2v_model, test_tagged)
                #
                # X_train = np.array(X_train)
                # y_train = np.array(y_train)
                #
                # X_test = np.array(X_test)
                # y_test = np.array(y_test)
                #
                # path_train = train['paths'].tolist()
                # path_test = test['paths'].tolist()
                #
                #
                # train_dataset = load_data_vgg_lstm(vec=X_train, label=y_train,
                #                                path=path_train, transform=transforms.Compose(tr_transform))
                # valid_dataset = load_data_vgg_lstm(vec=X_test, label=y_test,
                #                                path=path_test, transform=transforms.Compose(tr_transform))

        else:
            # Load D2V model
            model_dbow = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_db.mod")
            model_dmm = Doc2Vec.load("./models/gensim/pdf_split_d2v_gensim_1024_dm.mod")
            feature_length = 2048
            self.in_features = feature_length
            # model_dbow = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_db.mod".format(feature_length))
            # model_dmm = Doc2Vec.load("./models/pdf_split_d2v_gensim_{}_dm.mod".format(feature_length))
            d2v_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

            # Load dataset
            df_0 = pd.read_csv('./data/0.0_0502.csv')
            df_1 = pd.read_csv('./data/1.0_0502.csv')
            # df_2 = pd.read_csv('./data/2.0.csv')
            frames = [df_0, df_1]
            df = pd.concat(frames)
            df.reset_index()
            df.head()
            df.fillna('', inplace=True)
            df.head()
            df['text_processed'] = df['text'].apply(cleanText)
            train, test = train_test_split(df, test_size=0.3, random_state=42)

            if train_opt.augment:
                aug1 = naw.RandomWordAug()
                aug2 = naw.SplitAug()
                aug3 = nac.RandomCharAug(action="swap")
                aug4 = nac.RandomCharAug(action="delete")
                # aug5 = naw.BackTranslationAug(
                #     from_model_name='facebook/wmt19-en-de',
                #     to_model_name='facebook/wmt19-de-en'
                # )
                # aug6 = nas.AbstSummAug(model_path='t5-base')

                train = augment_text(df=train, aug_w2v=aug1, samples=2000)
                train = augment_text(df=train, aug_w2v=aug2, samples=2000)
                train = augment_text(df=train, aug_w2v=aug3, samples=2000)
                train = augment_text(df=train, aug_w2v=aug4, samples=2000)
                # train = augment_text(df=train, aug_w2v=aug5, samples=10)
                # train = augment_text(df=train, aug_w2v=aug6, samples=10)

            train_tagged = train.apply(
                lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)
            test_tagged = test.apply(
                lambda r: TaggedDocument(words=tokenize_text(r['text_processed']), tags=[r.labels]), axis=1)

            y_train, X_train = vec_for_learning(d2v_model, train_tagged)
            y_test, X_test = vec_for_learning(d2v_model, test_tagged)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test)

            if train_opt.oversampling:
                X_train, y_train = over_sampling(X_train, y_train)

            train_dataset = TensorDataset(X_train, y_train)

            valid_dataset = TensorDataset(X_test, y_test)


        self.training_generator = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                             num_workers=train_opt.num_worker)

        self.val_generator = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=train_opt.num_worker)

        # Load classifier model
        if self.model_name == 'mlp':
            self.model = MLP(input_dim=self.in_features, output_dim=self.num_cls)
        elif self.model_name == 'bert':
            self.model = vgg_bert(in_ch=3, out_ch=self.num_cls)
        elif self.model_name == 'bert_only':
            self.model = Bert_Only(in_ch=3, out_ch=self.num_cls)
        elif self.model_name == 'effb0':
            self.model = Effb0_Bert(out_ch=self.num_cls, language=train_opt.language, pre_trained=train_opt.pretrained)
        elif self.model_name == 'vgg_lstm':
            self.model = Vgg_Lstm(img_dim=3, embedding_dim=2048, output_dim=self.num_cls)
        elif self.model_name == 'vgg_only':
            self.model = Vgg_Only(img_dim=3, embedding_dim=2048, output_dim=self.num_cls)
        elif self.model_name == 'lstm_only':
            self.model = LSTM_Only(img_dim=3, embedding_dim=2048, output_dim=self.num_cls)
        else:
            self.model = LSTMNet(embedding_dim=self.in_features, output_dim=self.num_cls)

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        if train_opt.weighted != '':
            weight = torch.load(train_opt.weighted, map_location=self.device)
            self.model.load_state_dict(weight, strict=False)

        # Loss and optimizer
        self.optim = train_opt.optim

        if train_opt.loss == 'focal':
            self.criterion = FocalLoss(alpha=train_opt.alpha, gamma=2, reduction='mean').cuda() if torch.cuda.is_available() else FocalLoss(alpha=train_opt.alpha, gamma=2, reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss()

        if self.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'sam':
            base_optimizer = torch.optim.SGD
            self.optimizer = SharpnessAwareMinimization(self.model.parameters(), base_optimizer,
                                                   clip_norm=True, lr=self.lr, momentum=0.9)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10,
                                                                              T_mult=2)


        self.num_iter_per_epoch = len(self.training_generator)
        self.step = 0
        self.best_loss = 1e5
        self.best_acc = 0


    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.training_generator)
        epoch_loss = []
        epoch_acc = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                if self.model_name != 'bert' and self.model_name != 'vgg_lstm' and self.model_name != 'effb0' and self.model_name != 'vgg_only' and self.model_name != 'lstm_only' and self.model_name != 'bert_only':
                    inputs, labels = data
                    labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()
                    if 'lstm' in self.model_name:
                        inputs = torch.unsqueeze(inputs, dim=1)
                    # labels = labels.long()
                    # labels = torch.unsqueeze(labels, dim=1)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                else:
                    images, labels, b_ids, b_ms = data['img'], data['lb'], data['id'], data['mask']
                    images = images.permute(0, 3, 1, 2)
                    labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()

                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    b_ids = b_ids.to(self.device)
                    b_ms = b_ms.to(self.device)

                    if self.model_name == 'bert' or self.model_name == 'effb0' or self.model_name == 'bert_only':
                        b_ids = torch.squeeze(b_ids)
                        b_ms = torch.squeeze(b_ms)
                        outputs = self.model(images, b_ids, b_ms)
                    else:
                        b_ids = torch.unsqueeze(b_ids, dim=1)
                        outputs = self.model(images, b_ids)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)

                loss.backward()

                if self.optim == 'sam':
                    self.optimizer.first_step(zero_grad=True)
                    if self.model_name == 'vgg_lstm' or self.model_name == 'lstm_only' or self.model_name == 'vgg_only':
                        outputs = self.model(images, b_ids)
                    elif self.model_name == 'bert' or self.model_name == 'effb0' or self.model_name == 'bert_only':
                        outputs = self.model(images, b_ids, b_ms)
                    else:
                        outputs = self.model(inputs)
                    loss2 = self.criterion(outputs, labels)
                    loss2.backward()
                    self.optimizer.second_step(zero_grad=True, clip_norm=True)
                else:
                    self.optimizer.step()

                self.scheduler.step(epoch + iter / self.num_iter_per_epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, self.step)

                acc = calculate_accuracy(outputs, labels)
                epoch_acc.append(acc.item())

                epoch_loss.append(loss.item())
                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {}. Acc: {}'.format(
                        self.step, epoch+1, self.epochs, iter + 1, self.num_iter_per_epoch, loss, acc)
                progress_bar.set_description(descriptor)
                self.step += 1

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        mean_loss = np.mean(epoch_loss)
        mean_acc = np.mean(epoch_acc)
        train_descrip = '[Train] Epoch: {}. Mean Loss: {}. Mean Acc: {}'.format(epoch+1, mean_loss, mean_acc)
        print(train_descrip)
        self.writer.add_scalars('Loss', {'train': mean_loss}, epoch)
        self.writer.add_scalars('Accuracy', {'train': mean_acc}, epoch)

    def validation(self, epoch):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)
        epoch_loss = []
        epoch_acc = []

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    if self.model_name != 'bert' and self.model_name != 'vgg_lstm' and self.model_name != 'effb0' and self.model_name != 'vgg_only' and self.model_name != 'lstm_only' and self.model_name != 'bert_only':
                        inputs, labels = data
                        labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()
                        if 'lstm' in self.model_name:
                            inputs = torch.unsqueeze(inputs, dim=1)
                        # labels = labels.long()
                        # labels = torch.unsqueeze(labels, dim=1)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                    else:
                        images, labels, b_ids, b_ms = data['img'], data['lb'], data['id'], data['mask']
                        images = images.permute(0, 3, 1, 2)
                        labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()

                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        b_ids = b_ids.to(self.device)
                        b_ms = b_ms.to(self.device)

                        if self.model_name == 'bert' or self.model_name == 'effb0' or self.model_name == 'bert_only':
                            b_ids = torch.squeeze(b_ids)
                            b_ms = torch.squeeze(b_ms)
                            outputs = self.model(images, b_ids, b_ms)
                        else:
                            b_ids = torch.unsqueeze(b_ids, dim=1)
                            outputs = self.model(images, b_ids)

                    loss = self.criterion(outputs, labels)
                    acc = calculate_accuracy(outputs, labels)

                    epoch_acc.append(acc.item())
                    epoch_loss.append(loss.item())

                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {}. Acc: {}'.format(
                        epoch * len(progress_bar) + iter, epoch, self.epochs, iter + 1, len(progress_bar), loss, acc)

                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        val_epoch_loss = np.mean(epoch_loss)
        val_epoch_acc = np.mean(epoch_acc)
        val_descrip = '\n[Validation] Epoch: {}. Mean Loss: {}. Mean Acc: {}'.format(epoch+1, val_epoch_loss, val_epoch_acc)
        print(val_descrip)

        self.writer.add_scalars('Loss', {'val': val_epoch_loss}, epoch)
        self.writer.add_scalars('Accuracy', {'val': val_epoch_acc}, epoch)

        self.save_checkpoint(self.model, self.model_save_dir, 'last.pt')

        if self.best_loss > val_epoch_loss:
            self.best_loss = val_epoch_loss
            self.save_checkpoint(self.model, self.model_save_dir, 'best_val_loss.pt')

        if self.best_acc < val_epoch_acc:
            self.best_acc = val_epoch_acc
            self.save_checkpoint(self.model, self.model_save_dir, 'best_val_acc.pt')

    def test_model(self):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)

        confusion_matrix = np.zeros((self.num_cls, self.num_cls))

        y_true = []
        y = []
        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    if self.model_name != 'bert' and self.model_name != 'vgg_lstm' and self.model_name != 'effb0' and self.model_name != 'vgg_only' and self.model_name != 'lstm_only' and self.model_name != 'bert_only':
                        inputs, labels = data
                        labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()
                        if 'lstm' in self.model_name:
                            inputs = torch.unsqueeze(inputs, dim=1)
                        # labels = labels.long()
                        # labels = torch.unsqueeze(labels, dim=1)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)
                    else:
                        images, labels, b_ids, b_ms = data['img'], data['lb'], data['id'], data['mask']
                        images = images.permute(0, 3, 1, 2)
                        labels = F.one_hot(labels.long(), num_classes=self.num_cls).float()

                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        b_ids = b_ids.to(self.device)
                        b_ms = b_ms.to(self.device)

                        if self.model_name == 'bert' or self.model_name == 'effb0' or self.model_name == 'bert_only':
                            b_ids = torch.squeeze(b_ids)
                            b_ms = torch.squeeze(b_ms)
                            outputs = self.model(images, b_ids, b_ms)
                        else:
                            b_ids = torch.unsqueeze(b_ids, dim=1)
                            outputs = self.model(images, b_ids)

                    cm = calculate_confusion_maxtrix(outputs, labels, self.num_cls)
                    confusion_matrix += cm

                    y_pred = F.softmax(outputs, dim=1)
                    y_pred = y_pred.argmax(dim=1).cpu().numpy().tolist()
                    # y_true.append(labels.cpu().numpy().tolist())
                    # y.append(y_pred.tolist())
                    y_true = [*y_true, *labels.argmax(dim=1).cpu().numpy().tolist()]
                    y = [*y, *y_pred]

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
        df_cm = pd.DataFrame(confusion_matrix, index=[i for i in ["0", "1"]],
                             columns=[i for i in ["0", "1"]])
        plt.figure()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        sn.heatmap(df_cm, annot=True, cmap= "Blues", fmt='g')
        plt.savefig(self.model_save_dir + 'confusion_matrix.png')
        print('Testing accuracy %s' % accuracy_score(y_true, y))
        print('Testing F1 score: {}'.format(f1_score(y_true, y, average='weighted')))

    def start(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validation(epoch)
        # self.test_model()

    def save_checkpoint(self, model, saved_path, name):
        torch.save(model.state_dict(), saved_path + name)

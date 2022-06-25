import pytesseract
import cv2
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import numpy as np
import nltk
from bs4 import BeautifulSoup
import re
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

def reshape_token(token, size=128, off_set=True):
    """
    :param token: token input
    :param size: desired size
    :param off_set: padding both side
    :return: token after reshape
    """
    h, w = token.shape
    if h > w:
        scale = size / h
        resize_h = size
        resize_w = int(w * scale)
        off_set_w = (size-resize_w)//2
        off_set_h = 0
    else:
        scale = size / w
        resize_h = int(h * scale)
        resize_w = size
        off_set_h = (size - resize_h) // 2
        off_set_w = 0

    token = cv2.resize(token, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    n_token = np.zeros((size, size))

    if not off_set:
        n_token[0:resize_h, 0:resize_w] = token
    else:
        n_token[off_set_h:resize_h+off_set_h, off_set_w:resize_w+off_set_w] = token

    return n_token


def image2text(image, split='word'):
    """
    :param image: image input
    :return: ocr output
    """
    text = pytesseract.image_to_string(image)
    if split == 'word':
        text = text.split()
    elif split == 'line':
        text = text.split('\n')
        text = [s for s in text if s != '']
    else:
        text = text.replace('-\n', '')
    return text


def tokenization(text, method='gensim', shape=256, reshape=False):
    """
    :param text: text input
    :param method: vectorization method
    :param shape: target shape
    :return: target token
    """
    if method == 'hashing':
        vectorizer = HashingVectorizer(n_features=shape)
        token = vectorizer.transform(text)
        token = token.toarray()
        if reshape:
            token = reshape_token(token, size=shape, off_set=True)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
        token = vectorizer.fit_transform(text).toarray()
        if reshape:
            token = reshape_token(token, size=shape, off_set=True)
    else:
        token = [1]
    return token


def read_img_cv2(img_dir):
    """
    :param img_dir: image path
    :return: image with RGB format
    """
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

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


if __name__ == "__main__":
    data = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    tag_page = tag_page(data)
    print(tag_page)

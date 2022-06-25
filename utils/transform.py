from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
import numpy as np
import torch.nn.functional as tr
import torch

class tfidf(object):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, data):
        inputs, labels = data['X'], data['y']
        token = self.vectorizer.fit_transform([inputs]).toarray()
        data = {'X': token, 'y': labels}
        return data

class resize_data(object):

    def __init__(self, size=1024, off_set=True):
        self.size = size
        self.off_set = off_set

    def __call__(self, data):
        token, labels = data['X'], data['y']
        token = torch.unsqueeze(token, dim=0)
        n_token = tr.interpolate(input=token, size=self.size, mode='nearest')
        n_token = torch.squeeze(n_token)
        data = {'X': n_token, 'y': labels}
        return data

class resizer(object):

    def __init__(self, size=128, off_set=True):
        self.size = size
        self.off_set = off_set

    def __call__(self, data):
        token, labels = data['X'], data['y']
        h, w = token.shape
        if h > w:
            scale = self.size / h
            resize_h = self.size
            resize_w = int(w * scale)
            off_set_w = (self.size - resize_w) // 2
            off_set_h = 0
        else:
            scale = self.size / w
            resize_h = int(h * scale)
            resize_w = self.size
            off_set_h = (self.size - resize_h) // 2
            off_set_w = 0

        token = cv2.resize(token, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        n_token = np.zeros((1, self.size))

        if not self.off_set:
            n_token[0:resize_h, 0:resize_w] = token
        else:
            n_token[off_set_h:resize_h + off_set_h, off_set_w:resize_w + off_set_w] = token

        data = {'X': n_token, 'y': labels}

def rotate(image):
    degree = np.random.choice([90, -90])
    if degree==90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

class vision_aug(object):
    def __init__(self, rota=0.3, crop=0.3):
        self.rotate = rota
        self.crop = crop
    def __call__(self, sample):
        image, idb, mask, lb = sample['img'], sample['id'], sample['mask'], sample['lb']
        if self.rotate < np.random.rand():
            image = rotate(image)
        return {'img': torch.from_numpy(image).to(torch.float32), 'id': idb, 'mask': mask, 'lb': lb}


class Resize_img(object):
    def __init__(self, img_size=256, use_offset=True, mean=48):
        self.img_size = img_size
        self.use_offset = use_offset
        self.mean = np.array(mean)

    def __call__(self, sample):
        image, idb, mask, lb = sample['img'], sample['id'], sample['mask'], sample['lb']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # new_image = np.ones((self.img_size, self.img_size, 3)) * self.mean
        new_image = np.zeros((self.img_size, self.img_size, image.shape[2]))

        if self.use_offset:
            offset_w = (self.img_size - resized_width) // 2
            offset_h = (self.img_size - resized_height) // 2
            new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image
        else:
            new_image[0:resized_height, 0:resized_width] = image

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'id': idb, 'mask': mask, 'lb': lb}

class Normalizer(object):

    def __init__(self, mean=[0.1, 0.1, 0.1], std=[0.2, 0.2, 0.2]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, idb, mask, lb = sample['img'], sample['id'], sample['mask'], sample['lb']
        image = image/255.0
        return {'img': image, 'id': idb, 'mask': mask, 'lb': lb}

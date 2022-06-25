import glob
import tqdm
import json
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np

def extract_images(df):
    files = list(df["names"])
    labels = list(df["one_hot"])
    for f, label in tqdm.tqdm(zip(files, labels)):
        name = f.split('/')[-1][:-4]
        images = convert_from_path(f)
        image_count = 0
        for image, l in zip(images, label):
            filename = "images/{}/{}_p_{}.jpg".format(str(l), name, str(image_count))
            image.save(filename, 'JPEG')
            image_count += 1

def extract_doc_meta(json_file):
    with open(json_file, 'rb') as f:
        meta = json.load(f)
    return meta

def gold_to_onehot(original_indexes):
    all = np.array([])
    for i in original_indexes:
        temp = np.concatenate((np.ones((1)), np.zeros((i-1))))
        all = np.concatenate((all, temp))
    return list(all)

with open('corpus1/TrainTestSet/Trainset/Doclengths_of_the_individual_docs_TRAIN.json', 'rb') as f:
    meta = json.load(f)
names = ['corpus1/TrainTestSet/Trainset/data/' + name + "__concatenated.pdf" for name in meta.keys()]
pages = meta.values()
df_cp1 = pd.DataFrame({"names":names, "pages":pages})
df_cp1["one_hot"] = df_cp1["pages"].apply(gold_to_onehot)

with open('corpus2/TrainTestSet/Trainset/Doclengths_of_the_individual_docs_TRAIN.json', 'rb') as f:
    meta = json.load(f)
names = ['corpus2/TrainTestSet/Trainset/data/' + name + "__concatenated.pdf" for name in meta.keys()]
pages = meta.values()
df_cp2 = pd.DataFrame({"names":names, "pages":pages})
df_cp2["one_hot"] = df_cp2["pages"].apply(gold_to_onehot)

df = df_cp1.append(df_cp2)

df["one_hot"] = df["pages"].apply(gold_to_onehot)
print(len(df_cp1), len(df_cp2), len(df))

extract_images(df)
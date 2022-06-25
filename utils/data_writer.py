import csv
from utils.ocr_utils import image2text, read_img_cv2
import os
from tqdm import tqdm
from multiprocessing import Process
import pandas as pd

def csv_writer(img_dir, save_dir):
    imgs = os.listdir(img_dir)
    csv_file = save_dir + 'data0.1.csv'
    csv_writer = open(csv_file, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_writer, ['file_name', 'labels', 'text'])
    for file in tqdm(imgs, desc='Data writer progress'):
        image = read_img_cv2(img_dir + file)
        text = image2text(image, split='')
        writer.writerow({'file_name': file, 'labels': 0, 'text': text})

def export_doc_from_csv(csv_in, img_dir, csv_out):
    data = pd.read_csv(csv_in)
    data.head()
    texts = []
    for path in tqdm(data.paths):
        file_name = path.split('/')[-1]
        image = read_img_cv2(img_dir + file_name)
        text = image2text(image, split='')
        texts.append(text)
    texts = pd.Series(texts)
    df_0 = pd.DataFrame({"paths": data.paths, "labels": data.labels, "text": texts})
    df_0.to_csv(csv_out, index=False)


if __name__ == '__main__':
    # img_dir = 'D:/WorkSpace/freelancer_job/data/images/0.0.1/'
    # save_dir = 'D:/WorkSpace/freelancer_job/data/images/'
    # p = Process(target=csv_writer(img_dir, save_dir), args=('bob',))
    # p.start()
    # p.join()
    img_dir = 'D:/WorkSpace/freelancer_job/data/images/0.0/'
    csv_in = '../data/0.0_0422.csv'
    csv_out = '../data/0.0_0422_extracted.csv'
    p = Process(target=export_doc_from_csv(csv_in, img_dir, csv_out), args=('bob',))
    p.start()
    p.join()
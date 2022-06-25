import argparse
from doc2vec_gensim_train import doc2vec_Trainer


def get_args():
    parser = argparse.ArgumentParser('Doc2Vec PDF Splitting')
    parser.add_argument('-p', '--project', type=str, default='doc2vec_gensim', help='project name')
    parser.add_argument('-m', '--model', type=str, default='gensim', help='Choosing model')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./data/', help='Dataset name')
    parser.add_argument('-md', '--model_dir', type=str, default='./models/', help='Dataset dir')
    parser.add_argument('-ts', '--test_split', type=float, default=0.3, help='Train:Test ratio')
    parser.add_argument('-vs', '--vector_size', type=int, default=512, help='epochs')
    parser.add_argument('-ep', '--epochs', type=int, default=50, help='epochs')
    parser.add_argument('-ap', '--alpha', type=float, default=0.002, help='alpha')
    parser.add_argument('-db', '--distributed', type=str, default='dm', help='distributed') # change to dw to train the distributed bag of word model
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    trainer = doc2vec_Trainer(opt)
    trainer.start_train()
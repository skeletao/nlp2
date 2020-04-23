import pandas as pd 
from os import path

def gen_w2v_file(train_path, test_path, sentences_path):
    # load data
    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path) 
    sentences = train.iloc[:, -2].tolist() + train.iloc[:, -1].tolist() + test.iloc[:, -1].tolist()
    
    # put all sentence in one file
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for words in sentences:
            s = ' '.join(words) + '\n'
            f.write(s)

def build_w2v(sentences_path, min_count=100):
    pass


if __name__ == "__main__":
    data_path = './projects/01_QA_summarization_inference/data/'
    train_pkl_path = data_path + 'processed/train_cut_clear.pkl'
    test_pkl_path = data_path + 'processed/test_cut_clear.pkl'
    sentences_path = data_path + 'processed/sentences.txt'

    gen_w2v_file(train_pkl_path, test_pkl_path, sentences_path)
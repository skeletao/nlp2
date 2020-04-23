import pandas as pd 
from os import path
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec


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


# Defualt parameters in gensim:
##### sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
##### max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, 
##### hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>,
##### iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, 
##### compute_loss=False, callbacks=(), max_final_vocab=None
def build_w2v(sentences_path, size=256, min_count=5, iter=5):
    sentences = LineSentence(sentences_path)
    # sg: 1 for skip-gram; 0 for cbow
    model = Word2Vec(sentences, size=size, min_count=min_count, iter=iter, sg=1)
    return model


def train_model(model_path, sentences_path):
    model = Word2Vec.load(model_path)
    sentences = []
    with open(sentences_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            sentences.append(line)
    model.train(sentences, total_examples=len(sentences), epochs=model.iter)
    return model


def save_model(model, model_path):
    model.save(model_path)


if __name__ == "__main__":
    data_path = './projects/01_QA_summarization_inference/data/'
    train_pkl_path = data_path + 'processed/train_cut_clear.pkl'
    test_pkl_path = data_path + 'processed/test_cut_clear.pkl'
    sentences_path = data_path + 'processed/sentences.txt'

    gen_w2v_file(train_pkl_path, test_pkl_path, sentences_path)
    model = build_w2v(sentences_path)

    model_path = './projects/01_QA_summarization_inference/models/'
    w2v_model_path = model_path + 'word2vector.model'
    save_model(model, w2v_model_path)

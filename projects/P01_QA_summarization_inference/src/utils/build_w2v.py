import numpy as np 
import pandas as pd 
from os import path
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from tqdm import tqdm
import sys
from pathlib import Path


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

def build_w2v(sentences_path, size=256, min_count=5, iter=5):
    sentences = LineSentence(sentences_path)
    model = Word2Vec(sentences, size=size, min_count=min_count, iter=iter, sg=1)
    return model


def update_w2v(model_path, sentences_path):
    model = Word2Vec.load(model_path)
    sentences = []
    with open(sentences_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            sentences.append(line)
    model.train(sentences, total_examples=len(sentences), epochs=model.iter)
    return model


def build_ft(sentences_path, size=256, min_count=5, iter=5):
    sentences = LineSentence(sentences_path)
    model = FastText(sentences, size=size, min_count=min_count, iter=iter, sg=1)
    return model


def save_model(model, model_path):
    model.save(model_path)


def read_vocab(vocab_path):
    w2i = {}
    i2w = {}
    with open(vocab_path, encoding='utf-8') as f:
        for line in f:
            item = line.strip().split()
            try:
                w2i[item[0]] = int(item[1])
                i2w[int(item[1])] = item[0]
            except:
                print(line)
                continue
    return w2i, i2w


def build_embedding(vocab_path, model_path, model_type='Word2Vector'):
    # load model
    if model_type == 'Word2Vector':
        model = Word2Vec.load(model_path)
    elif model_type == 'FastText':
        model = FastText.load(model_path)

    # generage dict: index to vector; index is based on vocabulay
    w2i, _ = read_vocab(vocab_path)
    vocab_size = len(w2i)  
    vector_size = model.vector_size
    embedding = {}
    count = 0
    for v, i in w2i.items():
        try:
            embedding[i] = model[v]
            count = count + 1
        except:
            embedding[i] = np.random.uniform(-0.25, 0.25, vector_size).astype(np.float32)

    print(f"Found {count}/{vocab_size} words in: {Path(model_path).name}")
    return embedding


def load_tencent_embedding(vocab_path, model_path):
    w2i, _ = read_vocab(vocab_path)
    vocab_size = len(w2i)
    count = 0
    with open(model_path, encoding='utf-8') as f:
        header = f.readline()
        model_vocab_size, model_vector_size = map(int, header.strip().split())
        embedding = {i: np.random.uniform(-0.25, 0.25, model_vector_size) for i in range(vocab_size)}

        for _ in tqdm(range(model_vocab_size)):
            line = f.readline()
            wv = line.split(' ')
            word = wv[0]
            if word in w2i:
                try:
                    index = w2i[wv[0]]
                    embedding[index] = np.asarray(wv[1:], dtype=np.float32)
                    count = count + 1
                except:
                    print(line)
                    continue
            if count == vocab_size:
                break
        print(f"Found {count}/{vocab_size} words in: {Path(model_path).name}")
    return embedding


def save_embedding(embedding, embedding_path):
    with open(embedding_path, 'w', encoding='utf-8') as f:
        for i, vector in embedding.items():
            s = str(i) + ' ' + ' '.join(map(str, vector.tolist()))+'\n'
            f.write(s)


if __name__ == "__main__":
    # import ReadConfig
    config_path = str(Path(__file__).resolve().parent.parent.parent) 
    if config_path not in sys.path:
        sys.path.append(config_path)
    from config.readconfig import ReadConfig

    # get data path
    loc_path = ReadConfig()
    train_path = loc_path.get_path('train')
    test_path = loc_path.get_path('test')
    sentences_path = loc_path.get_path('w2v_sentences')
    w2v_model_path = loc_path.get_path('w2v_model')
    ft_model_path = loc_path.get_path('ft_model')
    tencent_model_path = loc_path.get_path('tencent_model')
    vocab_path = loc_path.get_path('vocab')
    w2v_embedding_path = loc_path.get_path('w2v_embedding')
    ft_embedding_path = loc_path.get_path('ft_embedding')
    tencent_embedding_path = loc_path.get_path('tencent_embedding')


    # prepare sentences for gensim models
    gen_w2v_file(train_path, test_path, sentences_path)

    # train model
    w2v_model = build_w2v(sentences_path)
    ft_model = build_ft(sentences_path)

    # save model
    save_model(w2v_model, w2v_model_path)
    save_model(ft_model, ft_model_path)

    # build embedding matrix
    w2v_embedding = build_embedding(vocab_path, w2v_model_path)
    ft_embedding = build_embedding(vocab_path, ft_model_path, 'FastText')
    tencent_embedding = load_tencent_embedding(vocab_path, tencent_model_path)

    # save embedding matrix
    save_embedding(w2v_embedding, w2v_embedding_path)
    save_embedding(ft_embedding, ft_embedding_path)
    save_embedding(tencent_embedding, tencent_embedding_path)

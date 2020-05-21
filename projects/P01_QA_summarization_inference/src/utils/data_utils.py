import os
import pickle
import time
from projects.P01_QA_summarization_inference.src.utils.batcher import Vocab
import numpy as np


def peek_lines(file_path, n):
    lines = []
    with open(file_path, encoding='utf-8') as f:
        for _ in range(n):
            lines.append(f.readline().strip('\n'))
        return lines


def dump_pkl(data, pkl_path, over_write=True):
    if os.path.exists(pkl_path) and not over_write:
        return

    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'save {pkl_path} ok.')


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)
    return res


def get_result_filename(params, commit=''):
    save_result_dir = params['test_save_dir']
    batch_size = params['batch_size']
    epochs = params['epochs']
    max_length_inp = params['max_dec_len']
    embedding_dim = params['embed_size']
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + f'_batch_size_{batch_size}_epochs_{epochs}_max_length_inp_{max_length_inp}' \
                          f'_embedding_dim_{embedding_dim}_{commit}.csv'

    result_save_path = os.path.join(save_result_dir, filename)
    return result_save_path


def load_word2vec(params):
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    embedding = load_pkl(params['word2vec_output'])
    embedding_dim = len(next(iter(embedding.values())))
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab.word2id), embedding_dim))
    for w, i in vocab.word2id.items():
        if w in embedding:
            embedding_matrix[i] = embedding[w]
    return embedding_matrix




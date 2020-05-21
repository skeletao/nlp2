import pandas as pd
from itertools import chain
from collections import Counter
from projects.P01_QA_summarization_inference.config.readconfig import ReadConfig


def get_all_words(train_pkl_path, test_pkl_path):
    # load data
    train = pd.read_pickle(train_pkl_path)
    test = pd.read_pickle(test_pkl_path)
    # if save in pickle type file, no need this conversion
    # train[train.columns[-2:]] = train[train.columns[-2:]].applymap(literal_eval)
    # test[test.columns[-1]] = test[test.columns[-1]].apply(literal_eval)

    # get all words 
    train['words'] = train[train.columns[-2]] + train[train.columns[-1]]
    test['words'] = test[test.columns[-1]]
    ws1 = train['words'].tolist()
    ws2 = test['words'].tolist()
    words = list(chain.from_iterable(ws1)) + list(chain.from_iterable(ws2))
    
    return words


def build_vocab(words, sort=True, min_count=0, lower=False):
    if lower:
        words = [str.lower(s) for s in words]

    counter = Counter(words)

    if sort:
        word_cnt = counter.most_common()
    else:
        word_cnt = list(counter.items())

    vocab = []
    reverse_vocab = []
    for i, v in enumerate(word_cnt):
        word, cnt = v
        if cnt > min_count:
            vocab.append((word, i))
            reverse_vocab.append((i, word))

    print('Build vocabulary: {len(vocab) words}')
    return vocab, reverse_vocab


def save_vocab(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in vocab:
            s = item[0] + ' ' + str(item[1]) +'\n'
            f.write(s)


if __name__ == "__main__":
    # get data path
    loc_path = ReadConfig()
    train_path = loc_path.get_path('train')
    test_path = loc_path.get_path('test')
    vocab_path = loc_path.get_path('vocab')

    # collect all words
    words = get_all_words(train_path, test_path)

    # build vocabulary
    vocab, _ = build_vocab(words)

    # save vocabulary
    save_vocab(vocab, vocab_path)




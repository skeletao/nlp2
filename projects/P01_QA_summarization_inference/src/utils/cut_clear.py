import pandas as pd 
import jieba
from jieba import posseg
from hanziconv import HanziConv
from functools import reduce
from ast import literal_eval
import re
import sys
from pathlib import Path


def remove_special_char(sentence):
    pattern = r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~——！，。？、￥…（）：【】《》‘’“”\s]+"
    s = re.sub(pattern, '', sentence)
    s = s.strip()
    return HanziConv.toSimplified(s)


def segment(sentence,cut_type='word',pos=False):
    if not isinstance(sentence, str):
        return []

    remove_special_char(sentence)

    if pos:
        if cut_type == 'word':
            return zip(*posseg.cut(sentence))
        else:
            chars = list(sentence)
            poss = [posseg.lcut(c)[0].flag for c in chars]
            return chars, poss
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        else:
            return list(sentence)


def read_stop_words(stop_words_path):
    words = set(['\u3000', '\u81a8', '\u5316', '\u98df', '\u54c1', '\xa0', '\u00a0', '\u2002', '\u2003'])
    with open(stop_words_path, encoding='utf-8') as f:
        for line in f:
            # words.add(line.strip()) dosen't work, it will remove ' ' this speical stop word
            words.add(line.strip('\n'))
    return words


def cut_file(train_path, test_path):
    # load raw data
    # train = pd.read_csv(train_path, encoding='utf-8', nrows=100)
    # test = pd.read_csv(test_path, encoding='utf-8', nrows=100)
    train = pd.read_csv(train_path, encoding='utf-8')
    test = pd.read_csv(test_path, encoding='utf-8')

    # drop 'Report" missed data in train
    train.dropna(axis=0, how='any', subset=[train.columns[-1]], inplace=True)

    # fill Na items with ''
    train.fillna('', inplace=True)
    test.fillna('', inplace=True)

    # cut content from 'Question, result is list, [] for empty and non-str input
    train_cut_columns = [s+'_cut' for s in train.columns[3:]]
    test_cut_columns = [s+'_cut' for s in test.columns[3:]] 
    train[train_cut_columns] = train.loc[:, 'Question':].applymap(segment) 
    test[test_cut_columns] = test.loc[:, 'Question':].applymap(segment)

    return train, test


def clear_file(train, test, stop_words_path):   
    # remove stop words
    stop_words = read_stop_words(stop_words_path)
    train[train.columns[-3:]] = train[train.columns[-3:]].applymap(lambda x: [w for w in x if w not in stop_words])
    test[test.columns[-2:]] = test[test.columns[-2:]].applymap(lambda x: [w for w in x if w not in stop_words])

    # merge Question and Dialogue
    train['QA_cut'] = train[train.columns[-3:-1]].apply(lambda z: reduce(lambda x, y: x+y, z), axis=1)
    test['QA_cut'] = test[test.columns[-2:]].apply(lambda z: reduce(lambda x, y: x+y, z), axis=1)

    # drop empty data rows again: [].astype(bool) = false
    train = train[train['Report_cut'].astype(bool)]
    train = train[train.QA_cut.astype(bool)]
    test = test[test.QA_cut.astype(bool)]

    return train, test


def save_data(train, test, train_path, test_path):
    train.to_pickle(train_path)
    test.to_pickle(test_path)


if __name__ == "__main__":
    # import ReadConfig
    config_path = str(Path(__file__).resolve().parent.parent.parent) 
    if config_path not in sys.path:
        sys.path.append(config_path)
    from config.readconfig import ReadConfig

    # get data path
    loc_path = ReadConfig()
    train_raw_path = loc_path.get_path('train_raw')
    test_raw_path = loc_path.get_path('test_raw')
    stop_words_path = loc_path.get_path('stop_words')
    train_path = loc_path.get_path('train')
    test_path = loc_path.get_path('test')

    # cut 
    train, test = cut_file(train_raw_path, test_raw_path)

    # clear 
    train, test = clear_file(train, test,  stop_words_path)

    # save
    save_data(train, test, train_path, test_path)

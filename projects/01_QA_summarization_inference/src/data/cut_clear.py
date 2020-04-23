import pandas as pd 
import jieba
from jieba import posseg
from hanziconv import HanziConv
from functools import reduce
from os import path
from ast import literal_eval


def segment(sentence,cut_type='word',pos=False):
    if not isinstance(sentence, str):
        return []

    sentence = HanziConv.toSimplified(sentence)

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
    words = set()
    with open(stop_words_path, encoding='utf-8') as f:
        for line in f:
            # words.add(line.strip('\n')) dosen't work, it will remove ' ' this speical stop word
            words.add(line.strip('\n'))
    return words


def cut_file(train_path, test_path):
    # load raw data
    # train = pd.read_csv(train_path, encoding='utf-8', nrows=10)
    # test = pd.read_csv(test_path, encoding='utf-8', nrows=10)
    train = pd.read_csv(train_path, encoding='utf-8')
    test = pd.read_csv(test_path, encoding='utf-8')

    # drop 'Report" missed data in train
    train.dropna(axis=0, how='any', subset=[train.columns[-1]], inplace=True)

    # fill Na items with ''
    train.fillna('', inplace=True)
    test.fillna('', inplace=True)

    # cut word, result is list, [] for Na,
    # and save in interim for backup
    train_cut_path = data_path + 'interim/train_cut.pkl'
    test_cut_path = data_path + 'interim/test_cut.pkl'
    train_cut_columns = [s+'_cut' for s in train.columns[3:]] # 3 means start with 'Question'
    test_cut_columns = [s+'_cut' for s in test.columns[3:]] # 3 means start with 'Question'

    if not path.exists(train_cut_path):
        train[train_cut_columns] = train.loc[:, 'Question':].applymap(segment) # 3 means start with 'Question'
        train.to_pickle(train_cut_path)
    else:
        train = pd.read_pickle(train_cut_path)
        # if save in pickle type file, no need this convertion
        # train[train_cut_columns] = train[train_cut_columns].applymap(literal_eval) 

    if not path.exists(test_cut_path):
        test[test_cut_columns] = test.loc[:, 'Question':].applymap(segment)
        test.to_pickle(test_cut_path)
    else:
        test = pd.read_pickle(test_cut_path)
        # if save in pickle type file, no need this convertion
        # test[test_cut_columns] = test[test_cut_columns].apply(literal_eval)
    return train, test


def clear_file(train, test, stop_words_path):   
    # remove stop words
    stop_words = read_stop_words(stop_words_path)
    train[train.columns[-3:]] = train[train.columns[-3:]].applymap(lambda x: [w for w in x if w not in stop_words])
    test[test.columns[-2:]] = test[test.columns[-2:]].applymap(lambda x: [w for w in x if w not in stop_words])

    # merge Question and Dialogue
    train['QA_cut'] = train[train.columns[-3:-1]].apply(lambda z: reduce(lambda x, y: x+y, z), axis=1)
    test['QA_cut'] = test[test.columns[-2:]].apply(lambda z: reduce(lambda x, y: x+y, z), axis=1)

    # drop 'Report" missed data in train again: [].astype(bool) = false
    train = train[train['Report'].astype(bool)]

    return train, test


def save_data(train, test, train_path, test_path):
    train.to_pickle(train_path)
    test.to_pickle(test_path)


if __name__ == "__main__":
    data_path = './projects/01_QA_summarization_inference/data/'
    train_path = data_path + 'raw/AutoMaster_TrainSet.csv'
    test_path = data_path +'raw/AutoMaster_TestSet.csv'
    stop_words_path = data_path + 'raw/stop_words.txt'

    # cut 
    train, test = cut_file(train_path, test_path)

    # clear 
    train, test = clear_file(train, test,  stop_words_path)

    # save
    train_cut_clear_path = data_path + 'processed/train_cut_clear.pkl'
    test_cut_clear_path = data_path + 'processed/test_cut_clear.pkl'
    save_data(train, test, train_cut_clear_path, test_cut_clear_path)

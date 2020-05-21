import tensorflow as tf
import pandas as pd
from os import path
from projects.P01_QA_summarization_inference.src.modeling.seq_to_seq import SequenceToSequence
from projects.P01_QA_summarization_inference.src.utils.train_helper import train_model
from projects.P01_QA_summarization_inference.src.utils.test_helper import greedy_decode
from projects.P01_QA_summarization_inference.src.utils.data_utils import get_result_filename


def train(params):
    assert params['mode'].lower() == "train", "Please change mode to 'train'"

    print('Creating the model ...')
    model = SequenceToSequence(params)

    print('Creating the checkpoint manager ...')
    ckpt_dir = path.join(params['seq2seq_model_dir'], 'checkpoint')
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    ckpt.restore(ckpt_mgr.latest_checkpoint)
    if ckpt_mgr.latest_checkpoint:
        print(f'Restored from {ckpt_mgr.latest_checkpoint}')
    else:
        print('Initializing from scratch')

    print('Start the training ...')
    train_model(model, params, ckpt_mgr)


def test(params):
    assert params['mode'].lower() == "test", "Please change mode to 'test'"

    print('Building the model ...')
    model = SequenceToSequence(params)

    print('Creating the checkpoint manager ...')
    ckpt_dir = path.join(params['seq2seq_model_dir'], 'checkpoint')
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    print('Restored model')
    ckpt.restore(ckpt_mgr.latest_checkpoint)

    print('Predicting the results ...')
    results = None
    if params['greedy_decode']:
        results = greedy_decode(model, params)

    return results


def save_results(results, params):
    # read file
    test_df = pd.read_csv(params['test_x_dir'], nrows=len(results))

    # write results
    test_df['Prediction'] = results

    # extract 'QID' and 'Prediction'
    test_df = test_df[['QID', 'Prediction']]

    # save in csv
    results_save_path = get_result_filename(params)
    test_df.to_csv(results_save_path, index=None, sep=',', encoding='utf-8')


def test_save(params):
    results = test(params)
    results = list(map(lambda x: x.replace(" ", ""), results))
    save_results(results, params)












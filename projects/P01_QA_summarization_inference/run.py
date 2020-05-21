# coding=utf-8
import sys
from os import path
import tensorflow as tf
from pathlib import Path
import argparse
from projects.P01_QA_summarization_inference.src.utils.train_eval_test import train
from projects.P01_QA_summarization_inference.src.utils.train_eval_test import test_save
from projects.P01_QA_summarization_inference.config.readconfig import ReadConfig

BASE_DIR = Path(__file__).resolve().parent.as_posix()
sys.path.append(BASE_DIR)
NUM_SAMPLES = 82867


def main():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=40, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=100,
                        help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30,
                        help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--vocab_size", default=30000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.00001, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer "
                             "API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)
    parser.add_argument('--cov_loss_wt', default=0.5, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)

    # path
    loc_path = ReadConfig()
    vocab_path = loc_path.get_path('vocab')
    w2v_embedding_path = loc_path.get_path('w2v_embedding')
    ft_embedding_path = loc_path.get_path('ft_embedding')
    tencent_embedding_path = loc_path.get_path('tencent_embedding')
    train_set_x_path = loc_path.get_path('train_set_x')
    train_set_y_path = loc_path.get_path('train_set_y')
    test_set_x_path = loc_path.get_path('test_set_x')
    test_raw_path = loc_path.get_path('test_raw')
    seq2seq_model_dir = loc_path.get_path('seq2seq_model')
    pgn_model_dir = loc_path.get_path('pgn_model')
    test_output_dir = loc_path.get_path('test_output')

    parser.add_argument("--seq2seq_model_dir", default=seq2seq_model_dir, help="Model folder")
    parser.add_argument("--pgn_model_dir", default=pgn_model_dir, help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default=train_set_x_path, help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default=train_set_y_path, help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default=test_set_x_path, help="test_seg_x_dir")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)
    parser.add_argument("--test_x_dir", default=test_raw_path, help="test_x_dir")
    parser.add_argument("--test_save_dir", default=test_output_dir, help="test_save_dir")
    parser.add_argument("--vocab_path", default=vocab_path, help="Vocab path")
    parser.add_argument("--word2vec_output", default=w2v_embedding_path, help="word to embedding path")

    # others
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--epochs", default=20, help="train epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=1300, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_epochs", default=5, help="Save checkpoints every N epochs", type=int)
    parser.add_argument("--loss_print_step", default=100, help="Print batch loss every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=50, help="Number of examples to test", type=int)
    parser.add_argument("--max_num_to_eval", default=2, help="max_num_to_eval", type=int)

    # mode
    parser.add_argument("--mode", default='train', help="training, eval or test options")
    parser.add_argument("--model", default='SequenceToSequence', help="which model to be selected")
    parser.add_argument("--greedy_decode", default=True, help="greedy_decoder")

    args = parser.parse_args()
    params = vars(args)

    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpu:
        tf.config.experimental.set_visible_devices(devices=gpu[0], device_type='GPU')

    if params["mode"] == "train":
        params["steps_per_epoch"] = NUM_SAMPLES // params["batch_size"]
        train(params)
    elif params["mode"] == "test":
        test_save(params)


if __name__ == '__main__':
    main()

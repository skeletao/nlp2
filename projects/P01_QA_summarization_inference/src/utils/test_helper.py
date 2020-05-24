import tensorflow as tf
import math
from tqdm import tqdm
import time
from projects.P01_QA_summarization_inference.src.utils.batcher import Vocab
from projects.P01_QA_summarization_inference.src.utils.batcher import batcher


def greedy_decode(model, params):
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    print(f'True vocab is {vocab}')

    print('Creating the batch set ...')
    dataset = batcher(vocab, params)

    results = []

    batch_size = params['batch_size']
    test_size = params['num_to_test']
    steps = math.ceil(test_size/batch_size)

    for _ in tqdm(range(steps)):
        t0 = time.time()
        enc_data, _ = next(iter(dataset))
        results += batch_greedy_decode(model, enc_data, vocab, params)
        # print(f'Time taken for 1 step {time.time()-t0} sec\n')
    return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # get actual batch data size
    batch_data = enc_data['enc_input']
    batch_size = batch_data.shape[0]

    # encode first
    inputs = tf.convert_to_tensor(batch_data)
    enc_output, enc_hidden = model.call_encoder(inputs)

    # prepare decoder
    dec_hidden = enc_hidden
    dec_input = tf.constant([vocab.start_token_index]*batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)

    # then decode
    predicts = [''] * batch_size
    for t in range(params['max_dec_len']):
        pred, dec_hidden, _ = model.decoder(dec_input, dec_hidden, enc_output)

        # [batch_size, vocab_size] --> [batch_size,]
        predicted_ids = tf.argmax(pred, axis=1).numpy()

        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '

        dec_input = tf.expand_dims(predicted_ids, 1)

    results = []
    stop_token = vocab.id_to_word(vocab.stop_token_index)
    stop_token_len = len(stop_token)
    for predict in predicts:
        predict = predict.strip()
        if stop_token in predict and predict[0:stop_token_len] != stop_token:
            predict = predict[:predict.index(stop_token)]
        results.append(predict)

    return results






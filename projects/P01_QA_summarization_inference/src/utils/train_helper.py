import tensorflow as tf
import time
from projects.P01_QA_summarization_inference.src.utils.batcher import Vocab
from projects.P01_QA_summarization_inference.src.utils.batcher import batcher


def train_model(model, params, ckpt_mgr):
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    print(f'True vocab is {vocab}')

    print('Creating the batch set ...')
    dataset = batcher(vocab, params)

    optimizer = tf.keras.optimizers.Adam(params['learning_rate'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    pad_token_index = vocab.pad_token_index

    def loss_function(real, pred):
        # [batch_size, dec_seq_len]
        mask = tf.math.not_equal(real, pad_token_index)
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        # [batch_size, dec_seq_len]
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # [batch_size,]
        loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
        # []
        loss_ = tf.reduce_mean(loss_)
        return loss_

    # @tf.function()
    def train_step(enc_input, dec_tar, dec_input):
        with tf.GradientTape() as tape:
            enc_output, hidden = model.call_encoder(enc_input)
            pred, _ = model(enc_output, dec_input, hidden, dec_tar)
            loss = loss_function(dec_tar, pred)

        variables = model.encoder.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(params['epochs']):
        t0 = time.time()
        best_loss = 10000
        step = 0
        total_loss = 0
        for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
            batch_loss = train_step(batch[0]["enc_input"],
                                    batch[1]["dec_target"],
                                    batch[1]["dec_input"])
            step += 1
            total_loss += batch_loss
            # print loss every 100 steps
            if step % params['loss_print_step'] == 0:
                print(f'Epoch {epoch+1} Batch {step} Loss {batch_loss.numpy():.4f}')

        # saving (checkpoint) the model every 2 epochs
        if epoch % params['checkpoints_save_epochs'] == 0:
            epoch_loss = total_loss/step
            if epoch_loss < best_loss:
                best_loss = epoch_loss
            ckpt_save_path = ckpt_mgr.save()
            print(f'Saving check point for epoch {epoch+1} at {ckpt_save_path}, best loss {best_loss:.4f}')
            print(f'Epoch {epoch+1} Loss {epoch_loss:.4f}')
        # lr = params['learning_rate'] * tf.math.pow(0.9, epoch+1)
        # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
        print(f'Time taken for 1 epoch {time.time()-t0} sec\n')


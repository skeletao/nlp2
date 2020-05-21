import tensorflow as tf
from projects.P01_QA_summarization_inference.src.modules import rnn_decoder
from projects.P01_QA_summarization_inference.src.modules import rnn_encoder
from projects.P01_QA_summarization_inference.src.utils.data_utils import load_word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()

        self.embedding_matrix = load_word2vec(params)
        self.params = params

        self.encoder = rnn_encoder.Encoder(params['vocab_size'],
                                           params['embed_size'],
                                           params['enc_units'],
                                           params['batch_size'],
                                           self.embedding_matrix)

        self.decoder = rnn_decoder.Decoder(params['vocab_size'],
                                           params['embed_size'],
                                           params['dec_units'],
                                           params['batch_size'],
                                           params['attn_units'],
                                           self.embedding_matrix)

    def call_encoder(self, enc_input):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_input, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_input, dec_hidden, dec_tar):
        predictions = []
        attentions = []
        for t in range(dec_tar.shape[1]):
            pred, dec_hidden, attn_dist = self.decoder(tf.expand_dims(dec_input[:, t], 1),
                                                       dec_hidden,
                                                       enc_output)
            predictions.append(pred)
            attentions.append(attn_dist)
        # [batch_size, vocab_size] * dec_seq_len --> [batch_size, dec_seq_len, vocab_size]
        # [batch_size, dec_units]
        return tf.stack(predictions, 1), dec_hidden







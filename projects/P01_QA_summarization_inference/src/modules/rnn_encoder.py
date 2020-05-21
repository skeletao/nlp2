import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, embedding_matrix=None, use_bi_gru=True):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       weights=[embedding_matrix],
                                                       trainable=False)
        self.use_bi_gru = use_bi_gru
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, enc_input, hidden):
        # [batch_size, sequence_length] --> [batch_size, sequence_length, embedding_dim]
        enc_input = self.embedding(enc_input)

        # another way to initialize hidden state
        # if self.use_bi_gru:
        #     hidden = self.gru.get_initial_state(enc_input)
        # else:
        #     hidden = self.gru.get_initial_state(enc_input)*2

        if self.use_bi_gru:
            output, forward_state, backward_state = self.bi_gru(enc_input, initial_state=hidden)
            state = tf.concat([forward_state, backward_state], axis=-1)
        else:
            output, state = self.gru(enc_input, initial_state=hidden)

        # [batch_size, sequence_length, enc_units]
        # [batch_size, enc_units]
        return output, state

    def initialize_hidden_state(self):
        if self.use_bi_gru:
            return tf.split(tf.zeros((self.batch_size, self.enc_units*2)), num_or_size_splits=2, axis=1)
        else:
            return tf.zeros((self.batch_size, self.enc_units))

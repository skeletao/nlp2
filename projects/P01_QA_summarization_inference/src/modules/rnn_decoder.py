import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # [batch_size, hidden_size] --> [batch_size, 1, hidden_size]
        query_with_time_axis = tf.expand_dims(query, axis=1)

        # calculate attention weights: v'tanh(w1ht+w2hs)
        # [batch_size, 1, hidden_size] --> [batch_size, 1, units]
        w1 = self.W1(query_with_time_axis)
        # [batch_size, sequence_length, hidden_size] --> [c]
        w2 = self.W2(values)
        # [batch_size, sequence_length, units] --> [batch_size, sequence_length, 1]
        score = self.V(tf.nn.tanh(w1+w2))
        attention_weights = tf.nn.softmax(score, axis=1)

        # calculate context vector
        # [batch_size, sequence_length, 1] .* [batch_size, sequence_length, enc_units]
        context_vector = attention_weights * values
        # [batch_size, sequence_length, enc_units] --> [batch_size, enc_units]
        context_vector = tf.reduce_mean(context_vector, axis=1)

        # [batch_size, enc_units]
        # [batch_size, sequence_length, 1] --> [batch_size, sequence_length]
        return context_vector, tf.squeeze(attention_weights, axis=-1)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, dec_units, batch_size, attn_units, embedding_matrix=None):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.batch_size = batch_size

        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_size,
                                                       weights=[embedding_matrix],
                                                       trainable=False)
        self.gru = tf.keras.layers.GRU(dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(attn_units)

    def call(self, dec_input, hidden, enc_output):
        # [batch_size, 1] --> [batch_size, 1, embedding_dim]
        dec_input = self.embedding(dec_input)

        # [batch_size, enc_units]
        context_vector, attn_dist = self.attention(hidden, enc_output)

        # [batch_size, 1, embedding_dim] + [batch_size, 1, enc_units] --> [batch_size, 1, embedding_dim+enc_units]
        dec_input = tf.concat([tf.expand_dims(context_vector, axis=1), dec_input], axis=-1)

        # state shape: [batch_size, dec_units]
        output, state = self.gru(dec_input)
        # [batch_size, 1, dec_units] --> [batch_size*1, dec_units]
        output = tf.reshape(output, (-1, output.shape[2]))

        # [batch_size*1, dec_units] --> [batch_size, vocab_size]
        out = self.fc(output)
        # [batch_size, vocab_size]
        # [batch_size, dec_units]
        # [batch_size, sequence_length]
        return out, state, attn_dist










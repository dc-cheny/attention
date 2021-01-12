import numpy as np
import tensorflow as tf

print(tf.__version__)


def positional_embedding(pos, model_size):
    """
    位置编码
    """
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE


# max_length = max(len(data_en[0]), len(data_fr_in[0]))
max_length = 128
MODEL_SIZE = 512

pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

            # Here we scale the score as described in the paper
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len)

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, query_len, model_size)
        return heads


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h

        # One Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)

        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        # num_layers FFN and Normalzation layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]

        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, sequence):
        sub_in = []
        for i in range(sequence.shape[1]):
            # compute the embedded vector
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))

            # add positional encoding to the embedded vector
            sub_in.append(embed + pes[i, :])
        sub_in = tf.concat(sub_in, axis=1)

        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):
            sub_out = []
            # Iterate along the sequence length
            for j in range(sub_in.shape[1]):
                # Compute the context vector towards the whole sequence
                attention = self.attention[i](tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)
                sub_out.append(attention)

            # Concatenate the result to have shape (batch_size, length, model_size)
            sub_out = tf.concat(sub_out, axis=1)
            # Residual connection
            sub_out = sub_in + sub_out
            # Normalize the output
            sub_out = self.attention_norm[i](sub_out)
            # The FFN input is the output of the Multi-Head Attention
            ffn_in = sub_out
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the resudual connection
            ffn_out = ffn_in + ffn_out
            # Normalize the output
            ffn_out = self.ffn_norm[i](ffn_out)

            # Assign the FFN output to the next layer's Multi-Head Attention input
            sub_in = ffn_out

        return ffn_out


encoder = Encoder(50000, 512, 6, 4)
encoder_output = encoder(np.array([[3] * 128] * 4))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_out):
        embed_out = []
        for i in range(sequence.shape[1]):
            embeded = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embeded + pes[i, :])
        embed_out = tf.concat(embed_out, axis=1)

        bot_sub_in = embed_out
        for i in range(self.num_layers):
            bot_sub_out = []

            for j in range(bot_sub_in.shape[1]):
                values = bot_sub_in[:, :j, :]
                attention = self.attention_bot[i](tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)
                bot_sub_out.append(attention)

            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            # residual layer
            bot_sub_out = bot_sub_out + bot_sub_in
            # normalization
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            mid_sub_in = bot_sub_out
            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.attention_mid[i](tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_out)
                mid_sub_out.append(attention)
            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out += mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # ffn
            ffn_in = mid_sub_out
            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out += ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)
        return logits


decoder = Decoder(50000, 512, 6, 4)
print(decoder(np.array([[3] * 128] * 4), encoder_output).shape)

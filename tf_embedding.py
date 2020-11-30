import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_datasets as tfds

print(tf.__version__)


embedding_layer = layers.Embedding(1000, 10)
print(embedding_layer)
result = embedding_layer(tf.constant([[1, 2, 3], [2, 56, 10]]))
print(result.numpy())


### multi LSTM

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

print(model.summary())

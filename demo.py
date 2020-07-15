from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
# from attention_keras import *

max_features = 20000
maxlen = 80
batch_size = 32

# data preprocessing
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
# print(x_train[0], y_train[0])

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)\

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
# print(S_inputs.shape, embeddings.shape)
# O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])




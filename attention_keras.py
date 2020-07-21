#! -*- coding: utf-8 -*-

from keras.layers import *
import keras.backend as K


def to_mask(x, mask, mode='mul'):
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


def extract_seq_patches(x, kernel_size, rate):
    pass


class OurLayer(Layer):
    """
     增加reuse方法，允许在定义layer时调用现成的层，这样可以重复使用。不然call过了以后不能复用
     """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs


class Attention(OurLayer):
    """
    多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None, mask_right=False, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super().build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        # mask.shape=[batch_size, seq_len] or [batch_size, seq_len, 1]
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)

        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # 维度置换
        qw = K.permute_dimensions(qw, [0, 2, 1, 3])
        kw = K.permute_dimensions(kw, [0, 2, 1, 3])
        vw = K.permute_dimensions(vw, [0, 2, 1, 3])

        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, [0, 3, 2, 1])
        a = to_mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                pass


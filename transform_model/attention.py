import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class MultiHeadAttention(Layer):
    def __init__(self, units, num_heads, **kwargs):
        """
        每个注意力depth = units // num_heads
        :param units: units = num_heads * depth
        :param num_heads: num_heads个自注意力
        :param kwargs:
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        assert units % num_heads == 0, 'units // num_heads == 0'
        self.depth = self.units // self.num_heads
        self.WQ = Dense(units=units)
        self.WK = Dense(units=units)
        self.WV = Dense(units=units)
        self.dense = Dense(self.units)

    def _split_heads(self, x, batch_size):
        """

        :param x: shape (batch_size, seq_len, units)
        :param batch_size:
        :return:
        """
        # shape (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # shape (batch_size, num_heads, seq_len, depth)
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x

    def _scaled_dot_product_attention(self, q, k, v, mask):
        """
        Args:
        - q: shape == (..., seq_len_q, depth)
        - k: shape == (..., seq_len_k, depth)
        - v: shape == (..., seq_len_v, depth_v)
        - seq_len_k == seq_len_v
        - mask: shape == (..., seq_len_q, seq_len_k)
        Returns:
        - output: weighted sum
        - attention_weights: weights of attention
        """

        # matmul_qk.shape: (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        # scale 重新归一化权重的分布
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # 使得在softmax后值趋近于0
            scaled_attention_logits += (mask * -1e9)

        # attention_weights.shape: (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)

        # output.shape: (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v, mask):
        """
        一般来说 seq_len_q == seq_len_k == seq_len_v
        :param q: (batch_size, seq_len_q, units)
        :param k: (batch_size, seq_len_k, units)
        :param v: (batch_size, seq_len_v, units)
        :param mask:
        :return:
        """
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)
        k = self.WK(q)
        v = self.WV(q)
        # shape (batch_size, num_heads, seq_len, depth)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # scaled_attention_outputs shape: (batch_size, num_heads, seq_len_q, depth_v)
        scaled_attention_outputs, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)
        # shape (batch_size, num_heads, seq_len_q, depth_v) -> (batch_size, seq_len_q, num_heads, depth_v)
        scaled_attention_outputs = tf.transpose(scaled_attention_outputs, perm=[0, 2, 1, 3])
        # concat_attention.shape: (batch_size, seq_len_q, units)
        concat_attention = tf.reshape(scaled_attention_outputs, (batch_size, -1, self.units))
        # output.shape : (batch_size, seq_len_q, units)
        output = self.dense(concat_attention)

        return output, attention_weights

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config().copy()
        config.update({
            'units': self.units,
            'num_heads': self.units
        })
        return config

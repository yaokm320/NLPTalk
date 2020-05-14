import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Embedding
from transform_model.attention import MultiHeadAttention
from transform_model.feed_forward import FeedForwardLayer
from transform_model.transform import get_position_embedding


class EncoderLayer(Layer):

    def __init__(self, units, num_heads, dff, rate=.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(units=units, num_heads=num_heads, trainable=True)
        self.feed_forward = FeedForwardLayer(units, dff, trainable=True)

        self.attention_normal = LayerNormalization(epsilon=1e-6)
        self.attention_n_dropout = Dropout(rate=rate)
        self.feed_forward_normal = LayerNormalization(epsilon=1e-6)
        self.feed_forward_n_dropout = Dropout(rate=rate)

    def call(self, inputs, training, encoder_padding_mask):
        attention_output, _ = self.attention(inputs, inputs, inputs, encoder_padding_mask)
        attention_output = self.attention_n_dropout(attention_output, training=training)
        output = self.attention_normal(inputs + attention_output)
        ff_n_output = self.feed_forward(output)
        ff_n_output = self.feed_forward_n_dropout(ff_n_output, training=training)
        output = self.feed_forward_normal(output + ff_n_output)
        return output

    def get_config(self):
        config = super(EncoderLayer, self).get_config().copy()
        return config


# model layer
class EncoderModel(Layer):
    def __init__(self, num_layers, vocab_size, max_length, units, num_heads, dff, rate=.1, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.max_length = max_length
        self.num_layers = num_layers
        self.units = units
        # 先进行词嵌入
        self.embedding = Embedding(vocab_size, units)
        # 得到词位置信息
        self.position_embedding = get_position_embedding(max_length, units)
        self.dropout = Dropout(rate)
        self.encoder_layers = [EncoderLayer(units, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, inputs, encoder_mask, training):
        """

        :param inputs: shape (batch_size, seq_len)
        :param train:
        :param encoder_mask:
        :return:
        """
        seq_len = tf.shape(inputs)[1]
        tf.debugging.assert_less_equal(seq_len, self.max_length, 'seq_len should be less or equal max_length')
        # shape (batch_size, seq_len, units)
        inputs = self.embedding(inputs)
        # 将inputs 数值放大  和 position位置信息想加 使得占的比重高
        inputs *= tf.math.sqrt(tf.cast(self.units, tf.float32))
        inputs += self.position_embedding[:, :seq_len, :]
        inputs = self.dropout(inputs, training=training)
        for i in range(self.num_layers):
            inputs = self.encoder_layers[i](inputs, training, encoder_mask)
        return inputs

    def get_config(self):
        config = super(EncoderModel, self).get_config().copy()
        config.update({
            'max_length', self.max_length,
            'num_layers', self.num_layers,
            'units', self.units
        })
        return config

# sample_encoder_model = EncoderModel(2, 8500, 40, 512, 8, 2048)
# sample_encoder_model_input = tf.random.uniform((64, 37))
# sample_encoder_model_output = sample_encoder_model(
#     sample_encoder_model_input, False, encoder_mask=None)
# print(sample_encoder_model_output.shape)

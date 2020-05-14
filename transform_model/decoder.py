import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Embedding
from transform_model.attention import MultiHeadAttention
from transform_model.feed_forward import FeedForwardLayer
from transform_model.transform import get_position_embedding


class DecoderLayer(Layer):
    def __init__(self, units, num_heads, dff, rate=.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.attention1 = MultiHeadAttention(units=units, num_heads=num_heads, trainable=True)
        self.attention2 = MultiHeadAttention(units=units, num_heads=num_heads, trainable=True)
        self.feed_forward = FeedForwardLayer(units, dff, trainable=True)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, decoder_mask, encoder_outputs, en_decoder_padding_mask, training):
        """
         inputs -> self attention -> add & normalize & dropout -> out1
        out1 , encoding_outputs -> attention -> add & normalize & dropout -> out2
        out2 -> ffn -> add & normalize & dropout -> out3
        :param inputs: decoder Input
        :param decoder_mask: decoder mask 只能看到前面的词
        :param encoder_outputs: encoder 输出
        :param training:
        :param en_decoder_padding_mask:
        :return:
        """
        attention1, attention1_weights = self.attention1(inputs, inputs, inputs, decoder_mask)
        attention1 = self.dropout1(attention1, training=training)
        output1 = self.layer_norm1(attention1 + inputs)

        attention2, attention2_weights = self.attention2(output1, encoder_outputs, encoder_outputs,
                                                         en_decoder_padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        output2 = self.layer_norm2(attention2 + output1)

        output = self.feed_forward(output2)
        ffn_output = self.dropout3(output, training=training)
        out3 = self.layer_norm3(ffn_output + output2)
        return out3, attention1_weights, attention2_weights

    def get_config(self):
        config = super(DecoderLayer, self).get_config().copy()
        return config


class DecoderModel(Layer):
    def __init__(self, num_layers, vocab_size, max_length, units, num_heads, dff, rate=.1, **kwargs):
        super(DecoderModel, self).__init__(**kwargs)
        self.max_length = max_length
        self.num_layers = num_layers
        self.units = units

        self.embedding = Embedding(vocab_size, units)
        self.position_embedding = get_position_embedding(max_length, units)

        self.dropout = Dropout(rate)
        self.decoder_layers = [DecoderLayer(units, num_heads, dff, rate) for _ in range(self.num_layers)]

    # inputs, decoder_mask, encoder_outputs, en_decoder_padding_mask, training
    def call(self, inputs, decoder_mask, encoder_outputs, en_decoder_padding_mask, training):
        # inputs shape (batch_size, seq_len)
        seq_len = tf.shape(inputs)[1]
        tf.debugging.assert_less_equal(seq_len, self.max_length, 'seq_len should be less or equal max_length')

        inputs = self.embedding(inputs)
        inputs *= tf.math.sqrt(tf.cast(self.units, tf.float32))
        inputs += self.position_embedding[:, :seq_len, :]
        inputs = self.dropout(inputs, training=training)
        for i in range(self.num_layers):
            inputs, _, _ = self.decoder_layers[i](inputs, decoder_mask, encoder_outputs, en_decoder_padding_mask,
                                                  training)
        return inputs

    def get_config(self):
        config = super(DecoderModel, self).get_config().copy()
        config.update({
            'max_length', self.max_length,
            'num_layers', self.num_layers,
            'units', self.units
        })
        return config

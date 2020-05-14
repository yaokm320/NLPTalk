import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class BahdanauAttention(Layer):
    '''
    实现注意力层
    '''
    def __init__(self, units, name='attention_layer', **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)

    def call(self, inputs):
        # query_shape == (batch_size, decoder_units)
        # values_shape = (batch_size, seq_length, encoder_units)
        query, values = inputs[0], inputs[1]
        # hidden_with_time_axis shape == (batch_size, 1, decoder_units)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, encoder_seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, encoder_seq_len, decoder_units)
        score = self.V(tf.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, encoder_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # shape = (batch_size, seq_len, decoder_units)
        # tensorflow中的*是点乘
        context_vector = attention_weights * values
        # context_vector shape after sum == (batch_size, encoder_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
        
    # tensorflow2.0 自定义层需要实现get_config 否则会报
    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    # 将它们转为字典键值并且返回使用
    def get_config(self):
        # config = {"units": self.units}
        base_config = super(BahdanauAttention, self).get_config()
        return dict(list(base_config.items()))
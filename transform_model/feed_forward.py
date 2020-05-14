from tensorflow.keras.layers import Dense, Layer


# 前向神经网络
class FeedForwardLayer(Layer):
    def __init__(self, units, dff, **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.units = units
        self.dff = dff
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(units)

    def call(self, inputs):
        output = self.dense1(inputs)
        output = self.dense2(output)
        return output

    def get_config(self):
        config = super(FeedForwardLayer, self).get_config().copy()
        config.update({
            'units': self.units,
            'dff': self.dff
        })
        return config

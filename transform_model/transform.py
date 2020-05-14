import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from transform_model.encoder import EncoderModel
from transform_model.decoder import DecoderModel


# batch_data.shape: [batch_size, seq_len]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


# attention_weights.shape: [3,3]
# [[1, 0, 0],
#  [4, 5, 0],
#  [7, 8, 9]]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


import numpy as np


# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# pos.shape: [sentence_length, 1]
# i.shape  : [1, d_model]
# result.shape: [sentence_length, d_model]
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_position_embedding(sentence_length, units):
    """

    :param sentence_length: 句子长度
    :param d_model: 词向量的长度
    :return:
    """
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(units)[np.newaxis, :], units)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # position_embedding.shape: [sentence_length, units)
    position_embedding = angle_rads
    # position_embedding.shape: [1, sentence_length, units]
    position_embedding = position_embedding[np.newaxis, ...]
    return tf.cast(position_embedding, dtype=tf.float32)


def get_transform_model(encoder_input, target_input, encoder_mask, decoder_mask, en_decoder_padding_mask,
                        num_layers, encoder_vocab_size, decoder_vocab_size, max_length, units, num_heads,
                        dff, training, rate=0.1):
    # (batch_size, encoder_seq_len, units)
    output = EncoderModel(num_layers, encoder_vocab_size, max_length, units,
                          num_heads, dff, rate)(encoder_input, encoder_mask, training)
    # (batch_size, decoder_seq_len, units)
    output = DecoderModel(num_layers, decoder_vocab_size, max_length, units,
                          num_heads, dff, rate)(target_input, decoder_mask, output, en_decoder_padding_mask, training)
    # shape (batch_size, decoder_seq_len, decoder_vocab_size)
    output = Dense(decoder_vocab_size)(output)
    model = Model([encoder_input, target_input, encoder_mask, decoder_mask, en_decoder_padding_mask], output)
    return model

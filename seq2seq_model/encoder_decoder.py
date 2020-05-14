import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Embedding, Bidirectional, Dense, Concatenate, Reshape
from seq2seq_model.bahdanau_attention import BahdanauAttention


def encoder_model(encoder_input, vocab_size, embedding_dim, encoder_units):
    """
    编码器模型
    encoder_input: 编码器的输入层，shape=(batch_size, encoder_seq_num),这里的encoder_seq_num就是每一个句子中单词的个数
    vocab_size: 字典中单词的数量
    embedding_dim: 每个单词embedding的维度
    encoder_units: 设置GRU单元的神经元的个数
    """
    # 嵌入层， shape=(batch_size, encoder_seq_num, embedding_dim)
    output = Embedding(vocab_size, embedding_dim)(encoder_input)
    # encoder_output_shape = (batch_size, encoder_seq_num, encoder_units)
    # encoder_state_shape = (batch_size, encoder_units)
    # GRU的每个单元都会生成一个（batch_size, encoder_units）的tensor，一共有seq_num个这样的GRU单元
    encoder_output, encoder_state = GRU(encoder_units, return_sequences=True, return_state=True, 
                                        recurrent_initializer='glorot_uniform')(output)
    model = Model(inputs = encoder_input, outputs = [encoder_output, encoder_state])
    return model


def decoder_model(decoder_input, hidden, encoder_outputs, vocab_size, embedding_dim, units):
    """
    值得注意的是：当前定义的解码器模型并不是一个完整的模型，他只是解码器中的一部分，
    因为解码器每一次都需要用到上一次的输出，所以需要将当前的解码器模型多次迭代。

    每一次都需要知道上一层的隐状态，这样才可以计算出context_vector
    第一层的隐状态可以直接使用编码器的最后输出的隐状态来代替。

    :param decoder_input: shape = (batch_size, 1)，解码器的输入，其实就是每一次输入一个单词
    :param hidden: (batch_size, decoder_units)，解码器的隐状态使用的是编码器的最后输入的状态
    :param encoder_outputs: (batch_size, encoder_seq_len, encoder_units)
    :param vocab_size: 字典中单词的数量
    :param embedding_dim: 单词的embedding的维度
    :param decoder_units: GRU单元中神经元的个数
    :return:
    """
    # 1. 注意力层
    # 注意力层的输入有两个，分别是编码器的output以及解码器的隐state, 计算得到注意力权重
    # 注意力层的输出有两个，分别是context_vector和注意力权重
    # context_vector_shape = (batch_size, encoder_units), 
    # attention_weights_shape = (batch_size, encoder_seq_len, 1),得到原句子中每个单词的权重数值

    # 对于第一个词来说，他没有上一个词的隐状态，因此这里直接使用编码器的最后的隐状态来表示。
    context_vector, attention_weights = BahdanauAttention(units, trainable=True)([hidden, encoder_outputs])

    # 2. 嵌入层
    # shape = (batch_size, 1, embedding_dim)  解码器每次输入只有一个词, 因为它是从生成的词预测下一个词, 一直重复
    output = Embedding(vocab_size, embedding_dim)(decoder_input)

    # 3. 拼接
    # 有两种实现方式
    # 1 将context_vector和输入进行拼接 送入net
    # 2 将context_vector和gru输出进行拼接 送入net
    # 这里是实现的第1个
    # context_vector_shape = (batch_size, 1, encoder_units)
    # output = (batch_size, 1, embedding_dim)
    # 将上面的两个进行拼接
    # shape = (batch_size, 1, embedding_dim + encoder_units)
    output = Concatenate(axis=-1)([tf.expand_dims(context_vector, 1), output])

    # 4. 通过GRU
    # output_shape = (batch_size, 1, decoder_units), 
    # state_shape = (batch_size, decoder_units)
    output, state = GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')(output)

    # 5. shape变化
    # output_shape = (batch_size * 1, decoder_units)
    output = Reshape(target_shape=(-1, units))(output)

    # 6. 全联接
    # shape (batch_size, vocab_size)
    output = Dense(vocab_size)(output)

    model = Model(inputs=[decoder_input, hidden, encoder_outputs], outputs=[output, state])
    return model


def inference(process, encoder, decoder, sentence):
    '''
    推理函数，给定编码器，解码器模型和问句，根据模型的参数得到答句
    encoder是编码器模型
    decoder是解码器模型
    sentence是问句
    '''
    inputs = process.val_process(sentence)
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    '''
    hidden = [tf.zeros((1, 256))]
    encoding_out, encoding_hidden = encoder(inputs, hidden)
    '''

    encoding_out, encoding_hidden = encoder(inputs)
    decoding_hidden = encoding_hidden
    # decoder的第一个输入是开始符
    decoding_input = tf.expand_dims([process.a_tokenizer.word_index['<eos>']], 0)
    for i in range(1000):
        predictions, decoding_hidden = decoder([decoding_input, decoding_hidden, encoding_out])
        predictions = tf.squeeze(predictions)
        predicted_id = tf.argmax(predictions).numpy()
        # print(predictions, len(predictions))
        # 碰到结束符 break
        if process.a_tokenizer.index_word[predicted_id] == '<sos>' or i > 20:
            return result, sentence
        result += process.a_tokenizer.index_word[predicted_id] + ' '
        # 将上一次的预测的输出单词，作为下一次的输入
        decoding_input = tf.expand_dims([predicted_id], 0)
    return result, sentence


# from tensorflow.keras.layers import Input
# # 测试编码器
# inputs = Input((23,))
# encoder = encoder_model(inputs, 1000, 56, 256)
# encoder.summary()
# print(encoder.output[0].shape, encoder.output[1].shape)

# # 测试解码器
# inputs2 = Input((1,))
# input3 = Input((256, ))
# input4 = Input((23, 256))
# decoder = decoder_model(inputs2, input3, input4, 1000, 56, 256)
# decoder.summary()

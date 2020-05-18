import tensorflow as tf
from util.pre_process import PreProcess
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import time
import os
from seq2seq_model.encoder_decoder import encoder_model, decoder_model, inference
from tensorflow.keras.layers import Input
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class NlpModel(object):

    def __init__(self, process, embedding_dim, units, batch_size, encoder_weights_path, decoder_weights_path):

        # process.q_lenght是每一个句子中单词的个数
        # shape = (batch_size, process.q_lenght)
        self.process = process
        self.embedding_dim = embedding_dim
        self.units = units
        self.encoder_weights_path = encoder_weights_path
        self.decoder_weights_path = decoder_weights_path
        self.batch_size = batch_size
        # self.process.length是样本的数量，即有多少段对话
        self.steps_per_epoch = self.process.length // self.batch_size
        self.encoder_input = Input((self.process.q_lenght,))
        # 编码器模型
        # q_vocab_size是预料中不重复单词的数量
        self.encoder = encoder_model(self.encoder_input, self.process.q_vocab_size, self.embedding_dim, self.units)

        self.decoder_input= Input((1,))
        self.hidden_input = Input((self.units,))
        self.encoder_output_input = Input((self.process.q_lenght, self.units))
        # 解码器模型
        # 一个完整的解码器是由下面的decoder多个叠加在一起得到的，下面的decoder相当于一个完整的解码器的一层
        self.decoder = decoder_model(self.decoder_input, self.hidden_input, self.encoder_output_input,
                                     self.process.a_vocab_size, self.embedding_dim, self.units)

        if os.path.exists(self.encoder_weights_path):
            self.encoder.load_weights(self.encoder_weights_path)

        if os.path.exists(self.decoder_weights_path):
            self.decoder.load_weights(self.decoder_weights_path)

        # （batch_size, seq_length）的问句和答句
        # 是一个生成器，每next一次，会产生batch_size个元素出来供训练
        self.data_generate = self.process.generate_data(self.batch_size)
        # 学习率衰减来做
        self.optimizer = Adam(lr=1e-3)

        self.loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(self, real, pred):
        """
        损失函数
        :param real:
        :param pred:
        :return:
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    # 转成图方法 加快运算
    # @tf.function
    def train_step(self, inp, targ):
        loss = 0
        with tf.GradientTape() as tape:
            encoding_output, encoding_hidden = self.encoder(inp)
            # 解码器的第一个隐状态 = 编码器最后的隐状态
            decoding_hidden = encoding_hidden
            # 解码器的第一个输入是一个固定的标记<eos>
            # shape (batch_size, 1)
            decoding_input = tf.expand_dims([self.process.a_tokenizer.word_index['<eos>']] * self.batch_size, 1)

            # feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, decoding_hidden = self.decoder([decoding_input, decoding_hidden, encoding_output])
                loss += self.loss_function(targ[:, t], predictions)
                decoding_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[0]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train(self, epochs, model_weights_path):
        """
        训练
        :param epochs:
        :param model_weights_path:
        :return:
        """
        # 计算有多少个epoch
        epoch_batch_num = self.process.length // self.batch_size

        # model weights path
        # model_weights_path = './models'
        if not os.path.exists(model_weights_path):
            os.mkdir(model_weights_path)

        for epoch in range(epochs):
            total_loss = 0
            start = time.time()
            for step in range(epoch_batch_num):
                # 以batch_size为一批进行训练
                inp, targ = next(self.data_generate)
                batch_loss = self.train_step(inp, targ)
                total_loss += batch_loss
                # if step % 100 == 0:
                print('Epoch {} step {} Loss {:.4f}'.format(epoch + 1, step + 1, batch_loss.numpy()))
            if (epoch + 1) % 5 == 0:
                # 每隔一定的epoch保存模型的参数
                self.encoder.save('./models/encoder_%02d.h5' % ((epoch + 1) / 5))
                self.decoder.save('./models/decoder_%02d.h5' % ((epoch + 1) / 5))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    def test(self, args, inference):
        sentence = args.sentence
        result, sentence = inference(self.process, self.encoder, self.decoder, sentence)
        print(sentence + '-->' + result.replace(' ', ''))


if __name__ == '__main__':

    # 参数路径与数据路径
    data_path = './data/qingyun.tsv'
    encoder_weights_path = './models/encoder.h5'
    decoder_weights_path = './models/decoder.h5'
    model_weights_path = './models'

    # 超参数定义
    batch_size = 64
    embedding_dim = 50
    units = 256
    epochs = 10

    process = PreProcess(data_path, samples_num=3000)

    nlp_model = NlpModel(process=process, embedding_dim=embedding_dim,units=units,batch_size=batch_size,
                         encoder_weights_path=encoder_weights_path,
                         decoder_weights_path=decoder_weights_path)
    # 训练模型
    nlp_model.train(epochs=epochs, model_weights_path=model_weights_path)

    # 测试模型
    parse = argparse.ArgumentParser(description='AITalk default setting params')
    parse.add_argument('--sentence', default='哈哈，你好啊', help='聊天内容')
    args = parse.parse_args()
    nlp_model.test(args, inference)




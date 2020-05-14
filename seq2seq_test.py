from model.seq2seq_attention import encoder_model, decoder_model, inference
from pre_process import PreProcess
from tensorflow.keras.layers import Input
import argparse
import os

process = PreProcess('./data/qingyun.tsv')

# define params
embedding_dim = 50
units = 256


def _main(args):
    sentence = args.sentence
    encoder_weights_path = args.encoder_weights_path
    decoder_weights_path = args.decoder_weights_path
    if not os.path.exists(encoder_weights_path) or not os.path.exists(decoder_weights_path):
        raise ValueError('weights path should exists')
    # get model
    encoder_input = Input((process.q_lenght,))
    encoder = encoder_model(encoder_input, process.q_vocab_size, embedding_dim, units)

    decoder_input, hidden_input, encoder_output_input = Input((1,)), Input((units,)), Input((process.q_lenght, units))
    decoder = decoder_model(decoder_input, hidden_input, encoder_output_input, process.a_vocab_size, embedding_dim,
                            units)
    # load model weights
    encoder.load_weights(encoder_weights_path, by_name=True)
    decoder.load_weights(decoder_weights_path, by_name=True)
    result, sentence = inference(process, encoder, decoder, sentence)
    print(sentence + '-->' + result.replace(' ', ''))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='AITalk default setting params')
    # argparse 使用示例
    # parser.add_argument('--sentence', default='', help='聊天内容', choices=['', ''], required=True)
    parse.add_argument('--encoder_weights_path', default='./models/encoder.h5', help='encoder weights')
    parse.add_argument('--decoder_weights_path', default='./models/decoder.h5', help='decoder weights')
    # 我怀疑你是个傻子-->你是不是觉得这日子过得太无聊了哦
    parse.add_argument('--sentence', default='哈哈', help='聊天内容')
    args = parse.parse_args()
    _main(args)

from flask import Flask, request, Response, json
from seq2seq_model.encoder_decoder import encoder_model, decoder_model, inference
from util.pre_process import PreProcess
from tensorflow.keras.layers import Input

'''
特别注意：

在pycharm里面右键运行带有flask的python程序会默认去bin目录下执行，这会导致代码中相对路径失败

因此，请不要在pycharm里面右键运行当前server.py程序

请在bash里面运行server.py程序

'''
app = Flask(__name__)


def get_response(sentence):
    # 构造模型
    encoder_input = Input((process.q_lenght,))
    encoder = encoder_model(encoder_input, process.q_vocab_size, embedding_dim, units)

    decoder_input, hidden_input, encoder_output_input = Input((1,)), Input((units,)), Input((process.q_lenght, units))
    decoder = decoder_model(decoder_input, hidden_input, encoder_output_input, process.a_vocab_size,
                            embedding_dim, units)
    # 加载训练的权重参数
    encoder.load_weights(encoder_weights_path)
    decoder.load_weights(decoder_weights_path)
    result, sentence = inference(process, encoder, decoder, sentence)
    return result.replace(' ', ''), sentence


@app.route('/talk', methods=['POST', 'GET'])
def talk():
    content = request.args.get('content')
    result, _ = get_response(content)
    return Response(json.dumps({'nickName': '小AI', 'content': result, 'type': 'text'}, ensure_ascii=False), content_type='application/json')


if __name__ == '__main__':

    # 参数与数据存储路径
    encoder_weights_path = './models/encoder.h5'
    decoder_weights_path = './models/decoder.h5'
    data_path = './data/qingyun.tsv'
    # 定义参数
    embedding_dim = 50
    units = 256
    process = PreProcess(data_path)

    app.run(host="127.0.0.1", port=8000, debug=False)
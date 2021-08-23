import jieba
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


class PreProcess(object):
    """
    数据预处理类
    """
    def __init__(self, file_path, max_length=25, samples_num=300, is_char=False,
                 random_choose=False, filter_max_lenght=False):
        """
        :param file_path:
        :param max_length 语句的最大长度
        :param samples_num:
        :param is_char:
        :param random_choose: 随机挑选样本
        """
        self.is_char = is_char
        # 将分词后的语句写入新文件
        abs_path = os.path.abspath(file_path)
        save_path = abs_path.split('.tsv')[0] + '_jieba.tsv'
        if not os.path.exists(save_path):
            # 如果文件不存在，则需要解析，只有第一次的时候需要解析
            self._parse_file(file_path, save_path)
        # 得到了问句和答句两个多维元祖
        q, a = self._load_parse_file(save_path)
        # 过滤掉长度过长的句子
        if filter_max_lenght:
            q, a, filter_length = self._filter_max_lenght(q, a, max_length)
            if samples_num > filter_length:
                samples_num = filter_length
        # 加载数据
        self.q_tensor, self.q_tokenizer, self.a_tensor, self.a_tokenizer = self._load_data(q, a)
        # 过滤掉超过max_length 的语句
        # 取samples_num个用来训练
        self._choose_train_samples(random_choose, samples_num)

        # self._random_shuffle()
        # self.q_tensor_train, self.q_tensor_val, self.a_tensor_train, self.a_tensor_val = \
        #     train_test_split(self.q_tensor, self.a_tensor, test_size=0.2)

    def _filter_max_lenght(self, q, a, max_length):
        '''
        过滤掉长度超过max_length的句子
        问句与答句任意一个超过25个词语，将问答句一起删除
        '''
        q_filter = []
        a_filter = []
        for q, a in zip(*[q, a]):
            if len(q) > max_length or len(a) > max_length:
                continue
            q_filter.append(q)
            a_filter.append(a)
        return q_filter, a_filter, len(q_filter)

    def _choose_train_samples(self, random_choose, samples_num):
        if random_choose:
            ids = np.random.choice(np.arange(self.length), samples_num, replace=False)
        else:
            ids = np.arange(samples_num)
        self.q_tensor = self.q_tensor[ids]
        self.a_tensor = self.a_tensor[ids]

    def _random_shuffle(self):
        ids = np.arange(self.length)
        np.random.shuffle(ids)
        self.q_tensor = self.q_tensor[ids]
        self.a_tensor = self.a_tensor[ids]

    def _parse_file(self, file_path, save_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            # 主要是把一些停用词去掉
            lines = self._purify(f.readlines())
            data = [line.split('\t') for line in lines]
            if self.is_char:
                pass
            else:
                data = [(jieba.lcut(qa[0]), jieba.lcut(qa[1])) for qa in data]
        f.close()
        with open(save_path, 'w') as f:
            for line in data:
                q, a = line
                content = ' '.join(q) + '\t' + ' '.join(a) + '\n'
                f.write(content)
        f.close()

    def _load_parse_file(self, file_path):
        '''
        加载解析文件，完成语料的预处理，主要是：
        将每一个对话封装成一个二元祖，并给每一句话的前后加上标识符
        返回的是[(问句),(答句)]
        '''
        with open(file_path, 'r', encoding='utf-8') as f:
            # 得到一个文本列表，文件中的每一行是一个列表元素
            lines = f.readlines()  
            # 每一行是一个对话语料，用\t分割，下面是将每一句话开头和结尾加上标记，并将每一个对话组成一个元祖
            data = [('<eos> ' + line.split('\t')[0] + ' <sos>', '<eos> ' + line.split('\t')[1].strip('\n') + ' <sos>') for line in lines]
        f.close()
        return zip(*data)

    def tokenize(self, qa):
        '''
        将文本数值化, 将每一个词映射成一个数值表示
        '''
        tokenizer = Tokenizer(filters='', split=' ', num_words=None)
        tokenizer.fit_on_texts(qa)
        # 得到得是一个列表，列表中的每一个元素还是一个列表，每一个列表代表一句话
        tensor = tokenizer.texts_to_sequences(qa)
        # 每一句话填充成相同长度;
        tensor = pad_sequences(tensor, padding='post', maxlen=25)
        return tensor, tokenizer

    def val_process(self, sentence):
        '''
        将给定的句子转化输入tensor
        '''
        '''
               # # 将一句话用jieba切词
        # words = jieba.cut(sentence)
        # # 为句子添加前后标记，<eos> 你好 啊 ， 我 是 张 <sos>
        # words = '<eos> ' + ' '.join(words) + ' <sos>'
        #
        # tensor = self.q_tokenizer.texts_to_sequences(words)
        # print(tensor)
        # tensor = pad_sequences(tensor, padding='post')
        # print(tensor)
        # print("*****************")
        # print(tensor.shape)
        # return tensor.reshape((1, -1)) 
        '''
        # 将一句话用jieba切词
        words = jieba.cut(sentence)
        # 为句子添加前后标记，<eos> 你好 啊 ， 我 是 张 <sos>
        words = ['<eos> ' + ' '.join(words) + ' <sos>']
        # texts_to_sequences接受一个列表，列表中每一个元素是一个句子
        # 将每一个句子转化成数值列表，返回的是由列表组成的列表
        tensor = self.q_tokenizer.texts_to_sequences(words)
        # 为了固定长度，额外添加一个定长的数组，方便padding的时候是定长的
        tensor.append([0 for i in range(25)])
        # 返回的是numpy对象
        tensor = pad_sequences(tensor, padding='post')
        # (1,24)
        tensor = tensor[:1, :]
        return tensor

    def _purify(self, lines):
        return [self.filter_sentence(line) for line in lines]

    # 去掉一些停用词
    def filter_sentence(self, sentence):
        return sentence.replace('\n', '').replace(' ', '').replace('，', ',').replace('。', '.'). \
            replace('；', '：').replace('？', '?').replace('！', '!').replace('“', '"'). \
            replace('”', '"').replace("‘", "'").replace("’", "'").replace('（', '(').replace('）', ')')

    def _load_data(self, q, a):
        q_tensor, q_tokenizer = self.tokenize(q)
        a_tensor, a_tokenizer = self.tokenize(a)
        return q_tensor, q_tokenizer, a_tensor, a_tokenizer

    @property
    def length(self):
        '''
        计算语料中句子的数量，其中问句和答句的数量应该是相等的
        '''
        if len(self.q_tensor) != len(self.a_tensor):
            raise ValueError('data load error please check it!!!')
        return len(self.q_tensor)

    @property
    def q_vocab_size(self):
        '''
        tokenizer.word_index是单词到数值的字典映射
        返回的是预料中不重复的单词的数量，其中+1是加上标记词
        '''
        return len(self.q_tokenizer.word_index) + 1

    @property
    def a_vocab_size(self):
        return len(self.a_tokenizer.word_index) + 1

    def test(self, lang, tensor):
        for t in tensor:
            if t != 0:
                print("%d ----> %s" % (t, lang.index_word[t]))

    @property
    def q_lenght(self):
        '''
        每一个句子中包含的单词的数量
        '''
        return len(self.q_tensor[-1])

    @property
    def a_lenght(self):
        return len(self.a_tensor[-1])

    def generate_data(self, bath_size):
        '''
        从全部的语料库中随机的选择batch_size数量的问答句
        需要注意的是，问句和回答句应该是一一对应的
        返回的是(batch_size, seq_length)维度的问句和答句的tensor
        '''
        if bath_size <= 0:
            raise ValueError('batch_size should > 0')
        i = 0
        while True:
            # 简单实现
            q_tensor = []
            a_tensor = []
            for _ in range(bath_size):
                if i == 0:
                    self._random_shuffle()
                q = self.q_tensor[i]
                a = self.a_tensor[i]
                q_tensor.append(q)
                a_tensor.append(a)
                i = (i + 1) % self.length
            yield np.array(q_tensor), np.array(a_tensor)

# process = PreProcess('./data/train.txt')
# print(len(process.q_tensor))
# data = process.val_process('你猜猜看我是谁')
# print(data, data.shape)
#
# print(process.test(process.a_tokenizer, process.a_tensor[-1]))
# print(process.a_tensor[-1])

# generate = process.generate_data(10)
# q, a = next(generate)
# print(q.shape, a.shape)

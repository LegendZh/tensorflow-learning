"""
实现一个基本的循环神经网路
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow import keras
import numpy as np

tf.random.set_seed(22)
np.random.seed(22)

# 本实验使用的是评价的数据集-IMDB数据集
# 里面的参数用来限制单词个数，这里选择最常见的10000个单词
batchsz = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# 对数据进行padding，完成对于长句子的剪枝操作，按照最大允许长度进行剪枝
# x_train:[b,80],x_test:[b,80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 小于batch_size的时候，需要过滤掉
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(x_train), tf.reduce_min(x_train))
print('x_test shape:', x_test.shape)


# 定义网络主体（单层RNN）
class MyRnn(keras.Model):
    def __init__(self, units):
        super(MyRnn, self).__init__()
        # 每一次进入到网络中的时候都要初始化
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # 变化成embedding编码数据类型
        # [b,80]=>[b,80,100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b,80,100], h_dim:64
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        # 增加第二个cell
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)
        # 全连接层,fc,[b,80,100]=>[b,64]
        # 转换成我们想要的分类结果
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        完成前向计算过程
        net(x) net(x,training=True):train mode
        net(x,training=False) 测试模式
        :param inputs: [b,80]
        :param training: 表示计算过程为训练过程还是测试过程
        :param mask:
        :return:
        """
        # 输入维度[b,80]
        x = inputs
        # embedding=>[b,80,100]
        x = self.embedding(x)
        # rnn cell compute, [b,80,100]=>[b]
        # [b,80,100]=>[b,64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word:[b,100]
            # x+wxh+h*whh 得到一个新的状态h1
            # 自动覆盖状态
            out, state0 = self.rnn_cell0(word, state0, training)
            # 加入第二层的卷积神经网络
            out1, state1 = self.rnn_cell1(out, state1)
            # out:[b,64]=》[b,1] 语义抽取完成之后的向量
        x = self.outlayer(out1)
        # 计算情感的概率
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 4

    model = MyRnn(units)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()

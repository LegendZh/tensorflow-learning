"""
使用keras，通过自定义神经网络层完成对于cifar10数据集的识别与分类
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    # 输入范围[0,255]=>[-1,1]
    """
    对数据的预处理函数
    :param x: 输入的数据集
    :param y: 数据集的label
    :return: 处理完成的数据集
    """
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
# [32,32,3]
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.one_hot(tf.squeeze(y), depth=10)
y_val = tf.one_hot(tf.squeeze(y_val), depth=10)
print('datasets', x.shape, y.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)


# 建立自己的神经网络层
class MyDense(layers.Layer):
    # 代替标准的网络层
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        # 设置为无偏置
        # self.bias=self.add_weight('b',[outp_dim])

    def call(self, inputs, training=None):
        # 建立单层神经网络需要执行的操作
        x = inputs @ self.kernel
        return x


# 定义自己的神经网络
class MyNetWork(keras.Model):
    def __init__(self):
        # 初始化整个神经网络
        super(MyNetWork, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)

    def call(self, inputs, training=None):
        """
        定义我们的网络前向传播结构
        :param inputs: [b,32,3,3]
        :param training:
        :return:
        """
        inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = self.fc5(x)
        return x


network = MyNetWork()
# 指定训练参数
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# 拟合神经网络,并进行验证
network.fit(train_db, epochs=50, validation_data=test_db, validation_freq=1)

# 模型的保存
network.evaluate(test_db)
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')

# 重新创建
network = MyNetWork()
# 指定训练参数
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print('loaded from ckpt/weights.ckpt')
network.evaluate(test_db)

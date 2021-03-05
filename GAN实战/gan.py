"""
实战操作GAN网络，使用Anime数据集作为例子
GAN网络比较复杂，具体解析见 https://blog.csdn.net/ifreewolf_csdn/article/details/89309912
WGAN的介绍见 https://www.cnblogs.com/Allen-rg/p/10305125.html
DCGAN的介绍见 https://blog.csdn.net/qq_33594380/article/details/84135797
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
import numpy as np


# 定义生成器Generator网络，作用是将一个随机的噪声生成正式的图片
class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        # z: [b,100]=>[b,3*3*512]=>[b,3,3,512]=>[b ,64,64,3]，相当于生成一个随机图像
        self.fc = layers.Dense(3 * 3 * 512)
        # 三个卷积层（反向卷积层）
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)
        # 数据进入卷积层
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        # 将像素的值压缩到[-1,1]
        x = tf.tanh(x)

        return x


# 定义测试Discriminator网络，作用是将噪声生成的图片和训练集中的图片进行对比
# 和前面的网络做一个对抗，这也是GAN网络的核心所在
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3]=>[b,1] 相当于生成一个分类器，观察positive的概率
        # 三个卷积层+一个全连接层
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')
        # 卷积层和BatchNormalazion层一般同时出现
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()
        # [b,h,w,3]=>[b,-1] 执行打平操作
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # flatten
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


def main():
    # 检测网络的输入输出
    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])
    # 对生成结果进行分类
    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)


if __name__ == '__main__':
    main()

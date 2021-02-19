"""
实现基础的resnet（resnet-18/34）
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow import keras


# 定义一个基本的Resnet网络单元(conv+batchnormalization+relu激活函数*2,加上一个反馈单元)
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride):
        """
        初始化一个基础的ResNet块，包含两层以及一个identity操作
        :param filter_num: 卷积核通道的数量（channel）
        :param stride: 移动的步长，用来指定图像输出的大小
        """
        super(BasicBlock, self).__init__()
        # 一个convolution layer,实际上是三层
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二层，两层组合在一起为一个resnet单元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        # 加入一个下采样层，为了实现通道升，尺度降的操作
        # 当stride步长参数不为1的时候，我们需要一个下采样操作来保持维度一致
        # 反之则不执行操作
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # [b,h,w,c],经过一个输入
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # identity层的输入与前面的卷积层进行融合
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


# 部署网络
class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100):
        """
        :param layer_dims: 每一层的维度（决定取舍，Resnet的核心）
        :param num_classes: 需要区分多少类
        """
        super(ResNet, self).__init__()
        # 第一层是一个固定的结构
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
        # 加入中间的ResBlock（一共是4个）
        self.layers1 = self.build_resblock(64, layer_dims[0])
        self.layers2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layers3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layers4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 卷积层输出后，输出进入全连接层
        self.avgpool = layers.GlobalAvgPool2D()  # 不确定的时候，利用一个均值变换层将图片的二维度消掉，变成一个一维的向量
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        """
        前向传播运算过程
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self.stem(inputs)

        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)

        x = self.avgpool(x)  # 不需要reshape,直接进入全连接层
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        """
        构建一个二阶的resnet block，只有第一个小的basic block可以执行下采样，第二个块是不能执行的
        也就是，一个二阶的block有4层，Resnet-18有4个这样的块
        :param stride: 定义步长，默认为1（走过两个卷积网络层）
        :param filter_num: 卷积核通道的数量（channel）
        :param blocks: 多少个resnet block
        :return:
        """
        res_blocks = Sequential()
        # 可能会出现下采样
        res_blocks.add(BasicBlock(filter_num, stride))
        # 后续的blocks不会做下采样，保持输入输出一致
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    """
    通过指定参数，定义一个resnet-18网络
    :return: resnet-18网络
    """
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])

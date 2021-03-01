"""
实现一个简单的自编码器
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(22)
np.random.seed(22)


# 保存图片的一个辅助函数
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


# 定义超参数
h_dim = 20  # 将784维的图片数据降维到20维
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 我们不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print('x_train shape:', x_train.shape, y_train.shape)
print('x_test shape:', x_test.shape, y_train.shape)


# 建立网络
class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder(编码器，抽象化)，一个三层的降维的神经网络
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders，将原始的特征图像释放成一个原始的图像
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None, mask=None):
        # [b,784]=>[b,10]
        h = self.encoder(inputs)
        # [b,10]=>[b,784]
        x_hat = self.decoder(h)
        return x_hat


# 创建模型
model = AE()
model.build(input_shape=(None, 784))
model.summary()

optimizer = keras.optimizers.Adam(lr=lr)

# 开始训练
for epoch in range(100):
    for step, x, in enumerate(train_db):
        # [b,28,28]=>[b,784]
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        # 评估生成器生成的结果
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b,784]=>[b,28,28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        # [b,28,28]=>[2b,28,28]
        # x:原始图片，x_hat：重建图片
        x_concat = tf.concat([x, x_hat], axis=0)
        # x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)

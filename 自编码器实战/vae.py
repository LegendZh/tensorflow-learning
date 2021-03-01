"""
实现一个VAE自编码器神经网络
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

z_dim = 10


# 新建VAE神经网络
class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)  # get mean prediction
        self.fc3 = layers.Dense(z_dim)

        # Decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    # 定义encoder的传播过程
    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)
        # get variance
        log_var = self.fc3(h)
        return mu, log_var

    # 定义decoder
    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    # 进行一个采样层的采样操作
    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(tf.shape(log_var))
        std = tf.exp(log_var) ** 0.5
        z = mu + std * eps
        return z

    def call(self, inputs, training=None, mask=None):
        # [b,784]=>[b,z_dim],[b,z_dim]
        mu, log_var = self.encoder(inputs)
        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        # 返回均值和方差，利用这两个值进行约束
        return x_hat, mu, log_var


model = VAE()
model.build(input_shape=(None, 784))
optimizer = keras.optimizers.Adam(lr=lr)

# 开始训练
for epoch in range(100):
    for step, x, in enumerate(train_db):
        # [b,28,28]=>[b,784]
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            # compute kl divergence(mu,var)~N(0,1)
            # log_var=>log方差的平方，具体公式见笔记19页
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            # 最终的loss是两个的一个结合
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]

            loss = rec_loss + 1. * kl_div
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

    # 编码器的结果
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_image/sampled_epoch_%d.png' % epoch)

    # 评估生成器生成的结果
    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_image/rec_epoch_%d.png' % epoch)

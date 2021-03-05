"""
利用数据集，完成训练
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
from gan import Generator, Discriminator
from dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):  # 把图片拼接后保存
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)
        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            # reset single row
            single_row = np.array([])
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # 生成真图片的交叉熵，让真的尽可能是真的
    # logits : [b, 1]
    # labels : [b] = [1, 1, 1, 1, ...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算假图片的交叉熵，与上面的函数作用相反，让假的尽可能是假的
    # logits : [b, 1]
    # labels : [b] = [0, 0, 0, 0, ...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c], 把 t 广播为 batch_x 的size, 来进行插值
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # 对于每个样本求二范数
    gp = tf.reduce_mean((gp - 1) ** 2)  # 求所有样本的均方差

    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. 将真实存在于训练集的图片视作为真
    # 2. 将由生成神经网络生成的图片标记为假
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    # WGAN的特殊性，使用Wasserstein DIstance去表示梯度
    loss = d_loss_fake + d_loss_real + 1. * gp  # lamda = 1
    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)

    loss = celoss_ones(d_fake_logits)  # 假尽可能真
    return loss


def main():
    # 设计随机种子，方便复现
    tf.random.set_seed(22)
    np.random.seed(22)
    # 设定相关参数
    z_dim = 100
    epochs = 3000000
    batch_size = 512  # 根据自己的GPU能力设计
    learning_rate = 0.002
    is_training = True
    # 加载数据（根据自己的路径更改），建立网络
    img_path = glob.glob(
        r'C:\Users\Jackie Loong\Downloads\DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master\data\faces\*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    # print(dataset, img_shape)
    # sample = next(iter(dataset))
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    # 建立优化器
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        # 随机取样出来的结果
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # 训练检测网络
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 训练生成网络
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'gp:', float(gp))
            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            # 生成的图片保存，images文件夹下, 图片名为：wgan-epoch.png
            img_path = os.path.join('images', 'gan-%d.png' % epoch)
            # 10*10, 彩色图片
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()

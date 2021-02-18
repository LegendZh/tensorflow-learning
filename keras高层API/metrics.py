"""
前面的代码使用的是之前对于mnist的识别代码
加入了keras.metrics方法进行统计
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics



def preprocess(x, y):
    """
    对数据的预处理函数
    :param x: 输入的数据集
    :param y: 数据集的label
    :return: 处理完成的数据集
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 导入数据
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
# print(x.shape, y.shape)
# 对数据进行预处理以及batch操作
batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

# 放入迭代器
db_iter = iter(db)
sample = next(db_iter)
print('batch: ', sample[0].shape, sample[1].shape)

# 新建网络(五层的神经网络)
# 注意，建立网络的时候要为输出的维度
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b,784]->[b,256]
    layers.Dense(128, activation=tf.nn.relu),  # [b,256]->[b,128]
    layers.Dense(64, activation=tf.nn.relu),  # [b,128]->[b,64]
    layers.Dense(32, activation=tf.nn.relu),  # [b,64]->[b,32]
    layers.Dense(10, activation=tf.nn.relu),  # [b,32]->[b,10]
])

# 初始化网络
model.build(input_shape=[None, 28 * 28])
model.summary()  # 调试网络
# 优化器，对数据进行梯度下降的更新
optimizers = optimizers.Adam(lr=1e-3)
# 1.初始化统计的meter
loss_meter = metrics.Mean()
acc_meter = metrics.Accuracy()


# 数据集的预处理
def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            # x:[b,28,28]
            # y:[b]
            x = tf.reshape(x, [-1, 28 * 28])

            # 利用梯度对网络进行更新
            with tf.GradientTape() as tape:
                # [b,784]=>[b,10], 直接调用，实现网络的传播
                logits = model(x)
                # onehot编码
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                # 2.更新loss值
                loss_meter.update_state(loss)

            # 利用构建好的优化器直接对数据进行优化(使用交叉熵进行发现传播)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizers.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                # 3.打印出来现在想要的loss
                print(step, 'loss: ', loss_meter.result().numpy())
                # 4.清除掉之前保存的准确率数据
                loss_meter.reset_states()

        # test,做前向传播，检查正确率
        # 调用之前要清空缓存
        acc_meter.reset_states()
        for x, y in db_test:
            # x:[b,28,28]
            # y:[b]
            x = tf.reshape(x, [-1, 28 * 28])
            # 不需要做梯度下降，所以不需要被包围
            # [b,784]=>[b,10]
            logits = model(x)
            # logits=>概率,[b,10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b,10] => [b]
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            # pred:[b];y:[b]
            acc_meter.update_state(y, pred)

        print(epoch, 'test acc:', acc_meter.result().numpy())


if __name__ == '__main__':
    main()

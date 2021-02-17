import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            # 利用构建好的优化器直接对数据进行优化(使用交叉熵进行发现传播)
            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizers.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss', float(loss_ce), float(loss_mse))

        # test,做前向传播，检查正确率
        total_correct = 0
        total_num = 0
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
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()

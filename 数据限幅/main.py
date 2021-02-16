import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 50.
y = tf.convert_to_tensor(y)
y = tf.one_hot(y, depth=10)
print('x:', x.shape, 'y:', y.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128).repeat(30)
x, y = next(iter(train_db))
print('sample:', x.shape, y.shape)


def main():
    # 784=>512
    w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.01))  # 解决梯度爆炸问题
    b1 = tf.Variable(tf.zeros([512]))
    # 512=>256
    w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.01))
    b2 = tf.Variable(tf.zeros([256]))
    # 256 => 10
    w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.01))
    b3 = tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr=0.01)

    for step, (x, y) in enumerate(train_db):
        # x:[128,28,28]
        # y:[128]

        # 需要去做一个维度变换，才能够将数据载入到我们的前向传播中
        # [b,28,28]->[b,28*28]
        x = tf.reshape(x, [-1, 28 * 28])

        # 包装在一个求导的环境下，方便计算
        # 出现与权值相关的代码一定要放进去
        with tf.GradientTape() as tape:  # 只会跟踪tf.Variable类型的数据，如果将权值定义成常量，会报错
            # x:[b,28*28]
            # h1=x@w1+b1
            # [b,784]@[784,256]+[256]->[b,256]+[256]->[b,256]+[b,256]
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)  # 非线性层激活函数
            # [b,256]->[b,128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层
            out = h2 @ w3 + b3

            # compute loss
            # out:[b,10]
            # y:[b]-> [b,10]

            # mse=mean(sum(y-out)^2)
            # [b,10]
            loss = tf.square(y - out)
            # mean:scalar
            loss = tf.reduce_mean(loss, axis=1)
            # [b]=>scalar
            loss = tf.reduce_mean(loss)
        # 利用上面包装好的部分计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 观察可能出现的梯度下降过多的问题
        print("==before==")
        for g in grads:
            print(tf.norm(g))

        # 限制下降的梯度幅度
        grads, _ = tf.clip_by_global_norm(grads, 15)
        print("==after==")
        for g in grads:
            print(tf.norm(g))
        # 更新梯度
        optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))

        if step % 100 == 0:
            print(step, 'loss:', float(loss))


if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# x: [60k,28,28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()

# 将数据集转化为一个tensor变量
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 一个batch取128个数值
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)  # 迭代器
sample = next(train_iter)
print('batch', sample[0].shape, sample[1].shape)

# 创建一个模拟的神经网络，加上权值
# 神经网络结构：[b,784]->[b,256]->[b,128]->[b,10]
# 输入参数为[dim_in,dim_out],输出为[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.01))  # 解决梯度爆炸问题
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.01))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
lr = 1e-3  # 学习率

for epoch in range(10):  # 对数据集进行多次迭代
    for step, (x, y) in enumerate(train_db):  # 每个batch进行迭代
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
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)  # 非线性层激活函数
            # [b,256]->[b,128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层
            out = h2 @ w3 + b3

            # compute loss
            # out:[b,10]
            # y:[b]-> [b,10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse=mean(sum(y-out)^2)
            # [b,10]
            loss = tf.square(y_onehot - out)
            # mean:scalar
            loss = tf.reduce_mean(loss)

        # 利用上面包装好的部分计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1=w1-lr*w1_grad
        # 对参数进行原地更新，效果与前面的类似
        w1.assign_sub(lr * grads[0])  # 对应上面的参数位置
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

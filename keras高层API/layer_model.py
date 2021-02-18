import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras


def preprocess(x, y):
    """
    对于输入数据的预处理
    :param x: 一张简单的图片，不是一个batch
    :param y: 图片的label
    :return: 预处理完成后的x和y
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets: ', x.shape, y.shape, x.min(), x.max())

# 对数据进行预处理以及batch操作
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)
db_test = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_test = db_test.map(preprocess).batch(batchsz)

# 放入迭代器
db_iter = iter(db)
sample = next(db_iter)
print('batch: ', sample[0].shape, sample[1].shape)

# # 新建网络(五层的神经网络)
# # 注意，建立网络的时候要为输出的维度
# model = Sequential([
#     layers.Dense(256, activation=tf.nn.relu),  # [b,784]->[b,256]
#     layers.Dense(128, activation=tf.nn.relu),  # [b,256]->[b,128]
#     layers.Dense(64, activation=tf.nn.relu),  # [b,128]->[b,64]
#     layers.Dense(32, activation=tf.nn.relu),  # [b,64]->[b,32]
#     layers.Dense(10, activation=tf.nn.relu),  # [b,32]->[b,10]
# ])
#
# # 初始化网络
# model.build(input_shape=[None, 28 * 28])
# # 优化器，对数据进行梯度下降的更新
# optimizers = optimizers.Adam(lr=1e-3)
# model.summary()  # 调试网络


# 定义我们自己的层，方便加入到网络中
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        # 使用父类的初始化方法
        super(MyDense, self).__init__()

        # 定义自己的变量
        # 这里面的add_weight是替换了原先的add_variable; 效果相同
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


# 利用自己定义的层定义自己的神经网络模型
class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = self.fc5(x)
        return x


network = Mymodel()

# 指定训练参数
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# 拟合神经网络,并进行验证
network.fit(db, epochs=10, validation_data=db_test,
            validation_freq=2)

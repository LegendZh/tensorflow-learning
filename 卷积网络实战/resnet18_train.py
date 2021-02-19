"""
使用cifar100数据集，检测resnet-18的训练结果
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from resnet import resnet18

tf.random.set_seed(2345)

def preprocess(x, y):
    # 简单数据处理
    # 解决梯度离散现象，把输入值的范围变成[-1,1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
# 将一维的数据挤压掉
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

# 建立训练数据集和测试数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 将数据打乱并把数据预处理之后，把64个作为一个batch
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# batch大小与显卡有关
test_db = test_db.map(preprocess).batch(128)

sample = next(iter(train_db))
print("sample:", sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # [b,32,32,3]=>[b,1,1,512]
    model=resnet18()
    model.build(input_shape=(None,32,32,3))
    model.summary() #查看网络的参数
    ###这里的代码在写神经网络时候基本可以不用变了###
    # 确定学习率0.0001
    optimizer = optimizers.Adam(lr=1e-3)
    # 统计acc随着迭代而做出的变化
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b,32,32,3]=>[b,1,1,512]
                # 送入全连接层
                logits = model(x)
                # 变为one_hot编码模式
                y_onehot = tf.one_hot(y, depth=100)

                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                # 计算loss的平均值
                loss = tf.reduce_mean(loss)

            # 对参数进行求导
            grads = tape.gradient(loss, model.trainable_variables)
            # 针对所有的Variable根据给出的学习率进行求导
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 打印出来loss信息
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        # 对测试集进行测试，观察正确率
        for x, y in test_db:
            # 将测试集数据喂入已经训练好的神经网络中
            logits = model(x)
            # 选出最有可能的分类
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # 计算准确预测的个数
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            # 计算准确率
            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
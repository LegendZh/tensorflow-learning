import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 构造函数，接受一个列表为参数
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
# 将数据网格化
X, Y = np.meshgrid(x, y)
print('X,Y maps:', x.shape, y.shape)
Z = himmelblau([X, Y])

# 绘制函数的图像
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# 随机初始化一个点值
# [1.,0.],[-4.,0.],[4.,0.]
x = tf.constant([1., 0.])

for step in range(200):  # 循环执行200次
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads  # 确定按照梯度移动的距离

    # 打印出来x的状态
    if step % 20 == 0:
        print('step {}: x={}, f(x)={}'.format(step, x.numpy(), y.numpy()))

# 结论：不同的初始值可能会找到不同的结果

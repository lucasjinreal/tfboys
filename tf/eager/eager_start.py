import tensorflow as tf

x = [[2.3, 3.4],
[3.4, 4.5]]

m = tf.matmul(x, x)
print(m)
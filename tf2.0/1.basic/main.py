import tensorflow as tf

a = tf.constant([[2, 3], [4, 5]])
print(a)
b = tf.add(a, 2)
print(b)
c = a*b
print(c)
print(c.numpy())
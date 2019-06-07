from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, unit):
        super(MyLayer, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(shape=(input_dim, unit), dtype=tf.float32),
                                   trainable=True)

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(unit,), dtype=tf.float32),
                                trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


x = tf.ones((3, 10))
print(x)
my_layers = MyLayer(10, 20)
out = my_layers(x)
print(out)

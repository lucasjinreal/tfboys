# -----------------------
#
# Copyright Jin Fagang @2018
# 
# 1/3/19
# test
# -----------------------

import tensorflow as tf
import numpy as np


class MySimpleLayer(tf.keras.layers.Layer):

    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units

        self.conv_1 = None
        self.fc_1 = None

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid')
        self.fc_1 = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        a = self.conv_1(inputs)
        return a


if __name__ == '__main__':

    x = np.random.random_sample([128, 128, 3])
    print(x)

    my_simple_layer = MySimpleLayer(output_units=16)
    model = tf.keras.Sequential([
        my_simple_layer,
    ])

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    model.compile(optimizer=opt)
    print('model compiled.')

    a = model(x)
    print(a)



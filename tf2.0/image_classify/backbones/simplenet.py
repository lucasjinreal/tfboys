'''
some randome and simple nets
'''


import tensorflow as tf
from tensorflow.keras import layers


def build_net_001(input_shape, n_classes):
    assert len(input_shape) == 3, 'only support 3 channels'
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
        padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model
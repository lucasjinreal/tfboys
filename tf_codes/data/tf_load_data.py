"""
there are many ways to load data into tensorflow
"""
import tensorflow as tf
import numpy as np


def load_1():
    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
        retyped_image = tf.cast(decoded_image, tf.float32)
        image = tf.reshape(retyped_image, [784])
        label = tf.cast(features['label'], tf.int32)
        return image, label

    files = ['output.tfrecords']
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(parser)
    image, label = dataset.make_one_shot_iterator().get_next()


def load_2():
    # for text line datasets
    files = ['test_file_1.txt', 'test_file_2.txt']
    dataset = tf.data.TextLineDataset(files)
    l = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for i in range(4):
            print(sess.run(l))
"""
using estimator api in cnn image classification
"""
import sys
import numpy as np
import os
import tensorflow as tf

num_classes = 3
batch_size = 32
input_size = (128, 128)


# just for dummy
def load_image(data_dir):
    """
    this function return all images path, labels, classes back
    """
    train_dir = data_dir
    all_classes = []
    all_images = []
    all_labels = []
    for i in os.listdir(train_dir):
        current_dir = os.path.join(train_dir, i)
        if os.path.isdir(current_dir):
            all_classes.append(i)
            for img in os.listdir(current_dir):
                if img.endswith('png') or img.endswith('jpg'):
                    all_images.append(os.path.join(current_dir, img))
                    all_labels.append(all_classes.index(i))
    return all_images, all_labels, all_classes


def load_flowers_data(d):
    """
    load images, labels, classes from flowers
    d: data/flowers/images/
    :param d:
    :return:
    """
    pass


def input_map_fn(img_path, label):
    # do some process to label
    one_hot = tf.one_hot(label, num_classes)
    img_f = tf.read_file(img_path)
    img_decodes = tf.image.decode_image(img_f, channels=3)
    return img_decodes, one_hot


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, input_size[0], input_size[1], 1])
    inputs = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)

    inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
    inputs = tf.layers.flatten(inputs=inputs)

    inputs = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
    # only for training
    inputs = tf.layers.dropout(inputs=inputs, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes, activation=tf.nn.relu)
    predictions = {
        "classes": tf.argmax(input=inputs, axis=1),
        "probabilities": tf.nn.softmax(inputs, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # loss is predict minus ground truth
        loss = predictions['probabilities'] - labels
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def test_data():
    # change into your image dir
    all_images, all_labels, all_classes = load_image('./data_flowers/flower_photos')
    num_classes = len(all_classes)
    print(all_classes)
    # convert all images list to tensor, using Dataset API to load
    train_data = tf.data.Dataset().batch(batch_size).from_tensor_slices(
        (tf.constant(all_images), tf.constant(all_labels)))
    train_data = train_data.map(input_map_fn)

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    next_elem = iterator.get_next()

    train_init_op = iterator.make_initializer(train_data)
    with tf.Session() as sess:
        sess.run(train_init_op)
        while True:
            try:
                print(sess.run(next_elem))
            except tf.errors.OutOfRangeError:
                print('data iterator finish.')
                break


def train():
    """
    we using estimator API to start train
    """
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./flower_checkpoints")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])


if __name__ == '__main__':
    train()

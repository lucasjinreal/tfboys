from alfred.dl.tf.common import mute_tf
mute_tf()
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from alfred.utils.log import logger as logging
import os
import sys


target_size = 224
# use_keras_fit = True
use_keras_fit = False
ckpt_path = './checkpoints/flowers_mbv2-{epoch}.ckpt'


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] /= 255.
    x['image'] = 2*x['image'] - 1
    return x['image'], x['label']

def build_model():
    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_shape=(target_size, target_size, 3),
        weights='imagenet',
        include_top=False)
    mobilenetv2.trainable=True
    model = tf.keras.Sequential([
        mobilenetv2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5)
    ])
    return model

def train():
    # using mobilenetv2 classify tf_flowers dataset
    dataset, _ = tfds.load('tf_flowers', with_info=True)
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(4).repeat()

    # init model
    model = build_model()
    # model.summary()
    logging.info('model loaded.')
    
    start_epoch = 0
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('passing resume since weights not there. training from scratch')

    if use_keras_fit:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.fit(train_dataset, epochs=50)
        model.save(ckpt_path.format(epoch='final'))
        # logging.error('this not currently possiable to fit a generator into keras.model.')
    else:
        loss_object = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

        train_loss = tf.metrics.Mean(name='train_loss')
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(start_epoch, 120):
            try:
                for batch, data in enumerate(train_dataset):
                    images, labels = data
                    with tf.GradientTape() as tape:
                        predictions = model(images)
                        loss = loss_object(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    train_loss(loss)
                    train_accuracy(labels, predictions)
                    if batch % 50 == 0:
                        logging.info('Epoch: {}, iter: {}, loss: {}, train_accuracy: {}'.format(
                        epoch, batch, train_loss.result(), train_accuracy.result()))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)



if __name__ == "__main__":
    train()

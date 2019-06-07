from alfred.dl.tf.common import mute_tf
mute_tf()
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from alfred.utils.log import logger as logging
import os
import sys
from backbones.simplenet import build_net_001

'''

700/700 [==============================] - 22s 31ms/step - loss: 0.3416 - accuracy: 0.9010
Epoch 20/50
700/700 [==============================] - 22s 32ms/step - loss: 0.3237 - accuracy: 0.9092
Epoch 21/50
700/700 [==============================] - 22s 31ms/step - loss: 0.3102 - accuracy: 0.9135
Epoch 22/50
700/700 [==============================] - 22s 32ms/step - loss: 0.3366 - accuracy: 0.9079
Epoch 23/50
700/700 [==============================] - 22s 32ms/step - loss: 0.3053 - accuracy: 0.9185
Epoch 24/50
700/700 [==============================] - 22s 32ms/step - loss: 0.2657 - accuracy: 0.9278
Epoch 25/50
700/700 [==============================] - 22s 32ms/step - loss: 0.3059 - accuracy: 0.9146
Epoch 26/50
700/700 [==============================] - 22s 32ms/step - loss: 0.2769 - accuracy: 0.9282
Epoch 27/50
700/700 [==============================] - 22s 31ms/step - loss: 0.2785 - accuracy: 0.9257
Epoch 28/50
700/700 [==============================] - 22s 32ms/step - loss: 0.2614 - accuracy: 0.9267
Epoch 29/50
700/700 [==============================] - 22s 31ms/step - loss: 0.2877 - accuracy: 0.9217
Epoch 30/50
700/700 [==============================] - 22s 31ms/step - loss: 0.2739 - accuracy: 0.9275
Epoch 31/50
700/700 [==============================] - 22s 32ms/step - loss: 0.2860 - accuracy: 0.9249
Epoch 32/50
700/700 [==============================] - 22s 31ms/step - loss: 0.2358 - accuracy: 0.9367
Epoch 33/50
700/700 [==============================] - 22s 31ms/step - loss: 0.2442 - accuracy: 0.9317

'''

target_size = 224
use_keras_fit = True
# use_keras_fit = False
ckpt_path = './checkpoints/simplenet/flowers_simplenet-{epoch}.ckpt'


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] /= 255.
    x['image'] = 2 * x['image'] - 1
    return x['image'], x['label']


def train():
    # using mobilenetv2 classify tf_flowers dataset
    dataset, meta = tfds.load('tf_flowers', with_info=True)
    # print(meta)
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(4).repeat()

    for data in train_dataset.take(2):
        print(data)

    model = build_net_001((224, 224, 3), 5)
    model.summary()
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
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                               save_weights_only=True,
                                               verbose=1,
                                               period=1)
        ]
        try:
            model.fit(train_dataset, epochs=50,
                      steps_per_epoch=700, callbacks=callbacks)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=0))
            logging.info('keras model saved.')
        model.save_weights(ckpt_path.format(epoch=0))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'flowers_simplenet.h5'))

    else:
        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.Adam()

        train_loss = tf.metrics.Mean(name='train_loss')
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(start_epoch, 120):
            try:
                for batch, data in enumerate(train_dataset):
                    images, labels = data
                    with tf.GradientTape() as tape:
                        predictions = model(images)
                        loss = loss_fn(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    train_loss(loss)
                    train_accuracy(labels, predictions)

                    # todo: the compute method is wrong, the loss does not decrease for why?
                    if batch % 10 == 0:
                        logging.info('Epoch: {}, iter: {}, loss: {}, train_acc: {}'.format(
                            epoch, batch, train_loss.result(), train_accuracy.result()))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)


if __name__ == "__main__":
    train()

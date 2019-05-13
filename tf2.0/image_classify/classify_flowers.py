from alfred.dl.tf.common import mute_tf
mute_tf()
import tensorflow as tf
import tensorflow_datasets as tfds
from alfred.utils.log import logger as logging


target_size = 256


def preprocess(x):
    x['image'] = tf.image.resize(x['image'], (target_size, target_size)) / 255.
    return x

def train():
    # using mobilenetv2 classify tf_flowers dataset
    dataset, metadata = tfds.load('tf_flowers', with_info=True)
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(12).repeat()

    # init model
    model = tf.keras.applications.MobileNetV2(input_shape=(target_size, target_size), include_top=False, classes=5)
    logging.info('model loaded.')

    for epoch in range(0, 120):
        try:
            for batch, data in enumerate(train_dataset):
                images, labels = data['image'], data['label']
                print(images)
                print(labels)
                logging.info('Epoch: {}, iter: {}, loss: {}'.format(epoch, batch, 0))
        except KeyboardInterrupt:
            logging.info('interrupted.')
            exit(0)



if __name__ == "__main__":
    train()

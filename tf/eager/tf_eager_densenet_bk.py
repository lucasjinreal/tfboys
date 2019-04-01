from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2

input_shape = (224, 224)
channels = 3
num_classes = 80
batch_size = 4


# GPU better using channels_first
def data_format():
    return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'


def image_shape(batch_size_):
    if data_format() == 'channels_first':
        return [batch_size_, channels, input_shape[0], input_shape[1]]
    return [batch_size_, input_shape[0], input_shape[1], channels]


def random_batch(batch_size_):
    images = np.random.rand(*image_shape(batch_size_)).astype(np.float32)
    labels = np.random.randint(
        low=0, high=num_classes, size=[batch_size_]).astype(np.int32)
    one_hot = np.zeros((batch_size_, num_classes)).astype(np.float32)
    one_hot[np.arange(batch_size_), labels] = 1.
    return images, one_hot


class ConvBlock(tf.keras.Model):
    """Convolutional Block consisting of (batchnorm->relu->conv).

    Arguments:
      num_filters: number of filters passed to a convolutional layer.
      data_format: "channels_first" or "channels_last"
      bottleneck: if True, then a 1x1 Conv is performed followed by 3x3 Conv.
      weight_decay: weight decay
      dropout_rate: dropout rate.
    """

    def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4,
                 dropout_rate=0):
        super(ConvBlock, self).__init__()
        self.bottleneck = bottleneck

        axis = -1 if data_format == "channels_last" else 1
        inter_filter = num_filters * 4
        # don't forget to set use_bias=False when using batchnorm
        self.conv2 = tf.keras.layers.Conv2D(num_filters,
                                            (3, 3),
                                            padding="same",
                                            use_bias=False,
                                            data_format=data_format,
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=l2(weight_decay))
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if self.bottleneck:
            self.conv1 = tf.keras.layers.Conv2D(inter_filter,
                                                (1, 1),
                                                padding="same",
                                                use_bias=False,
                                                data_format=data_format,
                                                kernel_initializer="he_normal",
                                                kernel_regularizer=l2(weight_decay))
            self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

    def call(self, x, training=True):
        output = self.batchnorm1(x, training=training)

        if self.bottleneck:
            output = self.conv1(tf.nn.relu(output))
            output = self.batchnorm2(output, training=training)

        output = self.conv2(tf.nn.relu(output))
        output = self.dropout(output, training=training)

        return output


class TransitionBlock(tf.keras.Model):
    """Transition Block to reduce the number of features.

    Arguments:
      num_filters: number of filters passed to a convolutional layer.
      data_format: "channels_first" or "channels_last"
      weight_decay: weight decay
      dropout_rate: dropout rate.
    """

    def __init__(self, num_filters, data_format,
                 weight_decay=1e-4, dropout_rate=0):
        super(TransitionBlock, self).__init__()
        axis = -1 if data_format == "channels_last" else 1

        self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv = tf.keras.layers.Conv2D(num_filters,
                                           (1, 1),
                                           padding="same",
                                           use_bias=False,
                                           data_format=data_format,
                                           kernel_initializer="he_normal",
                                           kernel_regularizer=l2(weight_decay))
        self.avg_pool = tf.keras.layers.AveragePooling2D(data_format=data_format)

    def call(self, x, training=True):
        output = self.batchnorm(x, training=training)
        output = self.conv(tf.nn.relu(output))
        output = self.avg_pool(output)
        return output


class DenseBlock(tf.keras.Model):
    """Dense Block consisting of ConvBlocks where each block's
    output is concatenated with its input.

    Arguments:
      num_layers: Number of layers in each block.
      growth_rate: number of filters to add per conv block.
      data_format: "channels_first" or "channels_last"
      bottleneck: boolean, that decides which part of ConvBlock to call.
      weight_decay: weight decay
      dropout_rate: dropout rate.
    """

    def __init__(self, num_layers, growth_rate, data_format, bottleneck,
                 weight_decay=1e-4, dropout_rate=0):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.axis = -1 if data_format == "channels_last" else 1

        self.blocks = []
        for _ in range(int(self.num_layers)):
            self.blocks.append(ConvBlock(growth_rate,
                                         data_format,
                                         bottleneck,
                                         weight_decay,
                                         dropout_rate))

    def call(self, x, training=True):
        for i in range(int(self.num_layers)):
            output = self.blocks[i](x, training=training)
            x = tf.concat([x, output], axis=self.axis)

        return x


class DenseNet(tf.keras.Model):
    """Creating the Densenet Architecture.

    Arguments:
      depth_of_model: number of layers in the model.
      growth_rate: number of filters to add per conv block.
      num_of_blocks: number of dense blocks.
      output_classes: number of output classes.
      num_layers_in_each_block: number of layers in each block.
                                If -1, then we calculate this by (depth-3)/4.
                                If positive integer, then the it is used as the
                                  number of layers per block.
                                If list or tuple, then this list is used directly.
      data_format: "channels_first" or "channels_last"
      bottleneck: boolean, to decide which part of conv block to call.
      compression: reducing the number of inputs(filters) to the transition block.
      weight_decay: weight decay
      rate: dropout rate.
      pool_initial: If True add a 7x7 conv with stride 2 followed by 3x3 maxpool
                    else, do a 3x3 conv with stride 1.
      include_top: If true, GlobalAveragePooling Layer and Dense layer are
                   included.
    """

    def __init__(self, depth_of_model, growth_rate, num_of_blocks,
                 output_classes, num_layers_in_each_block, data_format='channels_last',
                 bottleneck=True, compression=0.5, weight_decay=1e-4,
                 dropout_rate=0, pool_initial=False, include_top=True):
        super(DenseNet, self).__init__()
        self.depth_of_model = depth_of_model
        self.growth_rate = growth_rate
        self.num_of_blocks = num_of_blocks
        self.output_classes = output_classes
        self.num_layers_in_each_block = num_layers_in_each_block
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.compression = compression
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.pool_initial = pool_initial
        self.include_top = include_top

        # deciding on number of layers in each block
        if isinstance(self.num_layers_in_each_block, list) or isinstance(
                self.num_layers_in_each_block, tuple):
            self.num_layers_in_each_block = list(self.num_layers_in_each_block)
        else:
            if self.num_layers_in_each_block == -1:
                if self.num_of_blocks != 3:
                    raise ValueError(
                        "Number of blocks must be 3 if num_layers_in_each_block is -1")
                if (self.depth_of_model - 4) % 3 == 0:
                    num_layers = (self.depth_of_model - 4) / 3
                    if self.bottleneck:
                        num_layers //= 2
                    self.num_layers_in_each_block = [num_layers] * self.num_of_blocks
                else:
                    raise ValueError("Depth must be 3N+4 if num_layer_in_each_block=-1")
            else:
                self.num_layers_in_each_block = [
                                                    self.num_layers_in_each_block] * self.num_of_blocks

        axis = -1 if self.data_format == "channels_last" else 1

        # setting the filters and stride of the initial covn layer.
        if self.pool_initial:
            init_filters = (7, 7)
            stride = (2, 2)
        else:
            init_filters = (3, 3)
            stride = (1, 1)

        self.num_filters = 2 * self.growth_rate

        # first conv and pool layer
        self.conv1 = tf.keras.layers.Conv2D(self.num_filters,
                                            init_filters,
                                            strides=stride,
                                            padding="same",
                                            use_bias=False,
                                            data_format=self.data_format,
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=l2(
                                                self.weight_decay))
        if self.pool_initial:
            self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                                      strides=(2, 2),
                                                      padding="same",
                                                      data_format=self.data_format)
            self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)

        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

        # last pooling and fc layer
        if self.include_top:
            self.last_pool = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self.data_format)
            self.classifier = tf.keras.layers.Dense(self.output_classes)

        # calculating the number of filters after each block
        num_filters_after_each_block = [self.num_filters]
        for i in range(1, self.num_of_blocks):
            temp_num_filters = num_filters_after_each_block[i - 1] + (
                    self.growth_rate * self.num_layers_in_each_block[i - 1])
            # using compression to reduce the number of inputs to the
            # transition block
            temp_num_filters = int(temp_num_filters * compression)
            num_filters_after_each_block.append(temp_num_filters)

        # dense block initialization
        self.dense_blocks = []
        self.transition_blocks = []
        for i in range(self.num_of_blocks):
            self.dense_blocks.append(DenseBlock(self.num_layers_in_each_block[i],
                                                self.growth_rate,
                                                self.data_format,
                                                self.bottleneck,
                                                self.weight_decay,
                                                self.dropout_rate))
            if i + 1 < self.num_of_blocks:
                self.transition_blocks.append(
                    TransitionBlock(num_filters_after_each_block[i + 1],
                                    self.data_format,
                                    self.weight_decay,
                                    self.dropout_rate))

    def call(self, x, training=True):
        output = self.conv1(x)

        if self.pool_initial:
            output = self.batchnorm1(output, training=training)
            output = tf.nn.relu(output)
            output = self.pool1(output)

        for i in range(self.num_of_blocks - 1):
            output = self.dense_blocks[i](output, training=training)
            output = self.transition_blocks[i](output, training=training)

        output = self.dense_blocks[self.num_of_blocks - 1](output, training=training)
        output = self.batchnorm2(output, training=training)
        output = tf.nn.relu(output)

        if self.include_top:
            output = self.last_pool(output)
            output = self.classifier(output)

        return output


def train(_model):
    if isinstance(_model, DenseNet):
        np_images, np_labels = random_batch(batch_size)
        print('Random generate {} samples.'.format(np_images.shape[0]))
        dataset = tf.data.Dataset.from_tensors((np_images, np_labels)).repeat()
        (images, labels) = dataset.make_one_shot_iterator().get_next()
        print(images)
        print(labels)

        loss = _model(images, training=True)
        cross_ent = tf.losses.softmax_cross_entropy(
            logits=loss, onehot_labels=labels)
        regularization = tf.add_n(_model.losses)
        loss = cross_ent + regularization
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(loss)

        train_steps = 5000
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print('Start to train...')
            for i in range(train_steps):
                loss_, _ = sess.run([loss, train_op])
                print('step: {}, loss: {}'.format(i, loss_))
            print('Done!')


if __name__ == '__main__':
    model = DenseNet(depth_of_model=50,
                     growth_rate=32,
                     num_of_blocks=4,
                     output_classes=num_classes,
                     num_layers_in_each_block=[6, 12, 24, 16],
                     data_format=data_format()
                     )
    print('Model build done.')
    train(model)
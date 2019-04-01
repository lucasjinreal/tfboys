import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()


a = tf.one_hot(2, 6)
print(a)
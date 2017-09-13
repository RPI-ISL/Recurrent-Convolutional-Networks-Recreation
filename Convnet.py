import tensorflow as tf

slim = tf.contrib.slim

class Convnet:
  """Light implemenetation of the Alexnet architecture."""
  def __init__(self, input_tensor, logits_size):
    conv1 = slim.conv2d(input_tensor, 96, [11, 11])
    conv2 = slim.conv2d(conv1, 256, [5, 5])
    pool1 = slim.max_pool2d(conv2, [2, 2])
    conv3 = slim.conv2d(pool1, 384, [3, 3])
    pool2 = slim.max_pool2d(conv3, [2, 2])
    conv4 = slim.conv2d(pool2, 384, [3, 3])
    conv5 = slim.conv2d(conv4, 256, [3, 3])
    pool3 = slim.max_pool2d(conv5, [2, 2])
    flattened = slim.flatten(pool3)
    fc = slim.stack(flattened, slim.fully_connected, [4096, 4096])
    logits = slim.fully_connected(fc, logits_size, activation_fn=None)


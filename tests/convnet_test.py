import Convnet
import tensorflow as tf


def test_convent_sanity():
  input_tensor = tf.placeholder(tf.float32, (None, 256, 256, 3))
  Convnet.Convnet(input_tensor, 1000)


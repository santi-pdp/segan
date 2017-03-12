from __future__ import print_function
import tensorflow as tf
from ops import *
import numpy as np


def read_and_decode(filename_queue, canvas_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(canvas_size)
    wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    noisy.set_shape(canvas_size)
    noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.

    return wave, noisy

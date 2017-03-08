from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
from qrnn import QRNN_layer
import numpy as np


class Generator(object):

    def __init__(self, segan):
        self.segan = segan

    def __call__(self, noisy_w, is_ref, spk=None):
        # TODO: remove c_vec
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan

        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        #z = make_z([segan.batch_size, segan.z_dim])

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True

        def reuse_wrapper(packed, *args):
            """ Wrapper that processes the output of TF calls differently
            based on whether we are reusing variables or not.
            packed: Output of TF call
            args: List of names

            If make_vars is True, packed will contain all new variables, and
            we assign them to segan.<field> fields.
            If make_vars is False, packed is just the output tensor, and we
            only return it.
            """
            if make_vars:
                assert len(packed) == len(args) + 1, len(packed)
                out = packed[0]
            else:
                out = packed
            return out

        print('*** Building Generator ***')
        in_dims = noisy_w.get_shape().as_list()
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(im_dims) < 2 or len(im_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 3

        skips = []
        for block_idx, dilation in enumerate(segan.g_dilated_blocks):
                name = 'g_residual_block_{}'.format(block_idx)
                res_i, skip_i = residual_block(h_i,
                                               dilation, kwidth, num_kernels=1,
                                               bias_init=None, stddev=0.02,
                                               name=name)
                # feed the residual output to the next block
                h_i = res_i
                # accumulate the skip connections
                skips.append(skip_i)
        print('Amount of skip connections: ', len(skips))
        # TODO: last pooling for actual wave
        with tf.variable_scope('g_wave_pooling'):
            skip_T = tf.stack(skips, axis=0)
            skips_sum = tf.reduce_sum(skip_T, axis=1)
            skips_sum = tf.nn.relu(skips_sum)
            wave_a = conv1d(skips_sum, kwidth=1, num_kernels=1,
                            init=tf.truncated_normal_initializer(stddev=0.02))
            wave = tf.tanh(wave_a)
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
        print('Last residual wave shape: ', res_i.get_shape())
        print('*************************')
        segan.generator_built = True
        return wave

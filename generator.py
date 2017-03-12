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
        h_i = noisy_w
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 3
        z = make_z([segan.batch_size, h_i.get_shape().as_list()[1],
                    segan.g_enc_depths[-1]])
        h_i = tf.concat(2, [h_i, z])
        skip_out = True
        skips = []
        for block_idx, dilation in enumerate(segan.g_dilated_blocks):
                name = 'g_residual_block_{}'.format(block_idx)
                if block_idx >= len(segan.g_dilated_blocks) - 1:
                    skip_out = False
                if skip_out:
                    res_i, skip_i = residual_block(h_i,
                                                   dilation, kwidth, num_kernels=32,
                                                   bias_init=None, stddev=0.02,
                                                   do_skip = True,
                                                   name=name)
                else:
                    res_i = residual_block(h_i,
                                           dilation, kwidth, num_kernels=32,
                                           bias_init=None, stddev=0.02,
                                           do_skip = False,
                                           name=name)
                # feed the residual output to the next block
                h_i = res_i
                if segan.keep_prob < 1:
                    print('Adding dropout w/ keep prob {} '
                          'to G'.format(segan.keep_prob))
                    h_i = tf.nn.dropout(h_i, segan.keep_prob_var)
                if skip_out:
                    # accumulate the skip connections
                    skips.append(skip_i)
                else:
                    # for last block, the residual output is appended
                    skips.append(res_i)
        print('Amount of skip connections: ', len(skips))
        # TODO: last pooling for actual wave
        with tf.variable_scope('g_wave_pooling'):
            skip_T = tf.stack(skips, axis=0)
            skips_sum = tf.reduce_sum(skip_T, axis=0)
            skips_sum = leakyrelu(skips_sum)
            wave_a = conv1d(skips_sum, kwidth=1, num_kernels=1,
                            init=tf.truncated_normal_initializer(stddev=0.02))
            wave = tf.tanh(wave_a)
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
        print('Last residual wave shape: ', res_i.get_shape())
        print('*************************')
        segan.generator_built = True
        return wave, z

class AEGenerator(object):

    def __init__(self, segan):
        self.segan = segan

    def __call__(self, noisy_w, is_ref, spk=None, z_on=True):
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

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True

        print('*** Building Generator ***')
        in_dims = noisy_w.get_shape().as_list()
        h_i = noisy_w
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 31
        enc_layers = 7
        skips = []
        with tf.variable_scope('g_ae'):
            #AE to be built is shaped:
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            #FIRST ENCODER
            for layer_idx, layer_depth in enumerate(segan.g_enc_depths):
                h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   name='enc_{}'.format(layer_idx))
                print('Downconv {} -> {}'.format(h_i.get_shape(),
                                                 h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(segan.g_enc_depths) - 1:
                    print('Adding skip connection downconv {}'.format(layer_idx))
                    # store skip connection
                    # last one is not stored cause it's the code
                    skips.append(h_i)
                h_i = leakyrelu(h_i)

            if z_on:
                # random code is fused with intermediate representation
                z = make_z([segan.batch_size, h_i.get_shape().as_list()[1],
                            segan.g_enc_depths[-1]])
                h_i = tf.concat(2, [z, h_i])

            #SECOND DECODER (reverse order)
            g_dec_depths = segan.g_enc_depths[:-1][::-1] + [1]
            print('g_dec_depths: ', g_dec_depths)
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                # deconv
                h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='dec_{}'.format(layer_idx))
                print('Deconv {} -> {}'.format(h_i.get_shape(),
                                               h_i_dcv.get_shape()))
                h_i = h_i_dcv
                if layer_idx < len(g_dec_depths) - 1:
                    h_i = leakyrelu(h_i)
                    # fuse skip connection
                    skip_ = skips[-(layer_idx + 1)]
                    print('Fusing skip connection of shape {}'.format(skip_.get_shape()))
                    h_i = tf.concat(2, [h_i, skip_])

                else:
                    h_i = tf.tanh(h_i)

            wave = h_i
            print('Amount of skip connections: ', len(skips))
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
            print('Last wave shape: ', wave.get_shape())
            print('*************************')
            segan.generator_built = True
            if z_on:
                return wave, z
            else:
                return wave

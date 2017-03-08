from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
import numpy as np


def discriminator(self, wave_in, spk=None, reuse=False):
        """
        w_ins: waveform input
        spk: input of spk ids? (TODO)
        """
        # take the waveform as input "activation"
        in_dims = wave_in.get_shape().as_list()
        hi = wave_in
        if len(in_dims) == 2:
            hi = tf.expand_dims(wave_in, -1)
        elif len(im_dims) < 2 or len(im_dims) > 3:
            raise ValueError('Discriminator input must be 2-D or 3-D')

        batch_size = int(wave_in.get_shape()[0])
        # assert batch_size == 3 * self.batch_size

        # set up the disc_block function
        with tf.variable_scope('d_model') as scope:
            if reuse:
                scope.reuse_variables()
            def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation,
                           pooling=2):
                with tf.variable_scope('d_block_{}'.format(block_idx)):
                    print('D block {} input shape: {}'
                          ''.format(block_idx, input_.get_shape()), end=' *** ')
                    downconv_init = tf.truncated_normal_initializer(stddev=0.02)
                    hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                    init=downconv_init)
                    print('downconved shape: {} '
                          ''.format(hi_a.get_shape()), end=' *** ')
                    if bnorm:
                        print('Applying VBN', end=' *** ')
                        hi_a = self.vbn(hi_a, 'd_vbn_{}'.format(block_idx))
                    if activation == 'leakyrelu':
                        print('Applying Lrelu', end=' *** ')
                        hi = leakyrelu(hi_a)
                    elif activation == 'relu':
                        print('Applying Relu', end=' *** ')
                        hi = tf.nn.relu(hi_a)
                    else:
                        raise ValueError('Unrecognized activation {} '
                                         'in D'.format(activation))
                    return hi
            beg_size = self.canvas_size
            # apply input noisy layer to real and fake samples
            hi = gaussian_noise_layer(hi, self.disc_noise_std)
            print('*** Discriminator summary ***')
            for block_idx, fmaps in enumerate(self.d_num_fmaps):
                hi = disc_block(block_idx, hi, 5,
                                self.d_num_fmaps[block_idx],
                                False, 'relu')
                print()
            print('discriminator deconved shape: ', hi.get_shape())
            #hi_f = tf.nn.dropout(hi_f, self.disc_keep_prob)
            d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='logits_conv')
            d_logit_out = tf.squeeze(d_logit_out)
            print('discriminator output shape: ', d_logit_out.get_shape())
            print('*****************************')
            return d_logit_out

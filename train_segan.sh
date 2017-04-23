#!/bin/bash

# Place the CUDA_VISIBLE_DEVICES="xxxx" required before the python call
# e.g. to specify the first two GPUs in your system: CUDA_VISIBLE_DEVICES="0,1" python ...

# SEGAN with no pre-emph and no bias in conv layers (just filters to downconv + deconv)
#CUDA_VISIBLE_DEVICES="2,3" python main.py --init_noise_std 0. --save_path segan_vanilla \
#                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
#                                          --save_freq 50 --epoch 50

# SEGAN with pre-emphasis to try to discriminate more high freq (better disc of high freqs)
#CUDA_VISIBLE_DEVICES="1,2,3" python main.py --init_noise_std 0. --save_path segan_preemph \
#                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
#                                          --save_freq 50 --preemph 0.95 --epoch 86

# Apply pre-emphasis AND apply biases to all conv layers (best SEGAN atm)
CUDA_VISIBLE_DEVICES="1,2,3" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph \
                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
                                          --save_freq 50 --preemph 0.95 --epoch 86 --bias_deconv True \
                                          --bias_downconv True --bias_D_conv True

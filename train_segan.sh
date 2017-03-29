#!/bin/bash

# Place the CUDA_VISIBLE_DEVICES="xxxx" required before the python call
# e.g. to specify the first two GPUs in your system: CUDA_VISIBLE_DEVICES="0,1" python ...

python main.py --init_noise_std 0. --save_path segan_v1 \
               --init_l1_weight 100. --batch_size 100 --g_nl prelu \
               --save_freq 50

#!/bin/bash

python main.py --init_noise_std 0. --save_path segan_checkpoint \
               --init_l1_weight 100. --batch_size 100 --g_nl prelu \
               --save_freq 50

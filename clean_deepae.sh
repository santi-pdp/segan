#!/bin/bash


# guia file containing pointers to files to clean up
if [ $# -lt 1 ]; then
    echo 'ERROR: guia file must be provided!'
    echo "Usage: $0 <guia_file>"
    exit 1
fi
GUIA="$1"
SAVE_PATH="test_clean_results"
mkdir -p $SAVE_PATH

cat $GUIA | while read noisy_name; do
     CUDA_VISIBLE_DEVICES="" python main.py --init_noise_std 0. --save_path segan_deeperae_gd31_gprelu_100l1_rmsprop_lrg00002_lrd0002_dclassifier_dlrelu_bnorm_1out \
                                            --batch_size 100 --g_nl prelu --weights WWaveGAN-41800 --test_wav $noisy_name --clean_save_path $SAVE_PATH
done

python main.py --init_noise_std 1. --save_path segan_v1 ^
               --batch_size 100 --g_nl prelu --weights SEGAN-72900 ^
               --preemph 0.95 --bias_deconv True ^
               --bias_downconv True --bias_D_conv True ^
               --test_wav %1 --save_clean_path %2

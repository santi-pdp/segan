# Speech Enhancement Generative Adversarial Network

This is the repository of the SEGAN project. A Generative Adversarial approach has been taken to do speech enhancement (i.e. removing noise from corrupted speech signal) with a fully convolutional architecture schematized as follows:

![SEGAN_G](assets/segan_g.png)

## Data

The speech enhancement dataset used in this work [(Valentini et al. 2016)](http://ssw9.net/papers/ssw9_PS2-4_Valentini-Botinhao.pdf) can be found in [Edinburgh DataShare](http://datashare.is.ed.ac.uk/handle/10283/1942). However, **the following script downloads and prepares the data for TF format**:

```
./prepare_data.sh
```

Or alternatively download the dataset, convert the wav files to 16kHz sampling and set the `noisy` and `clean` training files paths in the config file `e2e_maker.cfg` in `cfg/`. Then run the script:

```
python make_tfrecords.py --force-gen
```

### Training

Once you have the TFRecords file created in `data/segan.tfrecords` you can simply run the training process with:

```
./train_segan.sh
```

By default this will take all the available GPUs in your system, if any. Otherwise it will just take the CPU.

### Loading model and prediction

**Doc under construction [ ! ]**

## Authors

Santiago Pascual, Antonio Bonafonte, Joan Serrà

##### Contact mail
santi.pascual@upc.edu

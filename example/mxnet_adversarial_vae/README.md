VAE-GAN in MXNet

Implementation of Autoencoding beyond pixels using a learned similarity metric based on the Tensorflow implementation of https://github.com/JeremyCCHsu/tf-vaegan/

*Please refer to their official Github for details*: https://github.com/andersbll/autoencoding_beyond_pixels

As the name indicates, VAE-GAN replaces GAN's generator with a variational auto-encoder, resulting in a model with both inference and generation components. 

Experiements

Dataset: caltech 101 silhouettes dataset from https://people.cs.umass.edu/~marlin/data.shtml

Usage

Using existing models

python vaegan_mxnet.py --test --testing_data_path [your dataset image path] --pretrained_encoder_path [pretrained encoder model path] --pretrained_generator_path [pretrained generator model path] [options]

Train a new model

python vaegan_mxnet.py --train --training_data_path [your dataset image path] [options]


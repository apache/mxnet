<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# VAE-GAN in MXNet

* Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300),
  based on the Tensorflow implementation: <https://github.com/JeremyCCHsu/tf-vaegan>

* Please refer to their official Github for details: [Autoencoding Beyond Pixels](https://github.com/andersbll/autoencoding_beyond_pixels)

* As the name indicates, VAE-GAN replaces GAN's generator with a variational auto-encoder, resulting in a model with both inference and generation components. 

# Experiements

* Dataset: caltech 101 silhouettes dataset from <https://people.cs.umass.edu/~marlin/data.shtml>

# Prerequisites

* Opencv
* Python packages required: scipy, scikit-learn and Pillow, opencv python package

# Environment Tested On

Deep Learning AMI (Ubuntu) - 2.0, p2.8xlarge

# Usage

If you want to train and test with the default options do the following:

1. Download the default dataset and convert from matlab file format to png file format
```
python convert_data.py
```
2. Train on the downloaded dataset and store the encoder model and generator model params.
```
python vaegan_mxnet.py --train
```
3. Test on the downloaded dataset
```
python vaegan_mxnet.py --test --testing_data_path /home/ubuntu/datasets/caltech101/test_data
```

* Using existing models:

```
python vaegan_mxnet.py --test --testing_data_path [your dataset image path] --pretrained_encoder_path [pretrained encoder model path] --pretrained_generator_path [pretrained generator model path] [options]
```

* Train a new model:

```
python vaegan_mxnet.py --train --training_data_path [your dataset image path] [options]
```

* Training on the CPU:

```
python vaegan_mxnet.py --train --use_cpu --training_data_path [your dataset image path] [options]
```

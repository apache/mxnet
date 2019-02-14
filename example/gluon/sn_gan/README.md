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

# Spectral Normalization GAN

This example implements [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) based on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Usage

Example runs and the results:

```python
python train.py --use-gpu --data-path=data
```

* Note that the program would download the CIFAR10 for you

`python train.py --help` gives the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        path of data.
  --batch-size BATCH_SIZE
                        training batch size. default is 64.
  --epochs EPOCHS       number of training epochs. default is 100.
  --lr LR               learning rate. default is 0.0001.
  --lr-beta LR_BETA     learning rate for the beta in margin based loss.
                        default is 0.5.
  --use-gpu             use gpu for training.
  --clip_gr CLIP_GR     Clip the gradient by projecting onto the box. default
                        is 10.0.
  --z-dim Z_DIM         dimension of the latent z vector. default is 100.
```

## Result

![SN-GAN](sn_gan_output.png)

## Learned Spectral Normalization

![alt text](https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/assests/sn.png)

## Reference

[Simple Tensorflow Implementation](https://github.com/taki0112/Spectral_Normalization-Tensorflow)
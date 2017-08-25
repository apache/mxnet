# Accelerate Convolutional Neural Networks

This tool aims to accelerate the test-time computation and decrease number of parameters of deep CNNs.


## How to use

Use ``accnn.py`` to get a new model by specifying an original model and the speeding-up ratio.

You may provide a json to explicitly control the architecture of the new model, otherwise the rank-selection algorithm would be used to do it automatically and the configuration would be saved to file ``config.json``.

``acc_conv.py`` and ``acc_fc.py`` would be involved automatically when using ``accnn.py`` while ``acc_conv.py`` and ``acc_fc.py`` can also be used seperately.

## Example

###Speedup whole network

- Speed up a model by 2 times and use ``rank-selection`` to determine ranks of each layer automatically

  ```bash
  python accnn.py -m MODEL-PREFIX --save-model new-vgg16 --ratio 2
  ```

- Use your own configuration file without ``rank-selection``

  ```bash
  python accnn.py -m MODEL-PREFIX --save-model new-model --config YOUR-CONFIG_JSON
  ```

###Speedup a single layer
  
- Decompose a convolutional layer:

  ```bash
  python acc_conv.py -m MODEL-PREFIX --layer LAYER-NAME --K NUM-FILTER --save-model new-model
  ```

- Decompose a fullyconnected layer:

  ```bash
  python acc_fc.py -m MODEL-PREFIX --layer LAYER-NAME --K NUM-HIDDEN --save-model new-model
  ```
- uses `--help` to see more options


## Results

The experiments are carried on a single machine with four Nvidia Titan X GPUs. The top-5 accuracy is evaluated on ImageNet validation dataset.



| Model | Top-5 accuracy  |  Theoretical speed up | CPU speed up | GPU speed up |
| ------------- | -----------: | -------------: | -----------: | -----------: |
| model0 | 89.6% |  1x|  1x|  1x|
| model1 | 88.6% |  2.4x|   2.2x|  1.1x|
| model2 | 89.8% |  2.4x|   2.2x|  1.1x|
| model3 | 87.5% |  3x|   2.6x|    1.2x|
| model4 | 89.6% |  3x|   2.6x|    1.2x|


 * ``model0`` is the original VGG16 model directly converted from Caffe Model Zoo
 * ``model1`` is the accelerated model based on ``config.json``
 * ``model2`` is the same as ``model1`` but is fine-tuned on ImageNet training dataset for 5 epochs
 * ``model3`` is the accelerated model based on rank-selection with 3 times speeding up
 * ``model4`` is the same as ``model3`` but is fine-tuned on ImageNet training dataset for 5 epochs
 * The experiments in GPU are carried with cuDNN 4
 
 
## Notes

* This tool is verified on the [VGG-16](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395) model converted from Caffe by ``caffe_converter`` tool.

* ``accnn.py`` tool only supports single input and output

* This tool mainly implements the algorithm of Cheng *et al.* [2] to decompose a convolutional layer to two convolutional layers both in spatial dimensions and across channels. ``acc_conv.py`` provides the function to replace a ``(N,d,d)`` conv. layer by two ``(K,d,1)`` and ``(N,1,d)`` conv. layers.

* The idea of ``rank-selection`` tool is based on the related work of Zhang *et al* [1] that we could use the product of PCA energy to determine the rank for each layer.

## Reference Paper

[1] Zhang, Xiangyu, et al. "Efficient and accurate approximations of nonlinear convolutional networks." arXiv preprint arXiv:1411.4229 (2014).

[2] Tai, Cheng, et al. "Convolutional neural networks with low-rank regularization." arXiv preprint arXiv:1511.06067 (2015).

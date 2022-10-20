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


# Improving accuracy with Intel® Neural Compressor

The accuracy of a model can decrease as a result of quantization. When the accuracy drop is significant, we can try to manually find a better quantization configuration (exclude some layers, try different calibration methods, etc.), but for bigger models this might prove to be a difficult and time consuming task. [Intel® Neural Compressor](https://github.com/intel/neural-compressor) (INC) tries to automate this process using several tuning heuristics, which aim to find the quantization configuration that satisfies the specified accuracy requirement.

**NOTE:**

Most tuning strategies will try different configurations on an evaluation dataset in order to find out how each layer affects the accuracy of the model. This means that for larger models, it may take a long time to find a solution (as the tuning space is usually larger and the evaluation itself takes longer).

## Installation and Prerequisites

- Install MXNet with oneDNN enabled as described in the [Get started](https://mxnet.apache.org/versions/master/get_started?platform=linux&language=python&processor=cpu&environ=pip&). (Until the 2.0 release you can use the nightly build version: `pip install --pre mxnet -f https://dist.mxnet.io/python`)

- Install Intel® Neural Compressor:

  Use one of the commands below to install INC (supported python versions are: 3.6, 3.7, 3.8, 3.9):

  ```bash
  # install stable version from pip
  pip install neural-compressor

  # install nightly version from pip
  pip install -i https://test.pypi.org/simple/ neural-compressor

  # install stable version from conda
  conda install neural-compressor -c conda-forge -c intel
  ```
  If you get into trouble with dependencies on `cv2` library you can run: `apt-get update && apt-get install -y python3-opencv`

## Configuration file

Quantization tuning process can be customized in the yaml configuration file. Below is a simple example:

```yaml
# cnn.yaml

version: 1.0

model:
  name: cnn
  framework: mxnet

quantization:
  calibration:
    sampling_size: 160 # number of samples for calibration

tuning:
  strategy:
    name: basic
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

We are using the `basic` strategy, but you could also try out different ones. [Here](https://github.com/intel/neural-compressor/blob/master/docs/tuning_strategies.md) you can find a list of strategies available in INC and details of how they work. You can also add your own strategy if the existing ones do not suit your needs.

Since the value of `timeout` in the example above is 0, INC will run until it finds a configuration that satisfies the accuracy criterion and then exit. Depending on the strategy this may not be ideal, as sometimes it would be better to further explore the tuning space to find a superior configuration both in terms of accuracy and speed. To achieve this, we can set a specific `timeout` value, which will tell INC how long (in seconds) it should run.

For more information about the configuration file, see the [template](https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/ptq.yaml) from the official INC repo. Keep in mind that only the `post training quantization` is currently supported for MXNet.

## Model quantization and tuning

In general, Intel® Neural Compressor requires 4 elements in order to run:  
1. Configuration file - like the example above  
2. Model to be quantized  
3. Calibration dataloader  
4. Evaluation function - a function that takes a model as an argument and returns the accuracy it achieves on a certain evaluation dataset. 

### Quantizing ResNet

The [quantization](https://mxnet.apache.org/versions/master/api/python/docs/tutorials/performance/backend/dnnl/dnnl_quantization.html#Quantization) sections described how to quantize ResNet using the native MXNet quantization. This example shows how we can achieve the similar results (with the auto-tuning) using INC.

1. Get the model

```python
import logging
import mxnet as mx
from mxnet.gluon.model_zoo import vision

logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

batch_shape = (1, 3, 224, 224)
resnet18 = vision.resnet18_v1(pretrained=True)
```

2. Prepare the dataset:

```python
mx.test_utils.download('http://data.mxnet.io/data/val_256_q90.rec', 'data/val_256_q90.rec')

batch_size = 16
mean_std = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939,
            'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}

data = mx.io.ImageRecordIter(path_imgrec='data/val_256_q90.rec',
                             batch_size=batch_size,
                             data_shape=batch_shape[1:],
                             rand_crop=False,
                             rand_mirror=False,
                             shuffle=False,
                             **mean_std)
data.batch_size = batch_size
```

3. Prepare the evaluation function:

```python
eval_samples = batch_size*10

def eval_func(model):
    data.reset()
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(data):
        if i * batch_size >= eval_samples:
            break
        x = batch.data[0].as_in_context(mx.cpu())
        label = batch.label[0].as_in_context(mx.cpu())
        outputs = model.forward(x)
        metric.update(label, outputs)
    return metric.get()[1]
```

4. Run Intel® Neural Compressor:

```python
from neural_compressor.experimental import Quantization
quantizer = Quantization("./cnn.yaml")
quantizer.model = resnet18
quantizer.calib_dataloader = data
quantizer.eval_func = eval_func
qnet = quantizer.fit().model
```

Since this model already achieves good accuracy using native quantization (less than 1% accuracy drop), for the given configuration file, INC will end on the first configuration, quantizing all layers using `naive` calibration mode for each. To see the true potential of INC, we need a model which suffers from a larger accuracy drop after quantization.

### Quantizing ResNet50v2

This example shows how to use INC to quantize ResNet50 v2. In this case, the native MXNet quantization introduce a huge accuracy drop (70% using `naive` calibration mode) and INC allows to automatically find better solution.

This is the [INC configuration file](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/resnet50v2_mse.yaml) for this example: 
```yaml
version: 1.0

model:
  name: resnet50_v2
  framework: mxnet

quantization:
  calibration:
    sampling_size: 192 # number of samples for calibration

tuning:
  strategy:
    name: mse
  accuracy_criterion:
    relative: 0.015
  exit_policy:
    timeout: 0
    max_trials: 500
  random_seed: 9527
```

It could be used with script below 
([resnet_mse.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/resnet_mse.py))
to find operator, which caused the most significant accuracy drop and disable it from quantization. 
You can find description of MSE strategy 
[here](https://github.com/intel/neural-compressor/blob/master/docs/tuning_strategies.md#user-content-mse).

```python
import mxnet as mx
from mxnet.gluon.model_zoo.vision import resnet50_v2
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import quantize_net

# Preparing input data
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)
batch_size = 64
num_calib_batches = 9
# set proper path to ImageNet data set below
dataset = mx.gluon.data.vision.ImageRecordDataset('../imagenet/rec/val.rec')
# Tuning with INC on whole data set takes a lot of time. Therefore, we take only a part of the data set
# as representative part of it:
dataset = dataset.take(num_calib_batches * batch_size)
transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=rgb_mean, std=rgb_std)])
# Note: as input data is used many times during tuning, it is better to have it prepared earlier.
#       Therefore, lazy parameter for transform_first is set to False.
val_data = mx.gluon.data.DataLoader(
    dataset.transform_first(transformer, lazy=False), batch_size, shuffle=False)
val_data.batch_size = batch_size

net = resnet50_v2(pretrained=True)

def eval_func(model):
  metric = mx.gluon.metric.Accuracy()
  for x, label in val_data:
    output = model(x)
    metric.update(label, output)
  accuracy = metric.get()[1]
  return accuracy


from neural_compressor.experimental import Quantization
quantizer = Quantization("resnet50v2_mse.yaml")
quantizer.model = net
quantizer.calib_dataloader = val_data
quantizer.eval_func = eval_func
qnet_inc = quantizer.fit().model
print("INC finished")
# You can save optimized model for the later use:
qnet_inc.export("__quantized_with_inc")
# You can see which configuration was applied by INC and which nodes were excluded from quantization,
# to achieve given accuracy loss against floating point calculation.
print(quantizer.strategy.best_qmodel.q_config['quant_cfg'])
```

#### Results:
Resnet50 v2 model could be prepared to achieve better performance with various calibration and tuning methods.  
It is done by 
[resnet_tuning.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/resnet_tuning.py) 
script on a small part of data set to reduce time required for tuning (9 batches). 
Later saved models are validated on a whole data set by 
[resnet_measurement.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/resnet_measurement.py)
script.
Accuracy results on the whole validation dataset (782 batches) are shown below.

| Optimization method  | Top 1 accuracy | Top 5 accuracy | Top 1 relative accuracy loss [%] | Top 5 relative accuracy loss [%] | Cost = one-time optimization on 9 batches [s] | Validation time [s] | Speedup |
|----------------------|-------:|-------:|------:|------:|-------:|--------:|------:|
| fp32 no optimization | 0.7699 | 0.9340 |  0.00 |  0.00 |   0.00 | 316.50 | 1.0 |
| fp32 fused           | 0.7699 | 0.9340 |  0.00 |  0.00 |   0.03 | 147.77 | 2.1 |
| int8 full naive      | 0.2207 | 0.3912 | 71.33 | 58.12 |  11.29 |  45.81 | **6.9** |
| int8 full entropy    | 0.6933 | 0.8917 |  9.95 |  4.53 |  80.23 |  46.39 | 6.8 |
| int8 smart naive     | 0.2210 | 0.3905 | 71.29 | 58.19 |  11.15 |  46.02 | 6.9 |
| int8 smart entropy   | 0.6928 | 0.8910 | 10.01 |  4.60 |  79.75 |  45.98 | 6.9 |
| int8 INC basic       | 0.7692 | 0.9331 | **0.09** |  0.10 | 266.50 |  48.32 | **6.6** |
| int8 INC mse         | 0.7692 | 0.9337 | **0.09** |  0.03 | 106.50 |  49.76 | **6.4** |
| int8 INC mycustom    | 0.7699 | 0.9338 | **0.00** |  0.02 | 370.29 |  70.07 | **4.5** |


Environment:  
- Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz (c6i.16xlarge Amazon EC2 instance)  
- Ubuntu 20.04.4 LTS (GNU/Linux Ubuntu 20.04.4 LTS 5.15.0-1017-aws ami-0558cee5b20db1f9c)  
- MXNet 2.0.0b20220902 (commit 3a19f0e50d75fedb05eb558a9c835726b57df4cf)  
- INC 1.13.1  
- scripts above were run as parameter for [run.sh](https://github.com/apache/incubator-mxnet/blob/master/benchmark/python/dnnl/run.sh) 
script to properly setup parallel computation parameters.  

For this model INC `basic`, `mse` and `mycustom` strategies found configurations meeting the 1.5% relative accuracy 
loss criterion. Only the `bayesian` strategy didn't find solution within 500 attempts limit. 
Although these results may suggest that the `mse` strategy is the best compromise between time spent
to find the optimized model and final model performance efficiency, different strategies may give 
better results for specific models and tasks. For example for ALBERT model there is no solution 
given by build-in INC strategies. For such situation you can create your custom strategy, similar 
to this one: 
[custom_strategy.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/custom_strategy.py). 
You can notice, that the most important thing done by INC
was to find the operator, which had the most significant impact on the loss of accuracy and disable it from quantization if needed. 
You can see below which operator was excluded by `mse` strategy in last print given by 
[resnet_mse.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization_inc/resnet_mse.py) 
script:  

{'excluded_symbols': ['**sg_onednn_conv_bn_act_0**'], 'quantized_dtype': 'auto', 'quantize_mode': 'smart', 'quantize_granularity': 'tensor-wise'}


## Tips
- In order to get a solution that generalizes well, evaluate the model (in eval_func) on a representative dataset.
- With `history.snapshot` file (generated by INC) you can recover any model that was generated during the tuning process:
  ```python
  from neural_compressor.utils.utility import recover

  quantized_model = recover(f32_model, 'nc_workspace/<tuning date>/history.snapshot', configuration_idx).model
  ```

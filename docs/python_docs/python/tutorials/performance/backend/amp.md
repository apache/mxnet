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

# Using AMP: Automatic Mixed Precision

Training Deep Learning networks is a very computationally intensive task. Novel model architectures tend to have increasing number of layers and parameters, which slows down training. Fortunately, new generations of training hardware as well as software optimizations, make it a feasible task.

However, where most of the (both hardware and software) optimization opportunities exists is in exploiting lower precision (like FP16) to, for example, utilize Tensor Cores available on new Volta and Turing GPUs. While training in FP16 showed great success in image classification tasks, other more complicated neural networks typically stayed in FP32 due to difficulties in applying the FP16 training guidelines.

That is where AMP (Automatic Mixed Precision) comes into play. It automatically applies the guidelines of FP16 training, using FP16 precision where it provides the most benefit, while conservatively keeping in full FP32 precision operations unsafe to do in FP16.

This tutorial shows how to get started with mixed precision training using AMP for MXNet. As an example of a network we will use SSD network from GluonCV.

## Data loader and helper functions

For demonstration purposes we will use synthetic data loader.


```python
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
from mxnet import autograd
from mxnet.test_utils import download_model
import gluoncv as gcv
from gluoncv.model_zoo import get_model

data_shape = 512
batch_size = 8
lr = 0.001
wd = 0.0005
momentum = 0.9

# training contexts
ctx = [mx.gpu(0)]

# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')
```


```python
class SyntheticDataLoader(object):
    def __init__(self, data_shape, batch_size):
        super(SyntheticDataLoader, self).__init__()
        self.counter = 0
        self.epoch_size = 200
        shape = (batch_size, 3, data_shape, data_shape)
        cls_targets_shape = (batch_size, 6132)
        box_targets_shape = (batch_size, 6132, 4)
        self.data = mx.nd.random.uniform(-1, 1, shape=shape, ctx=mx.cpu_pinned())
        self.cls_targets = mx.nd.random.uniform(0, 1, shape=cls_targets_shape, ctx=mx.cpu_pinned())
        self.box_targets = mx.nd.random.uniform(0, 1, shape=box_targets_shape, ctx=mx.cpu_pinned())
    
    def next(self):
        if self.counter >= self.epoch_size:
            self.counter = self.counter % self.epoch_size
            raise StopIteration
        self.counter += 1
        return [self.data, self.cls_targets, self.box_targets]
    
    __next__ = next
    
    def __iter__(self):
        return self
    
train_data = SyntheticDataLoader(data_shape, batch_size)
```


```python
def get_network():
    # SSD with RN50 backbone
    net_name = 'ssd_512_resnet50_v1_coco'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.nn.BatchNorm)
        net.initialize()
        net.collect_params().reset_ctx(ctx)

    return net
```

## Training in FP32

First, let us create the network.


```python
net = get_network()
net.hybridize(static_alloc=True, static_shape=True)
```



Next, we need to create a Gluon Trainer.


```python
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': lr, 'wd': wd, 'momentum': momentum})
```


```python
mbox_loss = gcv.loss.SSDMultiBoxLoss()

for epoch in range(1):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()

    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        if not (i + 1) % 50:
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()
```

output 

```text
INFO:root:[Epoch 0][Batch 49], Speed: 58.105 samples/sec, CrossEntropy=1.190, SmoothL1=0.688
INFO:root:[Epoch 0][Batch 99], Speed: 58.683 samples/sec, CrossEntropy=0.693, SmoothL1=0.536
INFO:root:[Epoch 0][Batch 149], Speed: 58.915 samples/sec, CrossEntropy=0.500, SmoothL1=0.453
INFO:root:[Epoch 0][Batch 199], Speed: 58.422 samples/sec, CrossEntropy=0.396, SmoothL1=0.399
```

## Training with AMP

### AMP initialization

In order to start using AMP, we need to import and initialize it. This has to happen before we create the network.


```python
from mxnet.contrib import amp

amp.init()
```
output:
```text
INFO:root:Using AMP
```


After that, we can create the network exactly the same way we did in FP32 training.


```python
net = get_network()
net.hybridize(static_alloc=True, static_shape=True)
```

For some models that may be enough to start training in mixed precision, but the full FP16 recipe recommends using dynamic loss scaling to guard against over- and underflows of FP16 values. Therefore, as a next step, we create a trainer and initialize it with support for AMP's dynamic loss scaling. Currently, support for dynamic loss scaling is limited to trainers created with `update_on_kvstore=False` option, and so we add it to our trainer initialization.


```python
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': lr, 'wd': wd, 'momentum': momentum},
    update_on_kvstore=False)

amp.init_trainer(trainer)
```

### Dynamic loss scaling in the training loop

The last step is to apply the dynamic loss scaling during the training loop and . We can achieve that using the `amp.scale_loss` function.


```python
mbox_loss = gcv.loss.SSDMultiBoxLoss()

for epoch in range(1):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()

    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                autograd.backward(scaled_loss)
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        if not (i + 1) % 50:
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()
```

output

```bash
INFO:root:[Epoch 0][Batch 49], Speed: 93.585 samples/sec, CrossEntropy=1.166, SmoothL1=0.684
INFO:root:[Epoch 0][Batch 99], Speed: 93.773 samples/sec, CrossEntropy=0.682, SmoothL1=0.533
INFO:root:[Epoch 0][Batch 149], Speed: 93.399 samples/sec, CrossEntropy=0.493, SmoothL1=0.451
INFO:root:[Epoch 0][Batch 199], Speed: 93.674 samples/sec, CrossEntropy=0.391, SmoothL1=0.397
```

We got 60% speed increase from 3 additional lines of code!

## Inference with AMP

To do inference with mixed precision for a trained model in FP32, you can use the conversion APIs: `amp.convert_model` for symbolic model and `amp.convert_hybrid_block` for gluon models. The conversion APIs will take the FP32 model as input and will return a mixed precision model, which can be used to run inference.
Below, we demonstrate for a gluon model and a symbolic model:
- Conversion from FP32 model to mixed precision model.
- Run inference on the mixed precision model.
- For AMP conversion of bucketing module please refer to [example/rnn/bucketing/README.md](https://github.com/apache/incubator-mxnet/blob/master/example/rnn/bucketing/README.md).

```python
with mx.Context(mx.gpu(0)):
    # Below is an example of converting a gluon hybrid block to a mixed precision block
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        model = get_model("resnet50_v1")
        model.collect_params().initialize(ctx=mx.current_context())
        model.hybridize()
        model(mx.nd.zeros((1, 3, 224, 224)))
        converted_model = amp.convert_hybrid_block(model)

    # Run dummy inference with the converted gluon model
    result = converted_model.forward(mx.nd.random.uniform(shape=(1, 3, 224, 224),
                                                          dtype=np.float32))

    # Below is an example of converting a symbolic model to a mixed precision model
    model_path = "model"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    prefix, epoch = mx.test_utils.download_model("imagenet1k-resnet-18", dst_dir=model_path)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                         arg_params,
                                                                         aux_params)

    # Run dummy inference with the converted symbolic model
    mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.current_context())
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])
    mod.set_params(result_arg_params, result_aux_params)
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                                label=[mx.nd.ones((1,))]))
    mod.get_outputs()[0].wait_to_read()
    print("Conversion and Inference completed successfully")
```

You can also customize the operators to run in FP16 versus the operator to run in FP32 or to conditionally run in FP32.
Also, you can force cast the params wherever possible to FP16. Below is an example which demonstrates both these use cases
for symbolic model. You can do the same for gluon hybrid block with `amp.convert_hybrid_block` API, `cast_optional_params` flag.

```python
with mx.Context(mx.gpu(0)):
    # Below is an example of converting a symbolic model to a mixed precision model
    # with only Convolution op being force casted to FP16.
    model_path = "model"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    prefix, epoch = mx.test_utils.download_model("imagenet1k-resnet-18", dst_dir=model_path)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # All Convolution ops should run in FP16, SoftmaxOutput and FullyConnected should run in FP32
    # cast_optional_params=True: Force cast params to FP16 wherever possible
    result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                         arg_params,
                                                                         aux_params,
                                                                         target_dtype_ops=["Convolution"],
                                                                         fp32_ops=["SoftmaxOutput", "FullyConnected"],
                                                                         cast_optional_params=True)

    # Run dummy inference with the converted symbolic model
    mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.current_context())
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])
    mod.set_params(result_arg_params, result_aux_params)
    mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                                label=[mx.nd.ones((1,))]))
    mod.get_outputs()[0].wait_to_read()

    # Assert that the params for conv are in FP16, this is because cast_optional_params is set to True
    assert mod._arg_params["conv0_weight"].dtype == np.float16
    # FullyConnected params stay in FP32
    assert mod._arg_params["fc1_bias"].dtype == np.float32

    print("Conversion and Inference completed successfully")

    # Serialize AMP model and save to disk
    mod.save_checkpoint("amp_tutorial_model", 0, remove_amp_cast=False)
```

## Current limitations of AMP

- AMP's dynamic loss scaling currently supports only Gluon trainer with `update_on_kvstore=False` option set
- Using `SoftmaxOutput`, `LinearRegressionOutput`, `LogisticRegressionOutput`, `MAERegressionOutput` with dynamic loss scaling does not work when training networks with multiple Gluon trainers and so multiple loss scales
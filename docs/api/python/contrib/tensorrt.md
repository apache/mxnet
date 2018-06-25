# MXNet-TensorRT Runtime Integration
## What is this?

This document described how to use the [MXNet](http://mxnet.incubator.apache.org/)-[TensorRT](https://developer.nvidia.com/tensorrt) runtime integration to accelerate model inference.

## Why is TensorRT integration useful? 

TensorRT can greatly speed up inference of deep learning models. One experiment on a Titan V (V100) GPU shows that with MXNet 1.2, we can get an approximately 3x speed-up when running inference of the ResNet-50 model on the CIFAR-10 dataset in single precision (fp32). As batch sizes and image sizes go up (for CNN inference), the benefit may be less, but in general, TensorRT helps especially in cases which have:
- Many bandwidth-bound layers (e.g. pointwise operations) that benefit from GPU kernel fusion.
- Inference use cases which have tight latency requirements and where the client application can't wait for large batches to be queued up.
- Embedded systems, where memory constraints are tighter than on servers.
- When performing inference in reduced precision, especially for integer (e.g. int8) inference. 

In the past, the main hindrance for the user wishing to benefit from TensorRT was the fact that the model needed to be exported from the framework first. Once the model got exported through some means (NNVM to TensorRT graph rewrite, via ONNX, etc.), one had to then write a TensorRT client application, which would feed the data into the TensorRT engine. Since at that point the model was independent of the original framework, and since TensorRT could only compute the neural network layers but the user had to bring their own data pipeline, this increased the burden on the user and reduced the likelihood of reproducibility (e.g. different frameworks may have slightly different data pipelines, or flexibility of data pipeline operation ordering). Moreover, since frameworks typically support more operators than TensorRT, one could have to resort to TensorRT plugins for operations that aren't already available via the TensorRT graph API.  

The current experimental runtime integration of TensorRT with MXNet resolves the above concerns by ensuring that:
- The graph is still executed by MXNet.
- The MXNet data pipeline is preserved.
- The TensorRT runtime integration logic partitions the graph into subgraphs that are either TensorRT compatible or incompatible.
- The graph partitioner collects the TensorRT-compatible subgraphs, hands them over to TensorRT, and substitutes the TensorRT compatible subgraph with a TensorRT library call, represented as a TensorRT node in NNVM.
- If a node is not TensorRT compatible, it won't be extracted and substituted with a TensorRT call, and will still execute within MXNet.

The above points ensure that we find a compromise between the flexibility of MXNet, and fast inference in TensorRT.  We do this with no additional burden to the user.  Users do not need to learn how TensorRT APIs work, and do not need to write their own client application or data pipeline.

## How do I build MXNet with TensorRT integration?

Building MXNet together with TensorRT is somewhat complex. The recipe will hopefully be simplified in the near future, but for now, it's easiest to build a Docker container with a Ubuntu 16.04 base. This Dockerfile can be found under the ci subdirectory of the MXNet repository. You can build the container as follows:

```
docker build -t ci/docker/Dockerfile.build.ubuntu_gpu_tensorrt mxnet_with_tensorrt
```

Next, we can run this container as follows (don't forget to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)):

```no-highlight
nvidia-docker run -ti --rm mxnet_with_tensorrt
```

After starting the container, you will find yourself in the /opt/mxnet directory by default.

## Running a "hello, world" model / unit test:

You can then run the LeNet-5 unit test, which will train LeNet-5 on MNIST.  The test will then run inference in MXNet both with, and without MXNet-TensorRT runtime integration.  Finally, the test will display a comparison of both runtime's accuracy scores. The test can be run as follows:

```no-highlight
python tests/python/tensorrt/test_tensorrt_lenet5.py
```

You should get a result similar to the following:

```no-highlight
Running inference in MXNet
[03:31:18] src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:107: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
Running inference in MXNet-TensorRT
[03:31:18] src/operator/contrib/nnvm_to_onnx.cc:152: ONNX graph construction complete.
Building TensorRT engine, FP16 available:1
    Max batch size:     1024
    Max workspace size: 1024 MiB
[03:31:18] src/operator/contrib/tensorrt.cc:85: TensorRT engine instantiated!!!
MXNet accuracy: 98.680000
MXNet-TensorRT accuracy: 98.680000
```

## Running a more complex model

To show that the runtime integration handles more complex models such as ResNet-50 (which includes batch normalization as well as skip connections), the relevant script is included in the `example/image_classification/tensorrt` directory.

## Building your own models

When building your own models, feel free to use the above ResNet-50 model as an example. Here, we highlight a small number of issues that need to be taken into account.

1. When loading a pre-trained model, the inference will be handled using the Symbol API, rather than the Module API.
2. In order to provide the weights from MXNet (NNVM) to the TensorRT graph converter before the symbol is fully bound (before the memory is allocated, etc.), the `arg_params` and `aux_params` need to be provided to the symbol's `simple_bind` method. The weights and other values (e.g. moments learned from data by batch normalization, provided via `aux_params`) will be provided via the `shared_buffer` argument to `simple_bind` as follows:
```python
executor = sym.simple_bind(ctx=ctx, data = data_shape,
    softmax_label=sm_shape, grad_req='null', shared_buffer=all_params, force_rebind=True)
```
3. To collect `arg_params` and `aux_params` from the dictionaries loaded by `model.load()`, we need to combine them into one dictionary:
```python
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)

all_params = merge_dicts(arg_params, aux_params)
```
This `all_params` dictionary cn be seem in use in the `simple_bind` call in `#2`.
4. Once the symbol is bound, we need to feed the data and run the `forward()` method. Let's say we're using a test set data iterator called `test_iter`. We can run inference as follows:
```python
for idx, dbatch in enumerate(test_iter):
    data = dbatch.data[0]
    executor.arg_dict["data"][:] = data
    executor.forward(is_train=False)
    preds = executor.outputs[0].asnumpy() 
    top1 = np.argmax(preds, axis=1)
```
5. **Note:** One can choose between running inference with and without TensorRT. This can be selected by changing the state of the `MXNET_USE_TENSORRT` environment variable. Let's first write a convenience function to change the state of this environment variable:
```python
def set_use_tensorrt(status = False):
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))
```
Now, assuming that the logic to bind a symbol and run inference in batches of `batch_size` on dataset `dataset` is wrapped in the `run_inference` function, we can do the following:
```python
print("Running inference in MXNet")
set_use_tensorrt(False)
mx_pct = run_inference(sym, arg_params, aux_params, mnist,
                       all_test_labels, batch_size=batch_size)

print("Running inference in MXNet-TensorRT")
set_use_tensorrt(True)
trt_pct = run_inference(sym, arg_params, aux_params, mnist,
                        all_test_labels,  batch_size=batch_size)
```
Simply switching the flag allows us to go back and forth between MXNet and MXNet-TensorRT inference. See the details in the unit test at `tests/python/tensorrt/test_tensorrt_lenet5.py`.

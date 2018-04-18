# Gradient Compression

Gradient Compression reduces communication bandwidth, and in some scenarios, it can make training more scalable and efficient without significant loss in convergence rate or accuracy. Example implementations with GPUs, CPUs, and distributed training are provided in this document. 


## Benefits

**Increased Speed**

For architectures with fully connected layers, the gradient compression capability is observed to speedup training by about 2x, depending on the size of the model and the network bandwidth of the instance. Bigger models see larger speedup with gradient compression.

**Minimal Accuracy Loss**

Gradient compression uses the approach of delaying the synchronization of weight updates which are small. Although small weight updates might not be sent for that batch, this information is not discarded. Once the weight updates for this location accumulate to become a larger value, they will be propagated. Since there is no information loss, but only delayed updates, it does not lead to a significant loss in accuracy or convergence rate. In distributed training experiments[1], the accuracy loss observed due to gradient compression was as low as 1%


## When to Use Gradient Compression

When training models whose architectures include large fully connected components, it can be helpful to use gradient compression. For larger models, as well as recurrent neural networks, the communication cost becomes a major factor. Such models stand to benefit greatly with gradient compression.


### GPU versus CPU

The greatest benefits from gradient compression are realized when using multi-node (single or multi-GPU) distributed training. Training on CPU would provide a lower compute density per compute node as compared to the massive compute density per compute node on a GPU. Due to this, the required communication bandwidth for CPU-based nodes during training is not as high as for GPU-based nodes. Hence, the benefits of gradient compression are lower for CPU-based nodes as compared to GPU-based nodes.


### Network Latency

Benefits of gradient compression can be found when using distributed training with network connected nodes. Depending on the network latency between nodes and the model's size, these can contribute to slow performance such that gradient compression may provide speed improvements.

You may not want to use gradient compression if you have low latency network communication.


### Model Size

Distributed training involves synchronization of weights after each batch. Larger models have much higher communication costs during training, hence such models stand to benefit much more from gradient compression.
When running distributed training with gradient compression, the quantize and dequantize operations happen on CPU parallelized with OpenMP. For smaller models, when training on GPUs, it helps to set `OMP_NUM_THREADS=1` on each node, so that the overhead of launching OMP threads doesn't cause the compression and decompression to be slow.

### Model Architecture

The communication bandwidth requirements during training vary across various neural network architectures and hence the benefits of gradient compression vary accordingly.

In networks which have significant fully connected components, since such layers have low compute cost on GPUs, communication becomes a bottleneck limiting the speed of distributed training. Gradient compression can help reduce the communication cost, and thus speed up training in such cases. We have observed speedup of about 2x on large fully connected neural networks. Models like AlexNet and VGG have large fully connected components as part of the network, hence stand to benefit from gradient compression. As with these models, Long Short-Term Memory architectures require more communication bandwidth, so they also exhibit speed improvements with gradient compression.

Architectures like Convolutional Neural Networks on the other hand have a higher compute cost, in which case some communication can be parallelized with computation. Since communication is not the bottleneck in such networks, gradient compression doesn't help much.


### Single Node Gradient Compression

When the training is configured to use device to device communication on a single node with multiple GPUs, gradient compression can be used to reduce the cost of communication. This can provide about 20% speedup for large models using older generation architectures. However, speed benefits may be negligible on a machine with a newer generation architecture where GPUs can communicate at low latency.


## Approach

The idea behind gradient compression comes from two observations:

First, when training large neural networks, the gradients of weights computed for a small mini-batch of training data are typically sparse. Only a small fraction of the weights have significant updates after each mini-batch. The synchronization of updates that are near zero can be safely delayed longer than the typical mini-batch size. This essentially means that the rate of weight-update can vary depending on the value of an individual weight.

Secondly, gradients can be compressed significantly by considering only those gradient elements whose absolute values exceed a threshold, and then quantizing them to use lower bits per gradient value. By compressing the gradients, we can reduce communication bandwidth. The delayed gradient values, in the form of quantization error and values that don't meet the threshold, are aggregated into a gradient residual which is communicated when it reaches the threshold.

## Technical Implementation

### Two Bit Quantization

Currently the supported type of quantization uses two bits for each gradient value. Any positive value greater than or equal to the threshold sets two bits as `11`, any negative value whose absolute value is greater or equal to the threshold sets two bits as `10`, and others are set to `00`. This enables us to store 16 quantized gradients as one float. The error in quantization, which is `original_value - quantized_value` is stored in the form of a gradient residual.

### Types of Kvstore

Supported types of `kvstore` are `device` and all distributed kvstores such as `dist_sync`, `dist_async`, and `dist_sync_device`. When `kvstore` is `device`, the communication between GPUs is compressed. Please note that this increases the memory usage of GPUs because of the additional residual stored. When using a distributed kvstore, worker-to-server communication is compressed. In this case, compression and decompression happen on the CPU, and gradient residuals will be stored on the CPU. Server-to-worker communication and device-to-device communication are not compressed to avoid multiple levels of compression.

## Enabling the Gradient Compression in MXNet

Gradient compression is a run-time configuration parameter to be enabled during training. Here are the MXNet APIs to enable gradient compression:

**Gluon API**:

```
trainer = gluon.Trainer(..., compression_params={'type’:'2bit', 'threshold':0.5})
```
A reference `gluon` implementation with a gradient compression option can be found in the [train.py script from a word-level language modeling RNN example](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/word_language_model/train.py).

**Module API**:

```
mod = mx.mod.Module(..., compression_params={'type’:'2bit', 'threshold':0.5})
```

A `module` example is provided with [this guide for setting up MXNet with distributed training](https://mxnet.incubator.apache.org/versions/master/faq/multi_devices.html#distributed-training-with-multiple-machines). It comes with the option of turning on gradient compression as an argument to the [train_mnist.py script](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/train_mnist.py).

### Configuration Details

**Threshold**

A default `threshold` value of `0.5` is good for most use cases, but to get the most benefit from gradient compression for a particular scenario, it can be beneficial to experiment. If the threshold is set to a very large value, say `10.0`, then the updates become too infrequent and the training will converge slower. Setting the threshold automatically is expected in a future release.

**Quantization**

This release supports 2-bit quantization for encoding of gradients to reduce the communication bandwidth during training. Future releases will support 1-bit quantization and other approaches for encoding of gradients based on experimental evidence of benefits and user demand.

**Sparse Format**

We believe that the density of data will need to be really low (i.e. around > 90% zeros) to reap benefits of the sparse format. However, this is an area of experimentation that will be explored in a future release.


## References

1. [Nikko Storm, Amazon.com, Scalable Distributed Training using commodity GPU cloud computing.](https://s3-us-west-2.amazonaws.com/amazon.jobs-public-documents/strom_interspeech2015.pdf)

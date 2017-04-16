# Train LSTM with Multiple GPUs Using Model Parallelism

LSTM evaluation is inherently hard because of its complex data dependency. LSTM training, which has greater data dependency in reverse order at its back propagation phase, is even harder to parallelize. For general information about LSTM, see the excellent [introduction](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher. For an example of LSTM training with model parallelism, see [example/model-parallelism-lstm/](https://github.com/dmlc/mxnet/blob/master/example/model-parallel-lstm/lstm.py).



## Model Parallelism: Using Multiple GPUs As a Pipeline

Recently, there's been a great deal of heated discussion about model parallelism in applied machine learning. It was originally designed for the super large convolutional layer in GoogleNet. We borrowed the idea of placing each layer in one GPU. The primitive for model parallelism is the layers in a neural network model. The benefit that provides is that the GPU doesn't have to maintain the parameters of all the layers in memory. This reduces the memory limitation for large-scale tasks; for example, machine translation.

<img width="517" alt="screen shot 2016-05-06 at 10 13 16 pm" src="https://cloud.githubusercontent.com/assets/5545640/15089697/d6f4fca0-13d7-11e6-9331-7f94fcc7b4c6.png">

In the preceding figure, different LSTM models are assigned to different GPUs. After GPU 1 finishes computing layer 1 with the first sentence, the output is given to GPU 2. At the same time, GPU 1 fetches the next sentence and start training. This is significantly different from data parallelism because there is no contention to update the shared model at the end of each iteration, and most of the communication happens during pipelining intermediate results between GPUs.

In the current implementation, the layers are defined in [lstm_unroll()](https://github.com/dmlc/mxnet/blob/master/example/model-parallel-lstm/lstm.py).

## Workload Partitioning

Implementing model parallelism requires a good knowledge of the training task in order to partition the network throughout the GPUs. Although it requires detailed analysis that is beyond the scope of a course project, we found that you can apply some general principles:

- To avoid data transmission, place neighbor layers in the same GPU.
- To avoid bottlenecks in a pipeline, balance the workload between GPUs.
- Remember that different kinds of layers have different computation-memory properties.

<img width="449" alt="screen shot 2016-05-07 at 1 51 02 am" src="https://cloud.githubusercontent.com/assets/5545640/15090455/37a30ab0-13f6-11e6-863b-efe2b10ec2e6.png">

Let's take a quick look at the two pipelines in the preceding diagram. They both have eight layers with a decoder and an encoder layer. Based on our first principle, it's unwise to place all neighbor layers in separate GPUs. We also want to balance the workload across GPUs. Although the LSTM layers consume less memory than the decoder/encoder layers, they consume more computation time because of the dependency of the unrolled LSTM. Thus, the partition on the left will be faster than the one on the right because the workload is more evenly distributed in model parallelism.

Currently, the layer partition is implemented in [lstm.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L187) and configured in [lstm_ptb.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L187) using the `group2ctx` option.

## Apply Bucketing to Model Parallelism

To run model parallelism with bucketing, you need to unroll an LSTM model for each bucket to obtain an executor for each. For details about how the model is bound, see [lstm.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L154).

On the other hand, because model parallelism partitions the model/layers, the input data has to be transformed/transposed to the agreed shape. For more details, see [bucket_io](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L154).

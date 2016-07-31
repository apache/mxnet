# Train LSTM with Multiple GPUs Using Model Parallelism

An example of LSTM training with model parallelism is provided in [example/model-parallelism-lstm/](https://github.com/dmlc/mxnet/blob/master/example/model-parallel-lstm/lstm.py). 

## Long term short memory (LSTM)

There's a very good [introduction](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to LSTM by Christopher. 

LSTM evaluation is inheritantly hard due to its complex data dependency. LSTM training, which has more data dependency in reverse order at its back propagation phase, is even harder to parallelize. 


## Model parallel: Using multiple GPUs as a pipeline 

Model parallelism has been under heated discussion in applied machine learning recently. Originally, it is designed for super large convolutional layer in GoogleNet. We borrow the idea to place each layer in one GPU. The primitive for model parallel is the layers in neural network model. The benefit it brings is that GPU does not have to maintain parameters of all the layers in memory, which relieves the memory limitation in large scale tasks(for example, machine translation). 

<img width="517" alt="screen shot 2016-05-06 at 10 13 16 pm" src="https://cloud.githubusercontent.com/assets/5545640/15089697/d6f4fca0-13d7-11e6-9331-7f94fcc7b4c6.png">

In the figure above, we assign different lstm model to different GPUs. After GPU1 finish computing layer 1 with first sentence. The output will be given to GPU 2. At the same time, GPU 1 will fetch the next sentence and start training. This is significantly different from data parallelism that there's no contention to update the shared model at the end of each iteration, and most of the communication happens during pipelining intermediate results between GPU's. 

In the current implementation, the layers are defined in [lstm_unroll()](https://github.com/dmlc/mxnet/blob/master/example/model-parallel-lstm/lstm.py). 

## Workload Partitioning

Implementing Model Parallelism requires good knowledge of training task to partition the network throughout the GPUs. Although it requires detailed analysis that is beyond the scope of a course project, we found that we can lay down some general principles.

- Place neighbor layers in the same GPU to avoid data transmition.
- Balancing the workload between GPUs to avoid bottleneck in a pipeline situation.
- Remember, different kind of layers have different computation-memory properties. 

<img width="449" alt="screen shot 2016-05-07 at 1 51 02 am" src="https://cloud.githubusercontent.com/assets/5545640/15090455/37a30ab0-13f6-11e6-863b-efe2b10ec2e6.png">

Let us have a quick look into the 2 pipeline above. They both have 8 layers with a decoder and an encoder layer. Clearly, based on our first principle, it is unwise to place all neighbor layers in separate GPUs. One other thing is we want to balance the workload accross GPUs. Here LSTM layers, although having less memory comsumptions than decoder/encoder layers, will take up more computation time because dependency of unrolled LSTM. Thus, the partition on the left will be better in speed than the right because of a more even workload in Model Parallelism.

Currently the layer partition is implemented in [lstm.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L187) and configured in [lstm_ptb.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L187) using the `group2ctx` option.

## Apply Bucketing to Model Parallel 

To run model parallelism with bucketing, we need to unroll an LSTM model for each bucket, so that we obtain an executor each. You can refer to [lstm.py](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L154) for more details about how the model is bind. 

On the other hand, because model parallel partitions the model/layers, input data has to be transformed/transposed to the agreed shape. You can refer to [bucket_io](https://github.com/eric-haibin-lin/mxnet/blob/master/example/model-parallel-lstm/lstm.py#L154) for more details. 




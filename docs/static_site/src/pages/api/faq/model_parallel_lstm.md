---
layout: page_category
title:  Model Parallel
category: faq
faq_c: Model
question: How can I train using multiple GPUs with model parallelism?
permalink: /api/faq/model_parallel_lstm
---
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


# Training with Multiple GPUs Using Model Parallelism
Training deep learning models can be resource intensive.
Even with a powerful GPU, some models can take days or weeks to train.
Large long short-term memory (LSTM) recurrent neural networks
can be especially slow to train,
with each layer, at each time step, requiring eight matrix multiplications.
Fortunately, given cloud services like AWS,
machine learning practitioners often  have access
to multiple machines and multiple GPUs.
One key strength of _MXNet_ is its ability to leverage
powerful heterogeneous hardware environments to achieve significant speedups.

There are two primary ways that we can spread a workload across multiple devices.
In a previous document, [we addressed data parallelism](/api/faq/distributed_training),
an approach in which samples within a batch are divided among the available devices.
With data parallelism, each device stores a complete copy of the model.
Here, we explore _model parallelism_, a different approach.
Instead of splitting the batch among the devices, we partition the model itself.
Most commonly, we achieve model parallelism by assigning the parameters (and computation)
of different layers of the network to different devices.

In particular, we will focus on LSTM recurrent networks.
LSTMS are powerful sequence models, that have proven especially useful
for [natural language translation](https://arxiv.org/pdf/1409.0473.pdf), [speech recognition](https://arxiv.org/abs/1512.02595),
and working with [time series data](https://arxiv.org/abs/1511.03677).
For a general high-level introduction to LSTMs,
see the excellent [tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.


## Model Parallelism: Using Multiple GPUs As a Pipeline
Model parallelism in deep learning was first proposed
for the _extraordinarily large_ convolutional layer in GoogleNet.
From this implementation, we take the idea of placing each layer on a separate GPU.
Using model parallelism in such a layer-wise fashion
provides the benefit that no GPU has to maintain all of the model parameters in memory.

<img width="517" alt="screen shot 2016-05-06 at 10 13 16 pm" src="https://cloud.githubusercontent.com/assets/5545640/15089697/d6f4fca0-13d7-11e6-9331-7f94fcc7b4c6.png">

In the preceding figure, each LSTM layer is assigned to a different GPU.
After GPU 1 finishes computing layer 1 for the first sentence, it passes its output to GPU 2.
At the same time, GPU 1 fetches the next sentence and starts training.
This differs significantly from data parallelism.
Here, there is no contention to update the shared model at the end of each iteration,
and most of the communication happens when passing intermediate results between GPUs.


## Workload Partitioning

Implementing model parallelism requires knowledge of the training task.
Here are some general heuristics that we find useful:

- To minimize communication time, place neighboring layers on the same GPUs.
- Be careful to balance the workload between GPUs.
- Remember that different kinds of layers have different computation-memory properties.

<img width="449" alt="screen shot 2016-05-07 at 1 51 02 am" src="https://cloud.githubusercontent.com/assets/5545640/15090455/37a30ab0-13f6-11e6-863b-efe2b10ec2e6.png">

Let's take a quick look at the two pipelines in the preceding diagram.
They both have eight layers with a decoder and an encoder layer.
Based on our first principle, it's unwise to place all neighboring layers on separate GPUs.
We also want to balance the workload across GPUs.
Although the LSTM layers consume less memory than the decoder/encoder layers, they consume more computation time because of the dependency of the unrolled LSTM.
Thus, the partition on the left will be faster than the one on the right
because the workload is more evenly distributed.

---
layout: page_category
title: Visualize Neural Networks
category: faq
faq_c: Model
question: How do I visualize neural networks as computation graphs?
permalink: /api/faq/visualize_graph
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
# How to visualize Neural Networks as computation graph

Here, we'll demonstrate how to use ```mx.viz.plot_network```
for visualizing your neural networks. ```mx.viz.plot_network```
represents the neural network as a computation graph consisting of nodes and edges.
The visualizations make clear which nodes correspond to inputs,
where the computation starts,
and which correspond to output nodes,
from which the result can be read.

## Prerequisites
You need the [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/)
and [Graphviz](https://www.graphviz.org/) libraries to visualize the network.
Please make sure you have followed [installation instructions]({{'get_started'|relative_url}})
in setting up above dependencies along with setting up MXNet.

## Visualize the sample Neural Network

```mx.viz.plot_network``` takes [Symbol]({{'/api/python/docs/api/symbol/index'|relative}}), with your Network definition, and optional node_attrs, parameters for the shape of the node in the graph,  as input and generates a computation graph.

We will now try to visualize a sample Neural Network for linear matrix factorization:
- Start Jupyter notebook server
```bash
  $ jupyter notebook
```
- Access Jupyter notebook in your browser - http://localhost:8888/.
- Create a new notebook - "File -> New Notebook -> Python 2"
- Copy and run below code to visualize a simple network.

```python
import mxnet as mx
user = mx.symbol.Variable('user')
item = mx.symbol.Variable('item')
score = mx.symbol.Variable('score')

# Set dummy dimensions
k = 64
max_user = 100
max_item = 50

# user feature lookup
user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)

# item feature lookup
item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)

# predict by the inner product, which is elementwise product and then sum
net = user * item
net = mx.symbol.sum_axis(data = net, axis = 1)
net = mx.symbol.Flatten(data = net)

# loss layer
net = mx.symbol.LinearRegressionOutput(data = net, label = score)

# Visualize your network
mx.viz.plot_network(net)
```
You should see computation graph something like the following image:
<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/SampleNetworkVisualization.png
width=400/>

# References
* [Example MXNet Matrix Factorization](https://github.com/dmlc/mxnet/blob/master/example/recommenders/demo1-MF.ipynb)
* [Visualizing CNN Architecture of MXNet Tutorials](http://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/)

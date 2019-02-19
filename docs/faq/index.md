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

# MXNet FAQ

```eval_rst
.. toctree::
   :hidden:
   :glob:

   *
```

This section addresses common questions about how to use _MXNet_. These include performance issues, e.g., how to train with multiple GPUs.
They also include workflow questions, e.g., how to visualize a neural network computation graph.
These answers are fairly focused. For more didactic, self-contained introductions to neural networks
and full working examples, visit the [tutorials section](../tutorials/index.md).

## API

* [What's the difference between the Module and Gluon APIs for Python?](../api/python/index.html)

## Modeling
* [How do I fine-tune pre-trained models to a new dataset?](finetune.html)

* [How do I work with variable-length input in MXNet (bucketing)?](bucketing.html)

* [How do I visualize neural networks as computation graphs?](visualize_graph.html)


## Scale
* [How can I train with multiple CPU/GPUs on a single machine with data parallelism?](multi_devices.html)

* [How can I train using multiple machines with data parallelism?](distributed_training.html)

* [How can I train using multiple GPUs with model parallelism?](model_parallel_lstm.html)


## Speed
* [How do I use gradient compression with distributed training?](gradient_compression.html)

* [Can I use nnpack to improve the CPU performance of MXNet?](nnpack.html)

* [What are the best setup and data-handling tips and tricks for improving speed?](perf.html)

* [How do I use mixed precision with MXNet or Gluon?](float16.html)

## Deployment Environments
* [Can I run MXNet on smart or mobile devices?](smart_device.html)

* [How to use data from S3 for training?](s3_integration.md)

* [How to run MXNet on AWS?](https://docs.aws.amazon.com/mxnet/latest/dg/whatis.html)

* [How to do distributed training using MXNet on AWS?](http://docs.aws.amazon.com/mxnet/latest/dg/mxnet-on-ec2-cluster.html)

* [How do I run MXNet on a Raspberry Pi for computer vision?](../tutorials/embedded/wine_detector.html)

* [How do I run Keras 2 with MXNet backend?](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/installation.md)

* [How to convert MXNet models to Apple CoreML format?](https://github.com/apache/incubator-mxnet/tree/master/tools/coreml)

## Security
* [How to run MXNet securely?](security.html)

## Extend and Contribute to MXNet

* [How do I join the MXNet development discussion?](../community/mxnet_channels.html)

* [How do I contribute a patch to MXNet?](../community/contribute.html)

* [How do I implement operators in MXNet backend?](add_op_in_backend.html)

* [How do I create new operators in MXNet?](new_op.html)

* [How do I implement sparse operators in MXNet backend?](https://cwiki.apache.org/confluence/display/MXNET/A+Guide+to+Implementing+Sparse+Operators+in+MXNet+Backend)

* [How do I contribute an example or tutorial?](https://github.com/apache/incubator-mxnet/tree/master/example#contributing)

* [How do I set MXNet's environmental variables?](env_var.html)

## Questions about Using MXNet
If you need help with using MXNet, have questions about applying it to a particular kind of problem, or have a discussion topic, please use our [forum](https://discuss.mxnet.io).

## Issue Tracker
We track bugs and new feature requests in the MXNet Github repo in the issues folder: [mxnet/issues](https://github.com/apache/incubator-mxnet/issues).

## Roadmap
MXNet is evolving fast. To see what's next and what we are working on internally, go to the [MXNet Roadmap](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Roadmap).

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

# MXNet's Ecosystem

Community contributions to MXNet have added many new valuable features and functionality to support use cases such as model serving & portability, easy and flexible APIs, and educational material like crash courses and online books. This ecosystem page lists the projects that use MXNet, teach MXNet, or augment MXNet in some way.


## Highlighted Project

[![Promo image](https://cdn-images-1.medium.com/max/800/1*PwIMdZM7tpt3rmcyhlk2FQ.png)](https://medium.com/apache-mxnet/announcing-keras-mxnet-v2-2-4b8404568e75)
#### [Keras-MXNet v2.2 released!](https://medium.com/apache-mxnet/announcing-keras-mxnet-v2-2-4b8404568e75)


## Contents

* [Learning MXNet and other Educational Resources](#learning-mxnet-and-other-educational-resources)
* [MXNet APIs](#mxnet-apis)
* [Toolkits to Extend MXNet](#toolkits-to-extend-mxnet)
* [Debugging and Visualization](#debugging-and-visualization)
* [Model Serving](#model-serving)
* [Model Zoos](#model-zoos)
* [Contributions](#contributions)


## Learning MXNet and other Educational Resources

* [Gluon 60 Minute Crash Course](https://gluon-crash-course.mxnet.io/) - deep learning practitioners can learn Gluon quickly with these six 10-minute tutorials.
    - [YouTube Series](https://www.youtube.com/playlist?list=PLkEvNnRk8uVmVKRDgznk3o3LxmjFRaW7s)
* [The Straight Dope](https://gluon.mxnet.io/) - a series of notebooks designed to teach deep learning using the Gluon Python API for MXNet.


## MXNet APIs

* [Clojure API](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package) - use MXNet with Clojure.
* [C++ API](../api/c++/index.html) - not be confused with the C++ backend, this API allows C++ programmers to train networks in C++.
* [Gluon Python Interface](../gluon/index.html) - train complex models imperatively and then deploy with a symbolic graph.
* [Julia API](../api/julia/index.html) *(Community Supported)* - train models with multiple GPUs using Julia.
* [Keras-MXNet](https://github.com/awslabs/keras-apache-mxnet) - design with Keras2 and train with MXNet as the backend for 2x or more speed improvement.
* [MinPy](https://github.com/dmlc/minpy) - Pure numpy practice with third party operator integration and MXNet as backend for GPU computing
* [Module Python API](../api/python/index.html) - backed by the Symbol API, you can define your network in a declarative fashion.
* [ONNX-MXnet API](../api/python/contrib/onnx.html) - train and use Open Neural Network eXchange (ONNX) model files.
* [Perl API](../api/perl/index.html) *(Community Supported)* - train models with multiple GPUs using Perl.
* [R API](https://mxnet.incubator.apache.org/api/r/index.html) *(Community Supported)* - train models with multiple GPUs using R.
* [Scala Infer API](../api/scala/infer.html) - model loading and inference functionality.
* [TensorFuse](https://github.com/dementrock/tensorfuse) - Common interface for Theano, CGT, TensorFlow, and MXNet (experimental) by [dementrock](https://github.com/dementrock)


## Toolkits to Extend MXNet

* [Gluon CV](https://gluon-cv.mxnet.io/) - state-of-the-art deep learning algorithms in computer vision.
* [Gluon NLP](https://gluon-nlp.mxnet.io/) - state-of-the-art deep learning models in natural language processing.
* [Sockeye](https://github.com/awslabs/sockeye) - a sequence-to-sequence framework for Neural Machine Translation


## Debugging and Visualization

* [MXBoard](https://github.com/awslabs/mxboard) - lets you to visually inspect and interpret your MXNet runs and graphs using the TensorBoard software.


## Model Serving

* [MXNet Model Server (MMS)](https://github.com/awslabs/mxnet-model-server) - simple yet scalable solution for model inference.


## Model Zoos

* [Gluon Model Zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) - models trained in Gluon and available through Gluon's model zoo API.
* [ONNX Model Zoo](https://github.com/onnx/models) - ONNX models from a variety of ONNX-supported frameworks.


## Contributions

Do you know of a project or resource in the MXNet ecosystem that should be listed here? Or would you like to get involved by providing your own contribution? Check out the [guide for contributing to MXNet](contribute.html), and browse the [design proposals](https://cwiki.apache.org/confluence/display/MXNET/Design+Proposals) to see what others are working on. You might find something you would like to help with or use those design docs as a template for your own proposal. Use one of the [developer communication channels](https://mxnet.incubator.apache.org/community/contribute.html#mxnet-dev-communications) if you would like to know more, or [create a GitHub issue](https://github.com/apache/incubator-mxnet/issues/new) if you would like to propose something for the MXNet ecosystem.

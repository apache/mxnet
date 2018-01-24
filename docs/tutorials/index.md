# Tutorials

MXNet has two primary high-level interfaces for its deep learning engine: the Gluon API and the Symbol API. Tutorials for each are provided below.

The difference between the two is an imperative versus symbolic programming style. Gluon makes it easy to prototype, build, and train deep learning models without sacrificing training speed by enabling both (1) intuitive imperative Python code development and (2) faster execution by automatically generating a symbolic execution graph using the hybridization feature.

**TLDR**: If you are new to deep learning or MXNet, you should start with the Gluon tutorials.

The Gluon and Symbol tutorials are in Python, but you can also find a variety of other MXNet tutorials, such as R, Scala, and C++ in the [Other Languages API Tutorials](#other-mxnet-api-tutorials) section below.

[Example scripts and applications](#example-scripts-and-applications) as well as [contribution](#contributing-tutorials) info is below.

<script type="text/javascript" src='../_static/js/options.js'></script>


<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Python</button>
</div>

<!-- Gluon vs Symbol -->
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Gluon</button>
  <button type="button" class="btn btn-default opt">Symbol</button>
</div>


<!-- Levels -->
<div class="gluon symbol">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Beginner</button>
  <button type="button" class="btn btn-default opt">Intermediate</button>
  <button type="button" class="btn btn-default opt">Advanced</button>
</div>
</div>


<!-- Beginner Topics -->
<div class="beginner">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Data Loading</button>
  <button type="button" class="btn btn-default opt">Basic Networks</button>
  <button type="button" class="btn btn-default opt">Linear Regression</button>
</div>
</div>


<!-- Intermediate Topics -->
<div class="intermediate">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Image Recognition</button>
  <button type="button" class="btn btn-default opt">Human Language</button>
  <button type="button" class="btn btn-default opt">Recommender Systems</button>
  <button type="button" class="btn btn-default opt">Customization</button>
</div>
</div>


<!-- Advanced Topics -->
<div class="advanced">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Distributed Training</button>
  <button type="button" class="btn btn-default opt">Optimization</button>
  <button type="button" class="btn btn-default opt">Adversarial Networks</button>
</div>
</div>
<!-- END - Main Menu -->
<hr>

<div class="gluon">
<div class="beginner">


<div class="data-loading">

- [Manipulate data the MXNet way with ndarray](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html)
- [Serialization - saving, loading and checkpointing](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html)
- [NDArray in Compressed Sparse Row Storage Format](http://mxnet.incubator.apache.org/tutorials/sparse/csr.html)
- [Sparse Gradient Updates](http://mxnet.incubator.apache.org/tutorials/sparse/row_sparse.html)
</div>


<div class="basic-networks">

- [Simple autograd example](http://mxnet.incubator.apache.org/tutorials/gluon/autograd.html)
- [Automatic differentiation with autograd](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)
- [Neural network building blocks with gluon](http://mxnet.incubator.apache.org/tutorials/gluon/gluon.html)
- [Hybrid network example](http://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html)
</div>

<div class="linear-regression">

- [Linear regression with gluon](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html)
</div>

</div> <!--end of beginner-->


<div class="intermediate">


<div class="image-recognition">

- [Handwritten digit recognition (MNIST)](http://mxnet.incubator.apache.org/tutorials/gluon/mnist.html)
- [Multilayer perceptrons in gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html)
- [Convolutional Neural Networks (CNNs) in gluon](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)
- [Multi-class object detection using CNNs in gluon](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)
- [Visual question answering in gluon](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html)
- [Transferring knowledge through fine-tuning (not hotdog)](http://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html)
</div>


<div class="human-language">

- [Simple Recurrent Neural Networks (RNNs) with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/simple-rnn.html)
- [Long Short-Term Memory (LSTM) RNNs with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/lstm-scratch.html)
- [Gated Recurrent Unit (GRU) RNNs with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/gru-scratch.html)
- [Advanced RNNs with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html)
- [Tree LSTM modeling for semantic relatedness](http://gluon.mxnet.io/chapter09_natural-language-processing/tree-lstm.html)
</div>


<div class="recommender-systems">

- [Introduction to recommender systems](http://gluon.mxnet.io/chapter11_recommender-systems/intro-recommender-systems.html)
- [Matrix factorization in recommendation systems](http://mxnet.incubator.apache.org/tutorials/python/matrix_factorization.html)
</div>


<div class="customization">

- [Designing a custom layer with gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html)
- [Creating custom operators with numpy](http://mxnet.incubator.apache.org/tutorials/gluon/customop.html)
</div>

</div> <!--end of intermediate-->


<div class="advanced">


<div class="distributed-training">

- [Training on multiple GPUs with gluon](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html)
- [Distributed training with multiple machines with gluon](http://gluon.mxnet.io/chapter07_distributed-learning/training-with-multiple-machines.html)
</div>


<div class="optimization">

- [Plumbing: A look under the hood of gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html)
- [Fast, portable neural networks with Gluon HybridBlocks](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)
</div>


<div class="adversarial-networks">

- [Introduction to Generative Adversarial Networks (GANs)](http://gluon.mxnet.io/chapter14_generative-adversarial-networks/gan-intro.html)
- [Deep convolutional GANs](http://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html)
- [Image transduction GANs (Pix2Pix)](http://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html)
</div>

</div> <!--end of advanced-->
</div> <!--end of gluon-->


<div class="symbol">


<div class="python">


<div class="beginner">


<div class="data-loading">

- [Intro to Data Loading in MXNet](http://mxnet.incubator.apache.org/tutorials/basic/ndarray.html)
- [NDArray Indexing](http://mxnet.incubator.apache.org/tutorials/basic/ndarray_indexing.html)
- [Loading external data with MXNet Data Loading API](http://mxnet.incubator.apache.org/tutorials/basic/data.html)
- [NDArray in Compressed Sparse Row Storage Format](http://mxnet.incubator.apache.org/tutorials/sparse/csr.html)
- [Sparse Gradient Updates](http://mxnet.incubator.apache.org/tutorials/sparse/row_sparse.html)
- [Distributed key-value store](http://mxnet.incubator.apache.org/tutorials/python/kvstore.html)
</div>


<div class="basic-networks">

- [Symbol API](http://mxnet.incubator.apache.org/tutorials/basic/symbol.html)
- [Module API](http://mxnet.incubator.apache.org/tutorials/basic/module.html)
</div>


<div class="linear-regression">

- [Linear regression](http://mxnet.incubator.apache.org/tutorials/python/linear-regression.html)
- [Train a Linear Regression Model with Sparse Symbols](http://mxnet.incubator.apache.org/tutorials/sparse/train.html)
</div>


</div> <!--end of beginner-->


<div class="intermediate">


<div class="image-recognition">

- [MNIST - handwriting recognition](http://mxnet.incubator.apache.org/tutorials/python/mnist.html)
<!-- broken #9532
- [Image recognition](http://mxnet.incubator.apache.org/tutorials/python/predict_image.html)
-->

- [Fine-tuning a pre-trained ImageNet model with a new dataset](http://mxnet.incubator.apache.org/faq/finetune.html)
</div>


<div class="human-language">

- [Text classification (NLP) on Movie Reviews](http://mxnet.incubator.apache.org/tutorials/nlp/cnn.html)
- [Connectionist Temporal Classification](http://mxnet.incubator.apache.org/tutorials/speech_recognition/ctc.html)
</div>


<div class="recommender-systems">

- [Matrix factorization in recommender systems](http://mxnet.incubator.apache.org/tutorials/python/matrix_factorization.html)
</div>


<div class="customization">

- [Fine-tuning with pre-trained models](http://mxnet.incubator.apache.org/faq/finetune.html)
</div>

</div> <!--end of intermediate-->


<div class="advanced">


<div class="adversarial-networks">

- [Generative Adversarial Networks](http://mxnet.incubator.apache.org/tutorials/unsupervised_learning/gan.html)
</div>


<div class="distributed-training">

- [Large scale image classification](http://mxnet.incubator.apache.org/tutorials/vision/large_scale_classification.html)
</div>


</div> <!--end of advanced-->

</div> <!--end of python-->


</div> <!--end of symbol-->




<hr>

## Other Languages API Tutorials


<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">C</button>
  <button type="button" class="btn btn-default opt">Scala</button>
  <button type="button" class="btn btn-default opt">R</button>
</div>
<hr>

<div class="c">

- [MNIST with the MXNet C++ API](http://mxnet.incubator.apache.org/tutorials/c%2B%2B/basics.html)
</div> <!--end of c++-->


<div class="r">

- [NDArray: Vectorized Tensor Computations on CPUs and GPUs with R](http://mxnet.incubator.apache.org/tutorials/r/ndarray.html)
- [Symbol API with R](http://mxnet.incubator.apache.org/tutorials/r/symbol.html)
- [Custom Iterator](http://mxnet.incubator.apache.org/tutorials/r/CustomIterator.html)
- [Callback Function](http://mxnet.incubator.apache.org/tutorials/r/CallbackFunction.html)
- [Five minute neural network](http://mxnet.incubator.apache.org/tutorials/r/fiveMinutesNeuralNetwork.html)
- [MNIST with R](http://mxnet.incubator.apache.org/tutorials/r/mnistCompetition.html)
- [Classify images via R with a pre-trained model](http://mxnet.incubator.apache.org/tutorials/r/classifyRealImageWithPretrainedModel.html)
- [Char RNN Example with R](http://mxnet.incubator.apache.org/tutorials/r/charRnnModel.html)
- [Custom loss functions in R](http://mxnet.incubator.apache.org/tutorials/r/CustomLossFunction.html)


</div> <!--end of r-->


<div class="scala">

- [Setup your MXNet with Scala on InelliJ](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html)
- [MNIST with the Scala API](http://mxnet.incubator.apache.org/tutorials/scala/mnist.html)
- [Use Scala to build a Long Short-Term Memory network that generates Barack Obama's speech patterns](http://mxnet.incubator.apache.org/tutorials/scala/char_lstm.html)
</div>

<hr>


## Example Scripts and Applications

More tutorials and examples are available in the [GitHub repository](https://github.com/apache/incubator-mxnet/tree/master/example).


## Learn More About Gluon!

Most of the Gluon tutorials are hosted on [gluon.mxnet.io](http://gluon.mxnet.io), and you may want to follow the chapters on directly the Gluon site.


## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, [review these details](https://github.com/apache/incubator-mxnet/tree/master/example#contributing) on example and tutorial writing.

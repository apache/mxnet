# Tutorials

MXNet tutorials can be found in this section. A variety of language bindings are available for MXNet (including Python, Scala, C++ and R) and we have a different tutorial section for each language.

Are you new to MXNet, and don't have a preference on language? We currently recommend starting with Python, and specifically the Gluon APIs (versus Module APIs) as they're more flexible and easier to debug.

Another great resource for learning MXNet is our [examples section](https://github.com/apache/incubator-mxnet/tree/master/example) which includes a wide variety of models (from basic to state-of-the-art) for a wide variety of tasks including: object detection, style transfer, reinforcement learning, and many others.

<hr>

## Python Tutorials

We have two types of API available for Python: Gluon APIs and Module APIs. [See here](/api/python/gluon/gluon.html) for a comparison.

A comprehensive introduction to Gluon can be found at [The Straight Dope](http://gluon.mxnet.io/). Structured like a book, it build up from first principles of deep learning and take a theoretical walkthrough of progressively more complex models using the Gluon API. Also check out the [60-Minute Gluon Crash Course](http://gluon-crash-course.mxnet.io/) if you're short on time or have used other deep learning frameworks before.

Use the tutorial selector below to filter to the relevant tutorials. You might see a download link in the top right corner of some tutorials. Use this to download a Jupyter Notebook version of the tutorial, and re-run and adjust the code as you wish.

<script type="text/javascript" src='../_static/js/options.js'></script>

<!-- Gluon vs Module -->
Select API:&nbsp;
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active" style="font-size:22px">Gluon</button>
  <button type="button" class="btn btn-default opt"   style="font-size:22px">Module</button>
</div>
<!-- END - Main Menu -->
<br>
<div class="gluon">

* Getting Started
    * [60-Minute Gluon Crash Course](http://gluon-crash-course.mxnet.io/) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
    * [MNIST Handwritten Digit Classification](/tutorials/gluon/mnist.html)
* Models
    * [Model Zoo: using pre-trained models](/tutorials/gluon/pretrained_models.html)
    * [Linear Regression](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
    * [Word-level text generation with RNN, LSTM and GRU](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
    * [Visual Question Answering](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
* Practitioner Guides
    * [Multi-GPU training](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
    * [Checkpointing and Model Serialization (a.k.a. saving and loading)](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/> ([Alternative](/tutorials/gluon/save_load_params.html))
    * [Inference using an ONNX model](/tutorials/onnx/inference_on_onnx_model.html)
    * [Fine-tuning an ONNX model on Gluon](/tutorials/onnx/fine_tuning_gluon.html)
    * [Visualizing Decisions of Convolutional Neural Networks](/tutorials/vision/cnn_visualization.html)
* API Guides
    * Core APIs
        * NDArray
            * [NDArray API](/tutorials/gluon/ndarray.html) ([Alternative](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>)
            * [Advanced NDArray API](/tutorials/basic/ndarray.html)
            * [NDArray Indexing](https://mxnet.incubator.apache.org/tutorials/basic/ndarray_indexing.html)
            * Sparse NDArray
                * [Sparse Gradient Updates (RowSparseNDArray)](/tutorials/sparse/row_sparse.html)
                * [Compressed Sparse Row Storage Format (CSRNDArray)](/tutorials/sparse/csr.html)
                * [Linear Regression with Sparse Symbols](/tutorials/sparse/train.html)
        * Symbol
            * [Symbol API](/tutorials/basic/symbol.html) (Caution: written before Gluon existed)
        * KVStore
            * [Key-Value Store API](/tutorials/python/kvstore.html)
    * Gluon APIs
        * Blocks and Operators
            * [Blocks](/tutorials/gluon/gluon.html) ([Alternative](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>)
            * [Custom Blocks](/tutorials/gluon/custom_layer.html) ([Alternative](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>)
            * [HybridBlocks](/tutorials/gluon/hybrid.html) ([Alternative](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>)
            * [Block Naming](/tutorials/gluon/naming.html)
            * [Custom Operators](/tutorials/gluon/customop.html)
        * Autograd
            * [AutoGrad API](/tutorials/gluon/autograd.html)
            * [AutoGrad API with chain rule](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
            * [AutoGrad API with Python control flow](http://gluon-crash-course.mxnet.io/autograd.html) <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/External_link_font_awesome.svg" alt="External link" height="15px" style="margin: 0px 0px 3px 3px;"/>
        * Data
            * [Datasets and DataLoaders](/tutorials/gluon/datasets.html)
            * [Applying Data Augmentation](/tutorials/gluon/data_augmentation.html)
            * [Data Augmentation with Masks (for Object Segmentation)](https://mxnet.incubator.apache.org/tutorials/python/data_augmentation_with_masks.html)
</div> <!--end of gluon-->

<div class="module">

* Getting Started
    * [Module API](/tutorials/basic/module.html)
    * [MNIST Handwritten Digit Classification](/tutorials/python/mnist.html)
* Models
    * [Linear Regression](/tutorials/python/linear-regression.html)
    * [Linear Regression with Sparse Symbols](/tutorials/sparse/train.html)
    * [MNIST Handwritten Digit Classification](/tutorials/python/mnist.html)
    * [Movie Review Classification using Convolutional Networks](/tutorials/nlp/cnn.html)
    * [Generative Adversarial Networks (GANs)](/tutorials/unsupervised_learning/gan.html)
    * [Recommender Systems using Matrix Factorization](/tutorials/python/matrix_factorization.html)
    * [Speech Recognition with Connectionist Temporal Classification Loss](/tutorials/speech_recognition/ctc.html)
* Practitioner Guides
    * [Predicting on new images using a pre-trained ImageNet model](/tutorials/python/predict_image.html)
    * [Fine-Tuning a pre-trained ImageNet model with a new dataset](/faq/finetune.html)
    * [Large-Scale Multi-Host Multi-GPU Image Classification](/tutorials/vision/large_scale_classification.html)
    * [Importing an ONNX model into MXNet](/tutorials/onnx/super_resolution.html)
* API Guides
    * Core APIs
        * NDArray
            * [NDArray API](/tutorials/gluon/ndarray.html)
            * [Advanced NDArray API](/tutorials/basic/ndarray.html)
            * [NDArray Indexing](/tutorials/basic/ndarray_indexing.html)
            * Sparse NDArray
                * [Sparse Gradient Updates (RowSparseNDArray)](/tutorials/sparse/row_sparse.html)
                * [Compressed Sparse Row Storage Format (CSRNDArray)](/tutorials/sparse/csr.html)
                * [Linear Regression with Sparse Symbols](/tutorials/sparse/train.html)
        * Symbol
            * [Symbol API](/tutorials/basic/symbol.html)
        * KVStore
            * [Key-Value Store API](/tutorials/python/kvstore.html)
    * Module APIs
        * [Module API](/tutorials/basic/module.html)
        * Data
            * [Data Iterators](/tutorials/basic/data.html)
            * [Applying Data Augmentation](/tutorials/python/data_augmentation.html)
            * [Types of Data Augmentation](/tutorials/python/types_of_data_augmentation.html)
</div> <!--end of module-->

<hr>

## Scala Tutorials

* Getting Started
    * [MXNet and JetBrain's IntelliJ](/tutorials/scala/mxnet_scala_on_intellij.html)
* Models
    * [MNIST Handwritten Digit Recognition with Fully Connected Network](/tutorials/scala/mnist.html)
    * [Barack Obama speech generation with Character-level LSTM](/tutorials/scala/char_lstm.html)

<hr>

## C++ Tutorials

* Models
    * [MNIST Handwritten Digit Recognition with Fully Connected Network](/tutorials/c%2B%2B/basics.html)

<hr>

## R Tutorials

* Getting Started
    * [Basic Classification & Regression](/tutorials/r/fiveMinutesNeuralNetwork.html)
    * [Using a pre-trained model for Image Classification](/tutorials/r/classifyRealImageWithPretrainedModel.html)
* Models
    * [MNIST Handwritten Digit Classification with Convolutional Network](/tutorials/r/mnistCompetition.html)
    * [Shakespeare generation with Character-level RNN](/tutorials/r/charRnnModel.html)
* API Guides
    * [NDArray API](/tutorials/r/ndarray.html)
    * [Symbol API](/tutorials/r/symbol.html)
    * [Callbacks](/tutorials/r/CallbackFunction.html)
    * [Custom Data Iterators](/tutorials/r/CustomIterator.html)
    * [Custom Loss Functions](/tutorials/r/CustomLossFunction.html)
 
<hr>
 
## Contributing Tutorials

We really appreciate contributions, and tutorials are a great way to share your knowledge and help the community. After you have followed [these steps](https://github.com/apache/incubator-mxnet/tree/master/example#contributing), please submit a pull request on Github.

And if you have any feedback on this section please raise an issue on Github.

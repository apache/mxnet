#Awesome MXNet 

This page contains a curated list of awesome MXnet examples, tutorials and blogs. It is inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning).

## Contributing

If you want to contribute to this list and the examples, please open a new pull request.

##List of examples

###Langauges Binding Examples
------------------
* [C++ examples](https://github.com/dmlc/mxnet/tree/master/example/cpp) - Example code for using C++ interface, including NDArray, symbolic layer and models.
* [MXNet Python](http://mxnet.readthedocs.org/en/latest/python/index.html) - Python libary
* [MXNetR](http://mxnet.readthedocs.org/en/latest/R-package/index.html) - R library
* [MXNet.jl](http://mxnetjl.readthedocs.org/en/latest/) - Julia library
* [gomxnet](https://github.com/jdeng/gomxnet) - Go binding
* [MXNet JNI](https://github.com/dmlc/mxnet/tree/master/amalgamation/jni) - JNI(Android) library
* [MXNet Amalgamation](https://github.com/dmlc/mxnet/tree/master/amalgamation) - Amalgamation (entire library in a single file)
* [MXNet Javascript](https://github.com/dmlc/mxnet.js/) - MXNetJS: Javascript Package for Deep Learning in Browser (without server)

###Deep Learning Examples
--------------
* [Image classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification) - Image classification on MNIST,CIFAR,ImageNet-1k,ImageNet-Full, ***with multiple GPU and distributed training***.
* [Recurrent Neural Net](https://github.com/dmlc/mxnet/tree/master/example/rnn) - LSTM and RNN for language modeling and character level generation (Char-RNN).
* [Autoencoder](https://github.com/dmlc/mxnet/tree/master/example/autoencoder) - Auto encoder training.
* [Numpy Operator Customization](https://github.com/dmlc/mxnet/tree/master/example/numpy-ops) - Example on quick customize new ops with numpy.
* [Adversary Sample Generation](adversary) - Find adversary sample by using fast sign method.
* [Neural Art](neural-style) -  Generate artistic style images.
* [Kaggle 1st national data science bowl](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb1) - a MXnet example for Kaggle Nation Data Science Bowl 1
* [Kaggle 2nd national data science bowl](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb2) - a tutorial for Kaggle Second Nation Data Science Bowl


###IPython Notebooks
-----------------
* [Predict with Pre-trained model](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb) - Notebook on how to predict with pretrained model.
* [composite symbol](notebooks/composite_symbol.ipynb) - A demo of how to composite a symbolic Inception-BatchNorm Network
* [cifar-10 recipe](notebooks/cifar10-recipe.ipynb) - A step by step demo of how to use MXNet
* [cifar-100](notebooks/cifar-100.ipynb) - A demo of how to train a 75.68% accuracy CIFAR-100 model
* [simple bind](notebooks/simple_bind.ipynb) - A demo of low level training API.

###Mobile App Examples
-------------------
* [MXNet Android Classification App](https://github.com/Leliana/WhatsThis) - Image classification on Android with MXNet.
* [MXNet iOS Classification App](https://github.com/pppoe/WhatsThis-iOS) - Image classification on iOS with MXNet.
* [Compile MXnet on Xcode (in Chinese)](http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/) - a step-by-step tutorial of compiling MXnet on Xcode for iOS app

###Web Predictive Services
-----------------------
* [MXNet Shinny](https://github.com/thirdwing/mxnet_shiny) - Source code for quickly creating a Shiny R app to host online image classification.
* [Machine Eye] (http://rupeshs.github.io/machineye/)- Web service for local image file/image URL classification without uploading.
## List of tutorials

### Deep learning for hackers with MXnet

* Deep learning for hackers with MXnet (1) GPU installation and MNIST [English](https://no2147483647.wordpress.com/2015/12/07/deep-learning-for-hackers-with-mxnet-1/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial1) - a tutorial of installing MXnet with GPU and introduction to deep learning by MNIST example.
* Deep learning for hackers with MXnet (2): Neural art [English](https://no2147483647.wordpress.com/2015/12/21/deep-learning-for-hackers-with-mxnet-2/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial2) - a tutorial of generating Van Gogh style cat paintings.

### Setup AWS instance with MXnet
* [Setup Amazon AWS GPU instance with MXnet](https://no2147483647.wordpress.com/2016/01/16/setup-amazon-aws-gpu-instance-with-mxnet/) - AWS GPU instance setup with GPU (CUDA with latest cuDNN and S3 support)

### Kaggle tutorials
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (Python)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18079/end-to-end-deep-learning-tutorial-0-0392) - an end-to-end python tutorial for Kaggle heart disease diagnose competition (public leaderboard score 0.0392)
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (R)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18122/deep-learning-model-in-r) - an end-to-end R tutorial for Kaggle heart disease diagnose competition

###Learning Note
* [Learning Note in Chinese](https://github.com/zhubuntu/MXNet-Learning-Note) - Mxnet learning note in chinese.

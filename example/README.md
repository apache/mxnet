#Awesome MXNet 

This page contains a curated list of awesome MXnet examples, tutorials and blogs. It is inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning).

  - [Contributing](#contributing)
  - [List of examples](#list-of-examples)
    - [Langauges Binding Examples](#language-binding-examples)
    - [Deep Learning Examples](#deep-learning-examples)
    - [IPython Notebooks](#ipython-notebooks)
    - [Mobile App Examples](#mobile-apps-examples)
    - [Web Predictive Services](#web-predictive-services)
  - [List of tutorials](#list-of-tutorials)
    - [GPU Technology Conference 2016 Hands-on session](#gtc2016-hands-on)
    - [Deep learning for hackers with MXnet](#deep-learning-for-hackers)
    - [MXnet setup on AWS](#mxnet-aws)
    - [Kaggle tutorials](#kaggle-tutorials)
    - [Learning Note](#learning-note)
  - [Machine Learning Challenge Winning Solutions](#winning-solutions)
  - [Tools with MXnet](#tools-with-mxnet)

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

##<a name="list-of-examples"></a>List of examples

###<a name="language-binding-examples"></a>Langauges Binding Examples
------------------
* [C++ examples](https://github.com/dmlc/mxnet/tree/master/example/cpp) - Example code for using C++ interface, including NDArray, symbolic layer and models.
* [MXNet Python](http://mxnet.readthedocs.org/en/latest/python/index.html) - Python libary
* [MXNetR](http://mxnet.readthedocs.org/en/latest/R-package/index.html) - R library
* [MXNet.jl](http://mxnetjl.readthedocs.org/en/latest/) - Julia library
* [gomxnet](https://github.com/jdeng/gomxnet) - Go binding
* [MXNet JNI](https://github.com/dmlc/mxnet/tree/master/amalgamation/jni) - JNI(Android) library
* [MXNet Amalgamation](https://github.com/dmlc/mxnet/tree/master/amalgamation) - Amalgamation (entire library in a single file)
* [MXNet Javascript](https://github.com/dmlc/mxnet.js/) - MXNetJS: Javascript Package for Deep Learning in Browser (without server)

###<a name="deep-learning-examples"></a>Deep Learning Examples
--------------
* [Image classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification) - Image classification on MNIST,CIFAR,ImageNet-1k,ImageNet-Full, ***with multiple GPU and distributed training***.
* [Recurrent Neural Net](https://github.com/dmlc/mxnet/tree/master/example/rnn) - LSTM and RNN for language modeling and character level generation (Char-RNN).
* [Autoencoder](https://github.com/dmlc/mxnet/tree/master/example/autoencoder) - Auto encoder training.
* [Numpy Operator Customization](https://github.com/dmlc/mxnet/tree/master/example/numpy-ops) - Example on quick customize new ops with numpy.
* [Adversary Sample Generation](adversary) - Find adversary sample by using fast sign method.
* [Neural Art](neural-style) -  Generate artistic style images.
* [Kaggle 1st national data science bowl](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb1) - a MXnet example for Kaggle Nation Data Science Bowl 1
* [Kaggle 2nd national data science bowl](https://github.com/dmlc/mxnet/tree/master/example/kaggle-ndsb2) - a tutorial for Kaggle Second Nation Data Science Bowl
* [CNN for Text Classification](cnn_text_classification) - a MXnet example for text classification
* [Chinese plate recognition](https://github.com/imistyrain/mxnet-mr) - Recognize Chinese vehicle plate, by [liuruoze](https://github.com/liuruoze)

###<a name="ipython-notebooks"></a>IPython Notebooks
-----------------
* [Predict with Pre-trained model](https://github.com/dmlc/mxnet/blob/master/example/notebooks/predict-with-pretrained-model.ipynb) - Notebook on how to predict with pretrained model.
* [composite symbol](notebooks/composite_symbol.ipynb) - A demo of how to composite a symbolic Inception-BatchNorm Network
* [cifar-10 recipe](notebooks/cifar10-recipe.ipynb) - A step by step demo of how to use MXNet
* [cifar-100](notebooks/cifar-100.ipynb) - A demo of how to train a 75.68% accuracy CIFAR-100 model
* [simple bind](notebooks/simple_bind.ipynb) - A demo of low level training API.

###<a name="mobile-apps-examples"></a>Mobile App Examples
-------------------
* [MXNet Android Classification App](https://github.com/Leliana/WhatsThis) - Image classification on Android with MXNet.
* [MXNet iOS Classification App](https://github.com/pppoe/WhatsThis-iOS) - Image classification on iOS with MXNet.
* [Compile MXnet on Xcode (in Chinese)](http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/) - a step-by-step tutorial of compiling MXnet on Xcode for iOS app

###<a name="web-predictive-services"></a>Web Predictive Services
-----------------------
* [MXNet Shinny](https://github.com/thirdwing/mxnet_shiny) - Source code for quickly creating a Shiny R app to host online image classification.
* [Machine Eye] (http://rupeshs.github.io/machineye/) - Web service for local image file/image URL classification without uploading.

##<a name="list-of-tutorials"></a>List of tutorials

###<a name="gtc2016-hands-on"></a>GPU Technology Conference 2016 Hands-on session

* [Video on GTC 2016 site] (http://on-demand.gputechconf.com/gtc/2016/video/L6143.html) 
* [Video backup in Mainland China](http://pan.baidu.com/s/1eS58Gue) 
* [iPython Notebook](https://github.com/dmlc/mxnet-gtc-tutorial) 

###<a name="deep-learning-for-hackers"></a>Deep learning for hackers with MXnet

* Deep learning for hackers with MXnet (1) GPU installation and MNIST [English](https://no2147483647.wordpress.com/2015/12/07/deep-learning-for-hackers-with-mxnet-1/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial1) - a tutorial of installing MXnet with GPU and introduction to deep learning by MNIST example.
* Deep learning for hackers with MXnet (2): Neural art [English](https://no2147483647.wordpress.com/2015/12/21/deep-learning-for-hackers-with-mxnet-2/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial2) - a tutorial of generating Van Gogh style cat paintings.

### <a name="mxnet-aws"></a>MXnet setup on AWS
* [Setup Amazon AWS GPU instance with MXnet](https://no2147483647.wordpress.com/2016/01/16/setup-amazon-aws-gpu-instance-with-mxnet/) - AWS GPU instance setup with GPU (CUDA with latest cuDNN and S3 support)
* [Intro Guide to AWS (MXnet with Julia)](http://www.datasciencebowl.com/aws_guide/) - A step-by-step guide of using spot instances with Amazon Web Services (AWS) to help you save money when training DSB models on Mxnet by [Mike Kim](http://www.datasciencebowl.com/author/mikekim/)

### <a name="kaggle-tutorials"></a>Kaggle tutorials
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (Python)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18079/end-to-end-deep-learning-tutorial-0-0392) - an end-to-end python tutorial for Kaggle heart disease diagnose competition (public leaderboard score 0.0392)
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (R)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18122/deep-learning-model-in-r) - an end-to-end R tutorial for Kaggle heart disease diagnose competition

### <a name="learning-note"></a>Learning Note
* [Learning Note in Chinese](https://github.com/zhubuntu/MXNet-Learning-Note) - Mxnet learning note in Chinese.
* [Getting Started with MXNet](https://indico.io/blog/getting-started-with-mxnet/) by [indico.io](https://indico.io) (Chinese Translation [MXnet实践](http://www.infoq.com/cn/articles/practise-of-mxnet) by [侠天](http://www.infoq.com/cn/author/%E4%BE%A0%E5%A4%A9) )
* [{mxnet} R package from MXnet, an intuitive Deep Learning framework including CNN & RNN] (http://tjo-en.hatenablog.com/entry/2016/03/30/233848) by [TJO](http://tjo-en.hatenablog.com/)
* [MXnet with R: combined power of deep learning](http://cos.name/2016/04/mxnet-r/) in Chinese by Tong He

## <a name="winning-solutions"></a>Machine Learning Challenge Winning Solutions

* Dmitrii Tsybulevskii, 1st place of the [Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification). Link to [the Kaggle interview](http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/).

## <a name="tools-with-mxnet"></a>Tools with MXnet
* [TensorFuse](https://github.com/dementrock/tensorfuse) - Common interface for Theano, CGT, TensorFlow, and mxnet (experimental) by [dementrock](https://github.com/dementrock)
* [MXnet-face](https://github.com/tornadomeet/mxnet-face) - Using mxnet for face-related algorithm by [tornadomeet](https://github.com/tornadomeet) where the single model get 97.13%+-0.88% accuracy on LFW, and with only 20MB size.
* [MinPy](https://github.com/dmlc/minpy) - Pure numpy practice with third party operator Integration and MXnet as backend for GPU computing
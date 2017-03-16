# Tutorials

The tutorials provide step-by-step instructions for creating models for specific types of applications.

They explain what each step accomplishes and why, and some include related information to enrich your understanding. Instead of procedures for creating your own model, some of the tutorials provide fully trained models or examples of models that you can use to create your own output.

## Examples

These tutorials cover use cases and applications of MXNet, organized by type and by language. To create your own applications, you can customize the commands provided.

### Python

#### Basics

The following tutorials walk you through the basic usage of MXNet, including manipulating arrays, building networks, loading and preprocessing data, etc.

- [CPU/GPU Array Manipulation](http://mxnet.io/tutorials/python/ndarray.html)
*How to use `mxnet.ndarray` (similar to numpy array but supports GPU) for tensor computation. Also explains MXNet's powerful automatic parallelization feature.*

- [Neural Network Graphs](http://mxnet.io/tutorials/python/symbol.html)
*How to use `mxnet.symbol` for building neural network graphs. Introduces usage of basic operators (layers) and shows how to build new ones.*

- [Training and Inference with Module](http://mxnet.io/tutorials/python/module.html)
*Train a simple deep neural network with the Module interface. The Module package provides intermediate-level and high-level interface for executing predefined networks.*

- [Mixing Array and Graphs (Advanced)](http://mxnet.io/tutorials/python/mixed.html)
*Show cases MXNet's signature support for mixing imperative and symbolic programming. Note that Module already provides a high-level interface by wrapping around these functionalities. So this tutorial is mainly for users who want to build things from scratches for extra flexibility.*

#### IO

- [Data Loading](http://mxnet.io/tutorials/python/data.html)
*How to write a data iterator to feed your custom data to Module (and other interfaces)*

- [Image IO](http://mxnet.io/tutorials/python/image_io.html)
*How to prepare, load, and pre-process images for training image classification networks.*

- [Record IO (Advanced)](http://mxnet.io/tutorials/python/record_io.html)
*How to pack free format data into a single binary file using MXNet's RecordIO format for easy and efficient IO.*

- [Custom Image IO (Advanced)](http://mxnet.io/tutorials/python/advanced_img_io.html)
*How to use `mxnet.image` package to easily write high performance and flexible data pipeline. `mxnet.image` uses MXNet's dependency engine to bypass python's slowness so you don't have to use multiprocessing.*

#### Computer Vision

The following tutorials explain how to develop applications that use machine learning to modify, classify, and segment images and video.

- [Handwritten Digit Classification](http://mxnet.io/tutorials/python/mnist.html)
*A simple example of classifying handwritten digits from the MNIST dataset using an MLP and convolutional network*

- [Image Classification](http://mxnet.io/tutorials/computer_vision/image_classification.html)
*An example of classifying various images of real-world objects using a convolutional neural network.*

- [Image Segmentation](http://mxnet.io/tutorials/computer_vision/segmentation.html)
*An example of segmenting various object out of an image using a convolutional neural network.*

- [Object Detection using Faster R-CNN](http://mxnet.io/tutorials/computer_vision/detection.html)
*An example of detecting object bounding boxes in an image using a region proposal network.*

- [Neural Art: Adding Artistic Style to Images](http://mxnet.io/tutorials/computer_vision/neural_art.html)
*An example of transferring styles of famous artists onto an image using a convolutional neural network.*

- [Large Scale Image Classification: Training With 14 Million Images on a Single Machine](http://mxnet.io/tutorials/computer_vision/imagenet_full.html)
*An advanced example of training a deep convolutional network on the large ImageNet dataset efficiently.*

- [Classifying Real-World Images With a Pre-Trained Model](http://mxnet.io/tutorials/python/predict_imagenet.html)
*An advanced example of using a large pre-trained model to classify the ImageNet data set*

#### Natural Language Processing

The following tutorials explain how to develop applications that understand, generate, and summarize text-based data.

- [Character Level LSTM](http://mxnet.io/tutorials/python/char_lstm.html)
*An example using an LSTM network to generate text, character by character, in the style of Barack Obama's speeches*

- [Text Classification using Convolutional Neural Network](http://mxnet.io/tutorials/nlp/cnn.html)
*An example of using a convolutional network to classify sentiment in text reviews*

- [NCE Loss](http://mxnet.io/tutorials/nlp/nce_loss.html)
*An advanced example of using NCE loss to speed up text classification with an LSTM model*

#### Speech Recognition

The following tutorials explain how to develop applications that map natural speech to text.

- [Speech LSTM](http://mxnet.io/tutorials/speech_recognition/speech_lstm.html)
*An example of training an LSTM acoustic model on the TIMIT dataset to recognize speech*

- [Baidu Warp CTC](http://mxnet.io/tutorials/speech_recognition/baidu_warp_ctc.html)
*An advanced example to training an LSTM to recognize speech with Baidu's implementation of the Connectionist Temporal Classification loss function*

#### Generative Networks

The following tutorial explains how to develop applications that generate content as data sets, such as images, text, music, and more.

- [Generative Adversarial Network](http://mxnet.io/tutorials/unsupervised_learning/gan.html)
*An example of using a GAN trained on the MNIST dataset on generating handwritten digits*

#### Unsupervised Machine Learning

The following tutorials explain how to develop applications for discovering existing structures and relationships in datasets.

- [Matrix Factorization](http://mxnet.io/tutorials/python/matrix_factorization.html)
*An example using matrix factorization to discover user preferences in the MovieLens dataset*

- [Auto Encoders](http://mxnet.io/tutorials/unsupervised_learning/auto_encoders.html)
*An example using a non-linear deep autoencoder to find low-dimensional representations for the MNIST dataset*

- [Recommendation Systems](http://mxnet.io/tutorials/general_ml/recommendation_systems.html)
*An example of using an autoencoder and matrix factorization to make a complete end to end recommendation system*

#### Visualization

The following tutorials show how visualization helps us in daily work, and better understanding towards neural network.

- [Understanding the vanishing gradient problem through visualization](https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/understanding_vanish_gradient.ipynb)
*An example of using the visualization component, [TensorBoard](https://github.com/dmlc/tensorboard), to have an intuitive understanding on the well-known vanished gradient problem in deep learning.*

#### Embedded

The following tutorials explain how to develop machine learning applications running on embedded devices, such as the Raspberry Pi

- [Raspberry Pi Object Classifier](http://mxnet.io/tutorials/embedded/wine_detector.html)
*An example using a Raspberry Pi equipped with the standard camera module to identify objects in real-time using a pretrained imageNet model*

### R

#### Computer Vision

The following tutorials explain how to develop applications that use machine learning to modify, classify, and segment images and video.

- [Handwritten Digit Classification](http://mxnet.io/tutorials/r/mnistCompetition.html)
*An example of classifying handwritten digits from the MNIST dataset using an MLP and convolutional network*

- [Classify Real-World Images with Pre-trained Model](http://mxnet.io/tutorials/r/classifyRealImageWithPretrainedModel.html)
*An advanced example of using a large pre-trained model to classify the ImageNet dataset*

- [Dogs vs. Cats classification with mxnet and R](https://statist-bhfz.github.io/cats_dogs_finetune) *End-to-end tutorial with an example of fine-tuning in R*
([source RMD](https://github.com/dmlc/mxnet/tree/master/docs/tutorials/r/CatsDogsFinetune.rmd))


#### Natural Language Processing

The following tutorials explain how to develop applications that understand, generate, and summarize text-based data.

- [Character Language Model using RNN](http://mxnet.io/tutorials/r/charRnnModel.html)
*An example using an LSTM network to generate text, character by character, in the style of Shakespeare*


#### Supervised Machine Learning

Applications that use traditional methods to model classification and regression problems.

- [Neural Networks with MXNet in Five Minutes](http://mxnet.io/tutorials/r/fiveMinutesNeuralNetwork.html)
*Using a multi-layer perceptron to do classification and regression tasks on the mlbench dataset*

### Scala

#### Get Started
- [MXNet Scala from IntelliJ](http://mxnet.io/tutorials/scala/mxnet_scala_on_intellij.html)
*How to create MXNet Scala examples or applications with the IntelliJ IDE.*

#### Applications
- [Handwritten Digit Classification](http://mxnet.io/tutorials/scala/mnist.html)
*A simple example of classifying handwritten digits from the MNIST dataset using a multilayer perceptron.*

- [Character Level LSTM](http://mxnet.io/tutorials/scala/char_lstm.html)
*An example using an LSTM network to generate text, character by character, in the style of Barack Obama's speeches*

## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, download the [tutorial template](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).

## Other Resources
- [MXNet Code Examples](https://github.com/dmlc/mxnet/tree/master/example)
- [MXNet Tutorials for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)

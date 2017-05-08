# Tutorials

These tutorials introduce fundamental concepts in deep learning and their realizations in _MXNet_. Under the _basics_ section, you'll find tutorials covering manipulating arrays, building networks, loading and preprocessing data, etc. Further sections introduce fundamental models for image classification, natural language processing, speech recognition, and unsupervised learning. While most tutorials are currently presented in Python, we also present a subset of tutorials using the R and Scala front ends.


## Python

### Basics

```eval_rst
.. toctree::
   :maxdepth: 1

   basic/ndarray
   basic/symbol
   basic/module
   basic/data
```

### Computer Vision

```eval_rst
.. toctree::
   :maxdepth: 1

   python/mnist
   python/predict_image
```

- [Object Detection using Faster R-CNN](https://github.com/dmlc/mxnet/tree/master/example/rcnn)

- [Object Detection using SSD](https://github.com/dmlc/mxnet/tree/master/example/ssd)

- [Neural Art - Transfer the style of one image onto the content the content of another image](https://github.com/dmlc/mxnet/tree/master/example/neural-style)

- [Large Scale Image Classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification)

### Natural Language Processing

- [Character-Level LSTM - Generate new text, one character at a time](http://mxnet.io/tutorials/python/char_lstm.html)

- [Text Classification using Convolutional Neural Networks](http://mxnet.io/tutorials/nlp/cnn.html)

- [NCE Loss - Speed up text classification with large output layers](http://mxnet.io/tutorials/nlp/nce_loss.html)

### Speech Recognition

- [Phoneme Classification - Use LSTM recurrent nets to recognize phonemes in audio](http://mxnet.io/tutorials/speech_recognition/speech_lstm.html)

- [Baidu Warp CTC - Jointly learn predictions and alignments with CTC loss](http://mxnet.io/tutorials/speech_recognition/baidu_warp_ctc.html)

### Unsupervised Learning and Generative Modeling

- [Generative Adversarial Networks](http://mxnet.io/tutorials/unsupervised_learning/gan.html)

- [Autoencoders - Find low dimensional representations of data](http://mxnet.io/tutorials/unsupervised_learning/auto_encoders.html)

- [Matrix Factorization - Discover latent factors of user preference in MovieLens data](http://mxnet.io/tutorials/python/matrix_factorization.html)

- [Recommender Systems - Build a complete recommender system with matrix factorization](http://mxnet.io/tutorials/general_ml/recommendation_systems.html)


## R

- [Neural Networks with MXNet in Five Minutes](http://mxnet.io/tutorials/r/fiveMinutesNeuralNetwork.html)

- [Classifying Handwritten Digits with Convolutional Neural Networks](http://mxnet.io/tutorials/r/mnistCompetition.html)

- [Classify Real-world Images with a Pre-trained Model](http://mxnet.io/tutorials/r/classifyRealImageWithPretrainedModel.html)

- [Dogs vs. Cats Classification with Fine-tuning](https://statist-bhfz.github.io/cats_dogs_finetune)

- [Character-Level Language Modeling with LSTM RNNs](http://mxnet.io/tutorials/r/charRnnModel.html)


## Scala

- [Create MXNet Scala Applications with the IntelliJ IDE](http://mxnet.io/tutorials/scala/mxnet_scala_on_intellij.html)

- [Handwritten Digit Classification with Multilayer Perceptrons](http://mxnet.io/tutorials/scala/mnist.html)

- [Character-Level Language Modeling with LSTM RNNs](http://mxnet.io/tutorials/scala/char_lstm.html)

## C++

- [Basics](http://mxnet.io/tutorials/c++/basics.html)

## Perl

- [Calculator, handwritten digits and roboshakespreare](http://blogs.perl.org/users/sergey_kolychev/2017/04/machine-learning-in-perl-part2-a-calculator-handwritten-digits-and-roboshakespeare.html)

## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, download the [tutorial template](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).

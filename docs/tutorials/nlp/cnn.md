# Text Classification Using a Convolutional Neural Network on MXNet

This is a slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in MXNet. You can get the source code for this example on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/cnn_text_classification).

In learning MXNet for Natural Language Processing (NLP), I followed this  blog ["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) to reimplement a convolutional neural network using the MXNet framwork.
I borrowed the data preprocessing code and corpus from the original author [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).

## Performance Comparison
I used the same pretrained word2vec [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) in Kim's paper. However, I didn't implement L2-normalization of weight on the penultimate layer, but did provide L2-normalization of gradients. I got a best dev accuracy score of 80.1%, which is close to the score of 81% reported in the original paper.

## Download and Train the Data 

1. Download the corpus from this repository: [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).

2. Use the GoogleNews word2vec tool to train 'data/rt.vec' on the corpus. I recommend using GoogleNews word2vec  because 
this corpus is small (contains about 10 Kb sentences). When using GoogleNews word2vec, this code loads it with gensim tools [gensim](https://github.com/piskvorky/gensim/tree/develop/gensim/models).


## References
- ["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
Implementing CNN for Text Classification in MXNet
============
It is a slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in MXNet.

Recently, I have been learning mxnet for Natural Language Processing (NLP). I followed this nice blog ["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) to reimplement it by mxnet framework.
Data preprocessing code and corpus are directly borrowed from original author [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf).

## Performance compared to original paper
I use the same pretrained word2vec [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) in Kim's paper. However, I don't implement L2-normalization of weights on penultimate layer, but provide a L2-normalization of gradients.
Finally, I got a best dev accuracy 80.1%, close to 81% that reported in the original paper.

## Data
Please download the corpus from this repository [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), :)

'data/rt.vec', this file was trained on the corpus by word2vec tool. I recommend to use GoogleNews word2vec, which could get better performance, since
this corpus is small (contains about 10K sentences).

When using GoogleNews word2vec, this code loads it with gensim tools [gensim](https://github.com/piskvorky/gensim/tree/develop/gensim/models).

## Remark
If I were wrong in CNN implementation via mxnet, please correct me.

## References
- ["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)


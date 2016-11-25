## Word2Vec in NCE-loss with Subword Representation

Reproducing the work Microsoft Research presented in CIKM'14, in which it's a basis of DSSM([Deep Semantics Similarity Model](https://www.microsoft.com/en-us/research/project/dssm/)), you can get its lectures [here](https://www.microsoft.com/en-us/research/publication/deep-learning-for-natural-language-processing-theory-and-practice-tutorial/).


### Motivation

The motivation is to design a more robust and scalable word vector system, by reducing the size of lookup-table, and handle unknown words(out-of-vocabulary) better.

 * Handle out-of-vocabulary.
 * Embedding lookup table size is dramatically reduced.

### Basics

<img src="https://github.com/zihaolucky/mxnet/blob/example/word2vec-nce-loss-with-subword-representations/example/nce-loss-subword-repr/slide1.png" width="700">

<img src="https://github.com/zihaolucky/mxnet/blob/example/word2vec-nce-loss-with-subword-representations/example/nce-loss-subword-repr/slide2.png" width="700">

Note that this word embedding method uses sub-word units to represent a word, while we still train word2vec model in its original way, the only difference is the vector representation of a word is no longer the word itself, but use several sub-word units' addition.

If you use sub-word sequence and feed into a word2vec training processing, it could not have the property we want to have in original word2vec method.

### Analysis

Here we print the training/validation log below, using text8 data, to get some intuitions on its benefits:

TODO.




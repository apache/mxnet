#Examples of NCE Loss

nce-loss is used to speedup multi-class classification when class num is huge.

## Toy example

* toy_softmax.py: a multi class example using softmax output
* toy_nce.py: a multi-class example using nce loss

## Word2Vec

* word2vec.py: a CBOW word2vec example using nce loss

You can run it by

```
./get_text8.sh
python word2vec.py

```

## LSTM

* lstm_word.py: a lstm example use nce loss

You can run it by

```
./get_text8.sh
python lstm_word.py
```

## References

You can refer to [http://www.jianshu.com/p/e439b43ea464](http://www.jianshu.com/p/e439b43ea464) for more details. (In Chinese)


## Word2Vec in NCE-loss with Subword Representation

```
./get_text8.sh
python wordvec_subwords.py
```

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

> Experiment data on MacBook Pro'16 with 4 cpus.

Here we print the training log below, using text8 data, to get some intuitions on its benefits:

*With subword units representation*

It converge much faster.

```
2016-11-26 19:07:31,742 Start training with [cpu(0), cpu(1), cpu(2), cpu(3)]
2016-11-26 19:07:31,783 DataIter start.
2016-11-26 19:07:45,099 Epoch[0] Batch [50]		Speed: 4020.37 samples/sec	Train-nce-auc=0.693178
2016-11-26 19:07:57,870 Epoch[0] Batch [100]	Speed: 4009.19 samples/sec	Train-nce-auc=0.741482
2016-11-26 19:08:10,196 Epoch[0] Batch [150]	Speed: 4153.73 samples/sec	Train-nce-auc=0.764026
2016-11-26 19:08:22,497 Epoch[0] Batch [200]	Speed: 4162.61 samples/sec	Train-nce-auc=0.785248
2016-11-26 19:08:34,708 Epoch[0] Batch [250]	Speed: 4192.69 samples/sec	Train-nce-auc=0.782754
2016-11-26 19:08:47,060 Epoch[0] Batch [300]	Speed: 4145.31 samples/sec	Train-nce-auc=0.773067
2016-11-26 19:08:59,696 Epoch[0] Batch [350]	Speed: 4051.81 samples/sec	Train-nce-auc=0.799248
2016-11-26 19:09:12,540 Epoch[0] Batch [400]	Speed: 3986.58 samples/sec	Train-nce-auc=0.806501
2016-11-26 19:09:25,874 Epoch[0] Batch [450]	Speed: 3839.86 samples/sec	Train-nce-auc=0.808727
2016-11-26 19:09:38,223 Epoch[0] Batch [500]	Speed: 4145.97 samples/sec	Train-nce-auc=0.816675
2016-11-26 19:09:50,929 Epoch[0] Batch [550]	Speed: 4029.56 samples/sec	Train-nce-auc=0.809086
2016-11-26 19:10:03,502 Epoch[0] Batch [600]	Speed: 4072.32 samples/sec	Train-nce-auc=0.827757
2016-11-26 19:10:16,070 Epoch[0] Batch [650]	Speed: 4073.90 samples/sec	Train-nce-auc=0.826251
2016-11-26 19:10:28,317 Epoch[0] Batch [700]	Speed: 4180.91 samples/sec	Train-nce-auc=0.834578
2016-11-26 19:10:40,945 Epoch[0] Batch [750]	Speed: 4054.38 samples/sec	Train-nce-auc=0.830133
2016-11-26 19:10:53,362 Epoch[0] Batch [800]	Speed: 4123.59 samples/sec	Train-nce-auc=0.834170
2016-11-26 19:11:05,645 Epoch[0] Batch [850]	Speed: 4168.32 samples/sec	Train-nce-auc=0.836135
2016-11-26 19:11:18,035 Epoch[0] Batch [900]	Speed: 4132.51 samples/sec	Train-nce-auc=0.842253
2016-11-26 19:11:30,257 Epoch[0] Batch [950]	Speed: 4189.27 samples/sec	Train-nce-auc=0.834119
2016-11-26 19:11:42,600 Epoch[0] Batch [1000]	Speed: 4148.01 samples/sec	Train-nce-auc=0.828049
2016-11-26 19:11:54,850 Epoch[0] Batch [1050]	Speed: 4179.55 samples/sec	Train-nce-auc=0.844856
2016-11-26 19:12:07,052 Epoch[0] Batch [1100]	Speed: 4196.35 samples/sec	Train-nce-auc=0.856587
2016-11-26 19:12:19,286 Epoch[0] Batch [1150]	Speed: 4185.10 samples/sec	Train-nce-auc=0.845370
2016-11-26 19:12:31,703 Epoch[0] Batch [1200]	Speed: 4123.25 samples/sec	Train-nce-auc=0.851430
2016-11-26 19:12:44,177 Epoch[0] Batch [1250]	Speed: 4104.76 samples/sec	Train-nce-auc=0.851357
2016-11-26 19:12:56,497 Epoch[0] Batch [1300]	Speed: 4155.90 samples/sec	Train-nce-auc=0.854957
2016-11-26 19:13:08,839 Epoch[0] Batch [1350]	Speed: 4148.39 samples/sec	Train-nce-auc=0.853684
2016-11-26 19:13:21,052 Epoch[0] Batch [1400]	Speed: 4192.37 samples/sec	Train-nce-auc=0.849442
2016-11-26 19:13:33,386 Epoch[0] Batch [1450]	Speed: 4151.24 samples/sec	Train-nce-auc=0.853365
2016-11-26 19:13:45,709 Epoch[0] Batch [1500]	Speed: 4154.65 samples/sec	Train-nce-auc=0.855938
```


*With word level representation*

```
2016-11-26 19:27:01,998 Start training with [cpu(0), cpu(1), cpu(2), cpu(3)]
2016-11-26 19:27:35,422 Epoch[0] Batch [50]	Speed: 1597.90 samples/sec	Train-nce-auc=0.552027
2016-11-26 19:28:06,299 Epoch[0] Batch [100]	Speed: 1658.24 samples/sec	Train-nce-auc=0.590524
2016-11-26 19:28:36,483 Epoch[0] Batch [150]	Speed: 1696.26 samples/sec	Train-nce-auc=0.625941
2016-11-26 19:29:07,379 Epoch[0] Batch [200]	Speed: 1657.18 samples/sec	Train-nce-auc=0.645201
2016-11-26 19:29:38,010 Epoch[0] Batch [250]	Speed: 1671.56 samples/sec	Train-nce-auc=0.643815
2016-11-26 19:30:09,533 Epoch[0] Batch [300]	Speed: 1624.20 samples/sec	Train-nce-auc=0.645837
2016-11-26 19:30:41,373 Epoch[0] Batch [350]	Speed: 1608.08 samples/sec	Train-nce-auc=0.645352
2016-11-26 19:31:12,989 Epoch[0] Batch [400]	Speed: 1619.66 samples/sec	Train-nce-auc=0.645995
2016-11-26 19:31:44,920 Epoch[0] Batch [450]	Speed: 1603.50 samples/sec	Train-nce-auc=0.641189
2016-11-26 19:32:16,419 Epoch[0] Batch [500]	Speed: 1625.49 samples/sec	Train-nce-auc=0.655360
2016-11-26 19:32:48,491 Epoch[0] Batch [550]	Speed: 1596.41 samples/sec	Train-nce-auc=0.648425
2016-11-26 19:33:19,620 Epoch[0] Batch [600]	Speed: 1644.78 samples/sec	Train-nce-auc=0.650669
2016-11-26 19:33:50,795 Epoch[0] Batch [650]	Speed: 1642.39 samples/sec	Train-nce-auc=0.661544
2016-11-26 19:34:25,131 Epoch[0] Batch [700]	Speed: 1491.14 samples/sec	Train-nce-auc=0.655027
2016-11-26 19:34:58,433 Epoch[0] Batch [750]	Speed: 1537.49 samples/sec	Train-nce-auc=0.659898
2016-11-26 19:35:32,100 Epoch[0] Batch [800]	Speed: 1520.78 samples/sec	Train-nce-auc=0.661189
2016-11-26 19:36:06,080 Epoch[0] Batch [850]	Speed: 1506.81 samples/sec	Train-nce-auc=0.668111
2016-11-26 19:36:40,387 Epoch[0] Batch [900]	Speed: 1492.40 samples/sec	Train-nce-auc=0.662804
```


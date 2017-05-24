# MXNet Model Zoo

MXNet features fast implementations of many state-of-the-art models reported in the academic literature. This Model Zoo is an
ongoing project to collect complete models, with python scripts, pre-trained weights as well as instructions on how to build and fine tune these models.

## How to Contribute a Pre-Trained Model (and what to include)

The Model Zoo has good entries for CNNs but is seeking content in other areas.

Issue a Pull Request containing the following:
* Gist Log
* .json model definition
* Model parameter file
* Readme file (details below)

Readme file should contain:
* Model Location, access instructions (wget)
* Confirmation the trained model meets published accuracy from original paper
* Step by step instructions on how to use the trained model
* References to any other applicable docs or arxiv papers the model is based on

## Convolutional Neural Networks (CNNs)

Convolutional neural networks are the state-of-art architecture for many image and video processing problems. Some available datasets include:

* [ImageNet](http://image-net.org/): a large corpus of 1 million natural images, divided into 1000 categories.
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 natural images (32 x 32 pixels) from 10 categories.
* [PASCAL_VOC](http://host.robots.ox.ac.uk/pascal/VOC/): A subset of ImageNet images with object bounding boxes.
* [UCF101](http://crcv.ucf.edu/data/UCF101.php): 13,320 videos from 101 action categories.
* [Mini-Places2](http://6.869.csail.mit.edu/fa15/project.html): Subset of the Places2 dataset. Includes 100,000 images from 100 scene categories.
* ImageNet 11k
* [Places2](http://places2.csail.mit.edu/download.html): There are 1.6 million train images from 365 scene categories in the Places365-Standard, which are used to train the Places365 CNNs. There are 50 images per category in the validation set and 900 images per category in the testing set. Compared to the train set of Places365-Standard, the train set of Places365-Challenge has 6.2 million extra images, leading to totally 8 million train images for the Places365 challenge 2016. The validation set and testing set are the same as the Places365-Standard.
* [Multimedia Commons](https://aws.amazon.com/public-datasets/multimedia-commons/): YFCC100M (99.2 million images and 0.8 million videos from Flickr) and supplemental material (pre-extracted features, additional annotations).

For instructions on using these models, see [the python tutorial on using pre-trained ImageNet models](http://mxnet.io/tutorials/python/predict_imagenet.html).

| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| [CaffeNet](http://data.dmlc.ml/mxnet/models/imagenet/caffenet/caffenet-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/caffenet/caffenet-0000.params) |   [Krizhevsky, 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | @jspisak |
| [Network in Network (NiN)](http://data.dmlc.ml/models/imagenet/nin/nin-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/nin/nin-0000.params) |  [Lin et al.., 2014](https://arxiv.org/pdf/1312.4400v3.pdf) | @jspisak |
| [SqueezeNet v1.1](http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1-0000.params) | [Iandola et al.., 2016](https://arxiv.org/pdf/1602.07360v4.pdf) | @jspisak |
| [VGG16](http://data.dmlc.ml/models/imagenet/vgg/vgg16-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params)| [Simonyan et al.., 2015](https://arxiv.org/pdf/1409.1556v6.pdf) | @jspisak |
| [VGG19](http://data.dmlc.ml/models/imagenet/vgg/vgg19-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/vgg/vgg19-0000.params) | [Simonyan et al.., 2015](https://arxiv.org/pdf/1409.1556v6.pdf) | @jspisak |
| [Inception v3 w/BatchNorm](http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params) | [Szegedy et al.., 2015](https://arxiv.org/pdf/1512.00567.pdf) | @jspisak |
| [ResidualNet152](http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params) | [He et al.., 2015](https://arxiv.org/pdf/1512.03385v1.pdf) | @jspisak |
| [ResNext101-64x4d](http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json) | ImageNet | [Param File](http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-0000.params) | [Xie et al.., 2016](https://arxiv.org/pdf/1611.05431.pdf) | @Jerryzcn |
| Fast-RCNN | PASCAL VOC | [Param File] | [Girshick, 2015](https://arxiv.org/pdf/1504.08083v2.pdf) | |
| Faster-RCNN | PASCAL VOC | [Param File] | [Ren et al..,2016](https://arxiv.org/pdf/1506.01497v3.pdf) | |
| Single Shot Detection (SSD) | PASCAL VOC | [Param File] | [Liu et al.., 2016](https://arxiv.org/pdf/1512.02325v4.pdf) | |
| [LocationNet](https://s3.amazonaws.com/mmcommons-tutorial/models/RN101-5k500-symbol.json) | [MultimediaCommons](https://aws.amazon.com/public-datasets/multimedia-commons/) | [Param File](https://s3.amazonaws.com/mmcommons-tutorial/models/RN101-5k500-0012.params) | [Weyand et al.., 2016](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf) | @jychoi84 @kevinli7 |



## Recurrent Neural Networks (RNNs) including LSTMs

MXNet supports many types of recurrent neural networks (RNNs), including Long Short-Term Memory ([LSTM](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf))
and Gated Recurrent Units (GRU) networks. Some available datasets include:

* [Penn Treebank (PTB)](https://www.cis.upenn.edu/~treebank/): Text corpus with ~1 million words. Vocabulary is limited to 10,000 words. The task is predicting downstream words/characters.
* [Shakespeare](http://cs.stanford.edu/people/karpathy/char-rnn/): Complete text from Shakespeare's works.
* [IMDB reviews](https://s3.amazonaws.com/text-datasets): 25,000 movie reviews, labeled as positive or negative
* [Facebook bAbI](https://research.facebook.com/researchers/1543934539189348): As a set of 20 question & answer tasks, each with 1,000 training examples.
* [Flickr8k, COCO](http://mscoco.org/): Images with associated caption (sentences). Flickr8k consists of 8,092 images captioned by AmazonTurkers with ~40,000 captions. COCO has 328,000 images, each with 5 captions. The COCO images also come with labeled objects using segmentation algorithms.


| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| LSTM - Image Captioning | Flickr8k, MS COCO | | [Vinyals et al.., 2015](https://arxiv.org/pdf/ 1411.4555v2.pdf) | @... |
| LSTM - Q&A System| bAbl | | [Weston et al.., 2015](https://arxiv.org/pdf/1502.05698v10.pdf) | |
| LSTM - Sentiment Analysis| IMDB | | [Li et al.., 2015](http://arxiv.org/pdf/1503.00185v5.pdf) | |


## Generative Adversarial Networks (GANs)

[Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) train a competing pair of
neural networks: a generator network which transforms a latent vector into content like an image, and a discriminator
network that tries to distinguish between generated content and supplied "real" training content.  When properly
trained the two achieve a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium).

| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| DCGANs | ImageNet | | [Radford et al..,2016](https://arxiv.org/pdf/1511.06434v2.pdf) | @... |
| Text to Image Synthesis |MS COCO| | [Reed et al.., 2016](https://arxiv.org/pdf/1605.05396v2.pdf) | |
| Deep Jazz	| | | [Deepjazz.io](https://deepjazz.io) | |



## Other Models

MXNet Supports a variety of model types beyond the canonical CNN and LSTM model types. These include deep reinforcement learning, linear models, etc.. Some available datasets and sources include:

* [Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit): A text corpus with a vocabulary of 3 million words architected for word2vec.
* [MovieLens 20M Dataset](http://grouplens.org/datasets/movielens/): 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
* [Atari Video Game Emulator](http://stella.sourceforge.net/): Stella is a multi-platform Atari 2600 VCS emulator released under the GNU General Public License (GPL).


| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| Word2Vec | Google News | | [Mikolov et al.., 2013](https://arxiv.org/pdf/1310.4546v1.pdf) | @... |
| Matrix Factorization | MovieLens 20M | | [Huang et al.., 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) | |
| Deep Q-Network | Atari video games | | [Minh et al.., 2015](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) | |
| Asynchronous advantage actor-critic (A3C) | Atari video games | | [Minh et al.., 2016](https://arxiv.org/pdf/1602.01783.pdf) | |

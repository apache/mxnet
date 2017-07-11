
GetMNIST_ubyte <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/train-images-idx3-ubyte') |
      !file.exists('data/train-labels-idx1-ubyte') |
      !file.exists('data/t10k-images-idx3-ubyte') |
      !file.exists('data/t10k-labels-idx1-ubyte')) {
    download.file('http://data.mxnet.io/mxnet/data/mnist.zip', destfile = 'data/mnist.zip')
    unzip('data/mnist.zip', exdir = 'data/')
  }
}

GetMNIST_csv <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/train.csv') |
      !file.exists('data/test.csv')) {
    download.file('https://s3-us-west-2.amazonaws.com/apache-mxnet/R/data/mnist_csv.zip',
                  destfile = 'data/mnist_csv.zip')
    unzip('data/mnist_csv.zip', exdir = 'data/')
  }
}

GetCifar10 <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/cifar/train.rec') |
      !file.exists('data/cifar/test.rec') |
      !file.exists('data/cifar/train.lst') |
      !file.exists('data/cifar/test.lst')) {
    download.file('http://data.mxnet.io/mxnet/data/cifar10.zip',
                  destfile = 'data/cifar10.zip')
    unzip('data/cifar10.zip', exdir = 'data/')
  }
}

GetInception <- function() {
  if (!dir.exists("model")) {
    dir.create("model/")
  }
  if (!file.exists('model/Inception-BN-0126.params')) {
    download.file('http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params',
                  destfile = 'model/Inception-BN-0126.params')
  }
  if (!file.exists('model/Inception-BN-symbol.json')) {
    download.file('http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json',
                  destfile = 'model/Inception-BN-symbol.json')
  }
}

GetCatDog <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/cats_dogs/cats_dogs_train.rec') |
      !file.exists('data/cats_dogs/cats_dogs_val.rec')) {
    download.file('https://s3-us-west-2.amazonaws.com/apache-mxnet/R/data/cats_dogs.zip',
                  destfile = 'data/cats_dogs.zip')
    unzip('data/cats_dogs.zip', exdir = 'data/')
  }
}

GetMovieLens <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/ml-100k/u.data')) {
    download.file('http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                  destfile = 'data/ml-100k.zip')
    unzip('data/ml-100k.zip', exdir = 'data/')
  }
}


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
    file.remove('data/mnist.zip')
  }
}

GetMNIST_csv <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/train.csv') |
      !file.exists('data/test.csv')) {
    download.file('https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip',
                  destfile = 'data/mnist_csv.zip')
    unzip('data/mnist_csv.zip', exdir = 'data/')
    file.remove('data/mnist_csv.zip')
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
    file.remove('data/cifar10.zip')
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
    download.file('https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/cats_dogs.zip',
                  destfile = 'data/cats_dogs.zip')
    unzip('data/cats_dogs.zip', exdir = 'data/')
    file.remove('data/cats_dogs.zip')
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
    file.remove('data/ml-100k.zip')
  }
}

GetISBI_data <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/ISBI/train-volume.tif') |
      !file.exists('data/ISBI/train-labels.tif')) {
    download.file('https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/ISBI.zip',
                  destfile = 'data/ISBI.zip')
    unzip('data/ISBI.zip', exdir = 'data/')
    file.remove('data/ISBI.zip')
  }
}

GetCaptcha_data <- function() {
  if (!dir.exists("data")) {
    dir.create("data/")
  }
  if (!file.exists('data/captcha_example/captcha_train.rec') |
      !file.exists('data/captcha_example/captcha_test.rec')) {
    download.file('https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/captcha_example.zip',
                  destfile = 'data/captcha_example.zip')
    unzip('data/captcha_example.zip', exdir = 'data/')
    file.remove('data/captcha_example.zip')
  }
}

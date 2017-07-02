
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

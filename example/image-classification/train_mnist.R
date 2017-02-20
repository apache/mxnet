require(argparse)
require(mxnet)

get_data <- function(data_dir) {
  if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)
  cwd <- getwd()
  print(paste("Current working directory:", cwd))
  print(paste("Data directory:", data_dir))
  if ((!file.exists(paste0(data_dir,"train-images-idx3-ubyte"))) ||
      (!file.exists(paste0(data_dir,"train-labels-idx1-ubyte"))) ||
      (!file.exists(paste0(data_dir,"t10k-images-idx3-ubyte" ))) ||
      (!file.exists(paste0(data_dir,"t10k-labels-idx1-ubyte" )))  ) {
    setwd(data_dir)
    download.file(url='http://data.mxnet.io/mxnet/data/mnist.zip',
                  destfile='mnist.zip', method='wget')
    unzip("mnist.zip", junkpaths = T)  # unzip only files not dirs
    file.remove("mnist.zip")
    setwd(cwd)
  }
}

# multi-layer perceptron
get_mlp <- function() {
    data <- mx.symbol.Variable('data')
    fc1  <- mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 <- mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  <- mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 <- mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  <- mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  <- mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    mlp
}

# LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
# Haffner. "Gradient-based learning applied to document recognition."
# Proceedings of the IEEE (1998)
get_lenet <- function() {
    data <- mx.symbol.Variable('data')
    # first conv
    conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
    tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                               kernel=c(2,2), stride=c(2,2))
    # second conv
    conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
    tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=c(2,2), stride=c(2,2))
    # first fullc
    flatten <- mx.symbol.Flatten(data=pool2)
    fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet <- mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    lenet
}

get_iterator <- function(data_shape) {
    flat <- TRUE
    if (length(data_shape) == 3) flat <- FALSE

    train           = mx.io.MNISTIter(
        image       = paste0(args$data_dir, "train-images-idx3-ubyte"),
        label       = paste0(args$data_dir, "train-labels-idx1-ubyte"),
        input_shape = data_shape,
        batch_size  = args$batch_size,
        shuffle     = TRUE,
        flat        = flat)

    val = mx.io.MNISTIter(
        image       = paste0(args$data_dir, "t10k-images-idx3-ubyte"),
        label       = paste0(args$data_dir, "t10k-labels-idx1-ubyte"),
        input_shape = data_shape,
        batch_size  = args$batch_size,
        flat        = flat)

    ret = list(train=train, value=val)
}

parse_args <- function() {
    parser <- ArgumentParser(description='train an image classifer on mnist')
    parser$add_argument('--network', type='character', default='mlp',
    #parser$add_argument('--network', type='character', default='lenet',
                                            choices = c('mlp', 'lenet'),
                        help = 'the cnn to use')
    parser$add_argument('--data-dir', type='character', default='data/mnist/', # needs trailing /
                        help='the input data directory (needs trailing /)')
    parser$add_argument('--gpus', type='character',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser$add_argument('--batch-size', type='integer', default=128,
                        help='the batch size')
    parser$add_argument('--lr_factor', type='double', default=.05,
                        help='the initial learning rate')
    parser$add_argument('--momentum', type='double', default=.9,
                        help='the momentum')
    parser$add_argument('--num_examples', type='double', default=60000,
                        help='the number of examples (60000 in the traning set)')
    parser$add_argument('--model-prefix', type='character',
                        help='the prefix of the model to load/save')
    parser$add_argument('--num-round', type='integer', default=10,
                        help='the number of iterations over training data to train the model')
    parser$add_argument('--kv-store', type='character', default='local',
                        help='the kvstore type')

    parser$parse_args()
}

args = parse_args()
if (args$network == 'mlp') {
    data_shape <- c(784)
    net <- get_mlp()
} else {
    data_shape <- c(28, 28, 1)
    net <- get_lenet()
}
# loda data
get_data(args$data_dir)

# train
source("train_model.R")
train_model.fit(args, net, get_iterator(data_shape))

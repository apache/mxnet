require(mxnet)

source("get_data.R")

if (Sys.getenv("R_GPU_ENABLE") != "" & as.integer(Sys.getenv("R_GPU_ENABLE")) == 1) {
  mx.ctx.default(new = mx.gpu())
  message("Using GPU for testing.")
}

print_inferred_shape <- function(net) {
  slist <- mx.symbol.infer.shape(symbol = net, data = c(168, 168, 1, 2))
  print(slist$out.shapes)
}

convolution_module <- function(net, kernel_size, pad_size, filter_count,
                               stride = c(1, 1), work_space = 2048, batch_norm = TRUE,
                               down_pool = FALSE, up_pool = FALSE, act_type = "relu",
                               convolution = TRUE) {
  if (up_pool) {
    net = mx.symbol.Deconvolution(net, kernel = c(2, 2), pad = c(0, 0),
                                  stride = c(2, 2), num_filter = filter_count,
                                  workspace = work_space)
    net = mx.symbol.BatchNorm(net)
    if (act_type != "") {
      net = mx.symbol.Activation(net, act_type = act_type)
    }
  }
  if (convolution) {
    conv = mx.symbol.Convolution(data = net, kernel = kernel_size, stride = stride,
                                 pad = pad_size, num_filter = filter_count,
                                 workspace = work_space)
    net = conv
  }
  if (batch_norm) {
    net = mx.symbol.BatchNorm(net)
  }
  
  if (act_type != "") {
    net = mx.symbol.Activation(net, act_type = act_type)
  }
  
  if (down_pool) {
    pool = mx.symbol.Pooling(net, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
    net = pool
  }
  print_inferred_shape(net)
  return(net)
}

get_unet <- function() {
  data = mx.symbol.Variable('data')
  kernel_size = c(3, 3)
  pad_size = c(1, 1)
  filter_count = 32
  pool1 = convolution_module(data, kernel_size, pad_size, filter_count = filter_count, down_pool = TRUE)
  net = pool1
  pool2 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2, down_pool = TRUE)
  net = pool2
  pool3 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, down_pool = TRUE)
  net = pool3
  pool4 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, down_pool = TRUE)
  net = pool4
  net = mx.symbol.Dropout(net)
  pool5 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 8, down_pool = TRUE)
  net = pool5
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  net = convolution_module(net, kernel_size, pad_size = c(2, 2), filter_count = filter_count * 4, up_pool = TRUE)
  net = mx.symbol.Crop(net, pool3, num.args = 2)
  net = mx.symbol.concat(c(pool3, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  
  net = mx.symbol.Concat(c(pool2, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = mx.symbol.Concat(c(pool1, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2, up_pool = TRUE)
  net = convolution_module(net, kernel_size, pad_size, filter_count = 1, batch_norm = FALSE, act_type = "")
  net = mx.symbol.SoftmaxOutput(data = net, name = 'sm')
  return(net)
}

context("Image segmentation")

test_that("UNET", {
  list.of.packages <- c("imager")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages, repos = "https://cloud.r-project.org/")
  GetISBI_data()
  library(imager)
  IMG_SIZE <- 168
  files <- list.files(path = "data/ISBI/train-volume/")
  a = 'data/ISBI/train-volume/'
  filess = paste(a, files, sep = '')
  list_of_images = lapply(filess, function(x) {
    x <- load.image(x)
    y <- resize(x, size_x = IMG_SIZE, size_y = IMG_SIZE)
  })
  
  train.x = do.call('cbind', lapply(list_of_images, as.vector))
  train.array <- train.x
  dim(train.array) <- c(IMG_SIZE, IMG_SIZE, 1, 30)
  
  files <- list.files(path = "data/ISBI/train-labels")
  b = 'data/ISBI/train-labels/'
  filess = paste(b, files, sep = '')
  list_of_images = lapply(filess, function(x) {
    x <- load.image(x)
    y <- resize(x, size_x = IMG_SIZE, size_y = IMG_SIZE)
  })
  
  train.y = do.call('cbind', lapply(list_of_images, as.vector))
  
  train.y[which(train.y < 0.5)] = 0
  train.y[which(train.y > 0.5)] = 1
  train.y.array = train.y
  dim(train.y.array) = c(IMG_SIZE, IMG_SIZE, 1, 30)
  
  devices <- mx.ctx.default()
  mx.set.seed(0)
  
  net <- get_unet()
  
  model <- mx.model.FeedForward.create(net, X = train.array, y = train.y.array,
                                       ctx = devices, num.round = 2,
                                       initializer = mx.init.normal(sqrt(2 / 576)),
                                       learning.rate = 0.05,
                                       momentum = 0.99,
                                       array.batch.size = 2)
})

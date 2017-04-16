library(mxnet)

convolution_module <- function(net, kernel_size, pad_size,
                               filter_count, stride = c(1, 1), work_space = 2048,
                               batch_norm = TRUE, down_pool = FALSE, up_pool = FALSE,
                               act_type = "relu", convolution = TRUE) {
    if (up_pool) {
      net = mx.symbol.Deconvolution(net, kernel = c(2, 2), pad = c(0, 0),
                                    stride = c(2, 2), num_filter = filter_count, workspace = work_space)
      net = mx.symbol.BatchNorm(net)
      if (act_type != "") {
        net = mx.symbol.Activation(net, act_type = act_type)
      }
    }
    if (convolution) {
      conv = mx.symbol.Convolution(data = net, kernel = kernel_size, stride = stride,
                                   pad = pad_size, num_filter = filter_count, workspace = work_space)
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
    return(net)
}

get_symbol <- function(num_classes = 10) {
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
  pool4 = convolution_module(net,
                             kernel_size,
                             pad_size,
                             filter_count = filter_count * 4,
                             down_pool = TRUE)
  net = pool4
  net = mx.symbol.Dropout(net)
  pool5 = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 8, down_pool = TRUE)
  net = pool5
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  
  # dirty "CROP" to wanted size... I was on old MxNet branch so used conv instead of crop for cropping
  net = convolution_module(net, c(4, 4), c(0, 0), filter_count = filter_count * 4)
  
  net = mx.symbol.Concat(c(pool3, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4, up_pool = TRUE)
  
  net = mx.symbol.Concat(c(pool2, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = convolution_module(net, kernel_size, pad_size,
                           filter_count = filter_count * 4, up_pool = TRUE)
  convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 4)
  net = mx.symbol.Concat(c(pool1, net), num.args = 2)
  net = mx.symbol.Dropout(net)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2)
  net = convolution_module(net, kernel_size, pad_size, filter_count = filter_count * 2, up_pool = TRUE)
  net = mx.symbol.Flatten(net)
  net = mx.symbol.FullyConnected(data = net, num_hidden = num_classes)
  net = mx.symbol.SoftmaxOutput(data = net, name = 'softmax')
  return(net)
}
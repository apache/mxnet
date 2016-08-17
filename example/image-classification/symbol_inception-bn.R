library(mxnet)

ConvFactory <- function(data, num_filter, kernel, stride = c(1, 1),
                        pad = c(0, 0), name = '', suffix = '') {
    conv <- mx.symbol.Convolution(data = data, num_filter = num_filter,
                                  kernel = kernel, stride = stride, pad = pad,
                                  name = paste('conv_', name, suffix, sep = ''))
    
    bn <- mx.symbol.BatchNorm(data = conv, name = paste('bn_', name, suffix, sep = ''))
    act <- mx.symbol.Activation(data = bn, act_type = 'relu', name = paste('relu_', name, suffix, sep = ''))
    return(act)
}

InceptionFactoryA <- function(data, num_1x1, num_3x3red, num_3x3, num_d3x3red,
                              num_d3x3, pool, proj, name) {
  # 1x1
  c1x1 <- ConvFactory(data = data, num_filter = num_1x1, kernel = c(1, 1), name = paste(name, '_1x1', sep = '')
    )
  # 3x3 reduce + 3x3
  c3x3r <- ConvFactory(data = data, num_filter = num_3x3red, kernel = c(1, 1),
                       name = paste(name, '_3x3', sep = ''), suffix = '_reduce')

  c3x3 <- ConvFactory(data = c3x3r, num_filter = num_3x3, kernel = c(3, 3),
                      pad = c(1, 1), name = paste(name, '_3x3', sep = ''))
  # double 3x3 reduce + double 3x3
  cd3x3r <- ConvFactory(data = data, num_filter = num_d3x3red, kernel = c(1, 1),
                        name = paste(name, '_double_3x3', sep = ''), suffix = '_reduce')

  cd3x3 <- ConvFactory(data = cd3x3r, num_filter = num_d3x3, kernel = c(3, 3),
                       pad = c(1, 1), name = paste(name, '_double_3x3_0', sep = ''))

  cd3x3 <- ConvFactory(data = cd3x3, num_filter = num_d3x3, kernel = c(3, 3),
                       pad = c(1, 1), name = paste(name, '_double_3x3_1', sep = ''))
  # pool + proj
  pooling <- mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(1, 1),
                               pad = c(1, 1), pool_type = pool,
                               name = paste(pool, '_pool_', name, '_pool', sep = ''))
  cproj <- ConvFactory(data = pooling, num_filter = proj, kernel = c(1, 1),
                       name = paste(name, '_proj', sep = ''))
  # concat
  concat_lst <- list()
  concat_lst <- c(c1x1, c3x3, cd3x3, cproj)
  concat_lst$num.args = 4
  concat_lst$name = paste('ch_concat_', name, '_chconcat', sep = '')
  concat = mxnet:::mx.varg.symbol.Concat(concat_lst)
  return(concat)
}

InceptionFactoryB <- function(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name) {
    # 3x3 reduce + 3x3
    c3x3r <- ConvFactory(data = data, num_filter = num_3x3red, kernel = c(1, 1),
                         name = paste(name, '_3x3', sep = ''), suffix = '_reduce')
    c3x3 <- ConvFactory(data = c3x3r, num_filter = num_3x3, kernel = c(3, 3),
                        pad = c(1, 1), stride = c(2, 2), name = paste(name, '_3x3', sep = ''))
    # double 3x3 reduce + double 3x3
    cd3x3r <- ConvFactory(data = data, num_filter = num_d3x3red, kernel = c(1, 1),
                         name = paste(name, '_double_3x3', sep = ''), suffix = '_reduce')
    cd3x3 <- ConvFactory(data = cd3x3r, num_filter = num_d3x3, kernel = c(3, 3),
                         pad = c(1, 1), stride = c(1, 1), name = paste(name, '_double_3x3_0', sep = ''))
    cd3x3 = ConvFactory(data = cd3x3, num_filter = num_d3x3, kernel = c(3, 3),
                        pad = c(1, 1), stride = c(2, 2), name = paste(name, '_double_3x3_1', sep = ''))
    # pool + proj
    pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(2, 2),
                                pad = c(1, 1), pool_type = "max",
                                name = paste('max_pool_', name, '_pool', sep = ''))
    # concat
    concat_lst <- list()
    concat_lst <- c(c3x3, cd3x3, pooling)
    concat_lst$num.args = 3
    concat_lst$name = paste('ch_concat_', name, '_chconcat', sep = '')
    concat = mxnet:::mx.varg.symbol.Concat(concat_lst)
    return(concat)
}

get_symbol <- function(num_classes = 1000) {
  # data
  data = mx.symbol.Variable(name = "data")
  # stage 1
  conv1 = ConvFactory(data = data, num_filter = 64, kernel = c(7, 7),
                      stride = c(2, 2), pad = c(3, 3), name = 'conv1')
  pool1 = mx.symbol.Pooling(data = conv1, kernel = c(3, 3), stride = c(2, 2),
                            name = 'pool1', pool_type = 'max')
  # stage 2
  conv2red = ConvFactory(data = pool1, num_filter = 64, kernel = c(1, 1),
                         stride = c(1, 1), name = 'conv2red')
  conv2 = ConvFactory(data = conv2red, num_filter = 192, kernel = c(3, 3),
                      stride = c(1, 1), pad = c(1, 1), name = 'conv2')
  pool2 = mx.symbol.Pooling(data = conv2, kernel = c(3, 3), stride = c(2, 2),
                            name = 'pool2', pool_type = 'max')
  # stage 2
  in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
  in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
  in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
  # stage 3
  in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
  in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
  in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
  in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
  in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
  # stage 4
  in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
  in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
  # global avg pooling
  avg = mx.symbol.Pooling(data = in5b, kernel = c(7, 7), stride = c(1, 1),
                          name = "global_pool", pool_type = 'avg')
  # linear classifier
  flatten = mx.symbol.Flatten(data = avg, name = 'flatten')
  fc1 = mx.symbol.FullyConnected(data = flatten,
                                 num_hidden = num_classes,
                                 name = 'fc1')
  softmax = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')
  return(softmax)
}

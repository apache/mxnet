require(mxnet)

source("get_data.R")

context("models")

if (Sys.getenv("R_GPU_ENABLE") != "" & as.integer(Sys.getenv("R_GPU_ENABLE")) == 1) {
  mx.ctx.default(new = mx.gpu())
  message("Using GPU for testing.")
}

test_that("MNIST", {
#   # Network configuration
   GetMNIST_ubyte()
   batch.size <- 100
   data <- mx.symbol.Variable("data")
   fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
   act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
   fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
   act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
   fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
   softmax <- mx.symbol.Softmax(fc3, name = "sm")
   
   dtrain = mx.io.MNISTIter(
     image="data/train-images-idx3-ubyte",
     label="data/train-labels-idx1-ubyte",
     data.shape=c(784),
     batch.size=batch.size,
     shuffle=TRUE,
     flat=TRUE,
     silent=0,
     seed=10)
   
   dtest = mx.io.MNISTIter(
     image="data/t10k-images-idx3-ubyte",
     label="data/t10k-labels-idx1-ubyte",
     data.shape=c(784),
     batch.size=batch.size,
     shuffle=FALSE,
     flat=TRUE,
     silent=0)
   
   mx.set.seed(0)

   # create the model
   model <- mx.model.FeedForward.create(softmax, X=dtrain, eval.data=dtest,
                                        ctx = mx.ctx.default(), num.round=1,
                                        learning.rate=0.1, momentum=0.9,
                                        initializer=mx.init.uniform(0.07),
                                        epoch.end.callback=mx.callback.save.checkpoint("chkpt"),
                                        batch.end.callback=mx.callback.log.train.metric(100))
   
   # do prediction
   pred <- predict(model, dtest)
   label <- mx.io.extract(dtest, "label")
   dataX <- mx.io.extract(dtest, "data")
   # Predict with R's array
   pred2 <- predict(model, X=dataX)
   
   accuracy <- function(label, pred) {
     ypred = max.col(t(as.array(pred)))
     return(sum((as.array(label) + 1) == ypred) / length(label))
   }

   expect_equal(accuracy(label, pred), accuracy(label, pred2))
   
   file.remove("chkpt-0001.params")
   file.remove("chkpt-symbol.json")
})

test_that("Regression", {
  data(BostonHousing, package = "mlbench")
  train.ind <- seq(1, 506, 3)
  train.x <- data.matrix(BostonHousing[train.ind,-14])
  train.y <- BostonHousing[train.ind, 14]
  test.x <- data.matrix(BostonHousing[-train.ind,-14])
  test.y <- BostonHousing[-train.ind, 14]
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, num_hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(fc1)
  
  demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
    res <- mean(abs(label - pred))
    return(res)
  })
  mx.set.seed(0)
  model <- mx.model.FeedForward.create(lro, X = train.x, y = train.y,
                                       ctx = mx.ctx.default(), num.round = 5,
                                       array.batch.size = 20,
                                       learning.rate = 2e-6,
                                       momentum = 0.9,
                                       eval.metric = demo.metric.mae)
  
  train.x <- data.matrix(BostonHousing[train.ind, -(13:14)])
  train.y <- BostonHousing[train.ind, c(13:14)]
  test.x <- data.matrix(BostonHousing[-train.ind, -(13:14)])
  test.y <- BostonHousing[-train.ind, c(13:14)]
  
  data <- mx.symbol.Variable("data")
  fc2 <- mx.symbol.FullyConnected(data, num_hidden=2)
  lro2 <- mx.symbol.LinearRegressionOutput(fc2)
  
  mx.set.seed(0)
  train_iter = mx.io.arrayiter(data = t(train.x), label = t(train.y))
  
  model <- mx.model.FeedForward.create(lro2, X = train_iter,
                                       ctx = mx.ctx.default(),
                                       num.round = 50,
                                       array.batch.size = 20,
                                       learning.rate = 2e-6,
                                       momentum = 0.9)
})


test_that("Classification", {
  data(Sonar, package = "mlbench")
  Sonar[, 61] <- as.numeric(Sonar[, 61]) - 1
  train.ind <- c(1:50, 100:150)
  train.x <- data.matrix(Sonar[train.ind, 1:60])
  train.y <- Sonar[train.ind, 61]
  test.x <- data.matrix(Sonar[-train.ind, 1:60])
  test.y <- Sonar[-train.ind, 61]
  mx.set.seed(0)
  model <- mx.mlp(train.x, train.y, hidden_node = 10,
                  out_node = 2, out_activation = "softmax",
                  num.round = 5, array.batch.size = 15,
                  learning.rate = 0.07,
                  momentum = 0.9,
                  eval.metric = mx.metric.accuracy)
})

test_that("Fine-tune", {
  GetInception()
  GetCatDog()
  train_iter <- mx.io.ImageRecordIter(path.imgrec = "./data/cats_dogs/cats_dogs_train.rec",
                                      batch.size  = 8, data.shape  = c(224, 224, 3),
                                      rand.crop   = TRUE, rand.mirror = TRUE)
  val_iter <- mx.io.ImageRecordIter(path.imgrec = "./data/cats_dogs/cats_dogs_val.rec",
                                    batch.size  = 8, data.shape  = c(224, 224, 3),
                                    rand.crop   = FALSE, rand.mirror = FALSE)
  inception_bn <- mx.model.load("./model/Inception-BN", iteration = 126)
  symbol <- inception_bn$symbol
  internals <- symbol$get.internals()
  outputs <- internals$outputs
  
  flatten <- internals$get.output(which(outputs == "flatten_output"))
  
  new_fc <- mx.symbol.FullyConnected(data = flatten, num_hidden = 2, name = "fc1")
  new_soft <- mx.symbol.SoftmaxOutput(data = new_fc, name = "softmax")
  arg_params_new <- mx.model.init.params(symbol = new_soft,
                                         input.shape = list("data" = c(224, 224, 3, 8)),
                                         output.shape = NULL,
                                         initializer = mx.init.uniform(0.1),
                                         ctx = mx.cpu())$arg.params
  fc1_weights_new <- arg_params_new[["fc1_weight"]]
  fc1_bias_new <- arg_params_new[["fc1_bias"]]
  
  arg_params_new <- inception_bn$arg.params
  
  arg_params_new[["fc1_weight"]] <- fc1_weights_new
  arg_params_new[["fc1_bias"]] <- fc1_bias_new

  #model <- mx.model.FeedForward.create(symbol = new_soft, X = train_iter, eval.data = val_iter,
  #                                     ctx = mx.ctx.default(), eval.metric = mx.metric.accuracy,
  #                                     num.round = 2, learning.rate = 0.05, momentum = 0.9,
  #                                     wd = 0.00001, kvstore = "local",
  #                                     batch.end.callback = mx.callback.log.train.metric(50),
  #                                     initializer = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  #                                     optimizer = "sgd",
  #                                     arg.params = arg_params_new,
  #                                     aux.params = inception_bn$aux.params)
})                                       

test_that("Matrix Factorization", {
  
  # Use fake random data instead of GetMovieLens() to remove external dependency
  set.seed(123)
  user <- sample(943, size = 100000, replace = T)
  item <- sample(1682, size = 100000, replace = T)
  score <- sample(5, size = 100000, replace = T)
  DF <- data.frame(user, item, score)
  
  max_user <- max(DF$user)
  max_item <- max(DF$item)
  DF_mat_x <- data.matrix(t(DF[, 1:2]))
  DF_y <- DF[, 3]
  k <- 64
  user <- mx.symbol.Variable("user")
  item <- mx.symbol.Variable("item")
  score <- mx.symbol.Variable("score")
  user1 <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(user), input_dim = max_user,
                               output_dim = k, name = "user1")
  item1 <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(item), input_dim = max_item,
                               output_dim = k, name = "item1")
  pred <- user1 * item1
  pred1 <- mx.symbol.sum_axis(pred, axis = 1, name = "pred1")
  pred2 <- mx.symbol.Flatten(pred1, name = "pred2")
  pred3 <- mx.symbol.LinearRegressionOutput(data = pred2, label = score, name = "pred3")

  mx.set.seed(123)
  
  CustomIter <- setRefClass( "CustomIter", fields = c("iter1", "iter2"),
                             contains = "Rcpp_MXArrayDataIter",
      methods = list(
        initialize = function(iter1, iter2) {
          .self$iter1 <- iter1
          .self$iter2 <- iter2
          .self
        },
        value = function() {
          user <- .self$iter1$value()$data
          item <- .self$iter2$value()$data
          score <- .self$iter1$value()$label
          list(user = user,
               item = item,
               score = score)
        },
        iter.next = function() {
          .self$iter1$iter.next()
          .self$iter2$iter.next()
        },
        reset = function() {
          .self$iter1$reset()
          .self$iter2$reset()
        },
        num.pad = function() {
          .self$iter1$num.pad()
        },
        finalize = function() {
          .self$iter1$finalize()
          .self$iter2$finalize()
        }
      )
    )
  
  user_iter = mx.io.arrayiter(data = DF[, 1], label = DF[, 3], batch.size = k)
  
  item_iter = mx.io.arrayiter(data = DF[, 2], label = DF[, 3], batch.size = k)
  
  train_iter <- CustomIter$new(user_iter, item_iter)
  
  model <- mx.model.FeedForward.create(pred3, X = train_iter, ctx = mx.ctx.default(),
                                       num.round = 5, initializer = mx.init.uniform(0.07),
                                       learning.rate = 0.07,
                                       eval.metric = mx.metric.rmse,
                                       momentum = 0.9,
                                       epoch.end.callback = mx.callback.log.train.metric(1),
                                       input.names = c("user", "item"),
                                       output.names = "score")
})

test_that("Captcha", {
  GetCaptcha_data()
  data.shape <- c(80, 30, 3)
  batch_size <- 40
  train <- mx.io.ImageRecordIter(
    path.imgrec   = "./data/captcha_example/captcha_train.rec",
    path.imglist  = "./data/captcha_example/captcha_train.lst",
    batch.size    = batch_size,
    label.width   = 4,
    data.shape    = data.shape,
    mean.img      = "mean.bin")
  
  val <- mx.io.ImageRecordIter(
    path.imgrec   = "./data/captcha_example/captcha_test.rec",
    path.imglist  = "./data/captcha_example/captcha_test.lst",
    batch.size    = batch_size,
    label.width   = 4,
    data.shape    = data.shape,
    mean.img      = "mean.bin")
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  conv1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 32)
  pool1 <- mx.symbol.Pooling(data = conv1, pool_type = "max", kernel = c(2, 2), stride = c(1, 1))
  relu1 <- mx.symbol.Activation(data = pool1, act_type = "relu")
  
  conv2 <- mx.symbol.Convolution(data = relu1, kernel = c(5, 5), num_filter = 32)
  pool2 <- mx.symbol.Pooling(data = conv2, pool_type = "avg", kernel = c(2, 2), stride = c(1, 1))
  relu2 <- mx.symbol.Activation(data = pool2, act_type = "relu")
  
  flatten <- mx.symbol.Flatten(data = relu2)
  fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 120)
  fc21 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
  fc22 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
  fc23 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
  fc24 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
  fc2 <- mx.symbol.Concat(c(fc21, fc22, fc23, fc24), dim = 0, num.args = 4)
  label <- mx.symbol.transpose(data = label)
  label <- mx.symbol.Reshape(data = label, target_shape = c(0))
  captcha_net <- mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")
  
  mx.metric.acc2 <- mx.metric.custom("accuracy", function(label, pred) {
    ypred <- max.col(t(pred)) - 1
    ypred <- matrix(ypred, nrow = nrow(label), ncol = ncol(label), byrow = TRUE)
    return(sum(colSums(label == ypred) == 4)/ncol(label))
  })
  
  mx.set.seed(42)
  
  train$reset()
  train$iter.next()
  
  input.names <- "data"
  input.shape <- sapply(input.names, function(n){dim(train$value()[[n]])}, simplify = FALSE)
  arg_names <- arguments(captcha_net)
  output.names <- "label"
  output.shape <- sapply(output.names, function(n){dim(train$value()[[n]])}, simplify = FALSE)
  params <- mx.model.init.params(captcha_net, input.shape, output.shape, 
                                 mx.init.Xavier(factor_type = "in", magnitude = 2.34),
                                 mx.cpu())

  #model <- mx.model.FeedForward.create(
  #  X                  = train,
  #  eval.data          = val,
  #  ctx                = mx.ctx.default(),
  #  symbol             = captcha_net,
  #  eval.metric        = mx.metric.acc2,
  #  num.round          = 1,
  #  learning.rate      = 1e-04,
  #  momentum           = 0.9,
  #  wd                 = 1e-05,
  #  batch.end.callback = mx.callback.log.train.metric(50),
  #  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  #  optimizer          = "sgd",
  #  clip_gradient      = 10)
})

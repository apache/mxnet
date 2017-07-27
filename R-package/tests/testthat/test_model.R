require(mxnet)

source("get_data.R")

context("models")

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
   devices = lapply(1:2, function(i) {
     mx.cpu(i)
   })
   
   # create the model
   model <- mx.model.FeedForward.create(softmax, X=dtrain, eval.data=dtest,
                                        ctx=devices, num.round=1,
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
                                       ctx = mx.cpu(), num.round = 5,
                                       array.batch.size = 20,
                                       learning.rate = 2e-6,
                                       momentum = 0.9,
                                       eval.metric = demo.metric.mae)
  
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
  #                                     ctx = mx.cpu(), eval.metric = mx.metric.accuracy,
  #                                     num.round = 2, learning.rate = 0.05, momentum = 0.9,
  #                                     wd = 0.00001, kvstore = "local",
  #                                     batch.end.callback = mx.callback.log.train.metric(50),
  #                                     initializer = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  #                                     optimizer = "sgd",
  #                                     arg.params = arg_params_new,
  #                                     aux.params = inception_bn$aux.params)
})                                       

test_that("Matrix Factorization", {
  GetMovieLens()
  DF <- read.table("./data/ml-100k/u.data", header = F, sep = "\t")
  names(DF) <- c("user", "item", "score", "time")
  max_user <- max(DF$user)
  max_item <- max(DF$item)
  DF_mat_x <- data.matrix(t(DF[, 1:2]))
  DF_y <- DF[, 3]
  k <- 64
  user <- mx.symbol.Variable("user")
  item <- mx.symbol.Variable("item")
  score <- mx.symbol.Variable("label")
  user1 <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(user), input_dim = max_user,
                               output_dim = k, name = "user1")
  item1 <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(item), input_dim = max_item,
                               output_dim = k, name = "item1"
    )
  pred <- user1 * item1
  pred1 <- mx.symbol.sum_axis(pred, axis = 1, name = "pred1")
  pred2 <- mx.symbol.Flatten(pred1, name = "pred2")
  pred3 <- mx.symbol.LinearRegressionOutput(data = pred2, label = score, name = "pred3")
  devices = lapply(1:2, function(i) {
    mx.cpu(i)
  })
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
          label <- .self$iter1$value()$label
          list(user = user,
               item = item,
               label = label)
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
  
  model <- mx.model.FeedForward.create(pred3, X = train_iter, ctx = devices,
                                       num.round = 5, initializer = mx.init.uniform(0.07),
                                       learning.rate = 0.07,
                                       eval.metric = mx.metric.rmse,
                                       momentum = 0.9,
                                       epoch.end.callback = mx.callback.log.train.metric(1),
                                       input.names = c("user", "item"),
                                       output.names = "label")
})

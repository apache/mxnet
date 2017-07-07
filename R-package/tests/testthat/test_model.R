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
                                       ctx = mx.cpu(), num.round = 50,
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
                  num.round = 20, array.batch.size = 15,
                  learning.rate = 0.07, 
                  momentum = 0.9,
                  eval.metric = mx.metric.accuracy)
})


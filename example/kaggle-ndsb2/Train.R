# Train.R for Second Annual Data Science Bowl
# Deep learning model with GPU support
# Please refer to https://mxnet.readthedocs.org/en/latest/build.html#r-package-installation
# for installation guide

require(mxnet)
require(data.table)

##A lenet style net, takes difference of each frame as input.
get.lenet <- function() {
  source <- mx.symbol.Variable("data")
  source <- (source-128) / 128
  frames <- mx.symbol.SliceChannel(source, num.outputs = 30)
  diffs <- list()
  for (i in 1:29) {
    diffs <- c(diffs, frames[[i + 1]] - frames[[i]])
  }
  diffs$num.args = 29
  source <- mxnet:::mx.varg.symbol.Concat(diffs)
  net <-
    mx.symbol.Convolution(source, kernel = c(5, 5), num.filter = 40)
  net <- mx.symbol.BatchNorm(net, fix.gamma = TRUE)
  net <- mx.symbol.Activation(net, act.type = "relu")
  net <-
    mx.symbol.Pooling(
      net, pool.type = "max", kernel = c(2, 2), stride = c(2, 2)
    )
  net <-
    mx.symbol.Convolution(net, kernel = c(3, 3), num.filter = 40)
  net <- mx.symbol.BatchNorm(net, fix.gamma = TRUE)
  net <- mx.symbol.Activation(net, act.type = "relu")
  net <-
    mx.symbol.Pooling(
      net, pool.type = "max", kernel = c(2, 2), stride = c(2, 2)
    )
  # first fullc
  flatten <- mx.symbol.Flatten(net)
  flatten <- mx.symbol.Dropout(flatten)
  fc1 <- mx.symbol.FullyConnected(data = flatten, num.hidden = 600)
  # Name the final layer as softmax so it auto matches the naming of data iterator
  # Otherwise we can also change the provide_data in the data iter
  return(mx.symbol.LogisticRegressionOutput(data = fc1, name = 'softmax'))
}

network <- get.lenet()
batch_size <- 32

# CSVIter is uesed here, since the data can't fit into memory
data_train <- mx.io.CSVIter(
  data.csv = "./train-64x64-data.csv", data.shape = c(64, 64, 30),
  label.csv = "./train-stytole.csv", label.shape = 600,
  batch.size = batch_size
)

data_validate <- mx.io.CSVIter(
  data.csv = "./validate-64x64-data.csv",
  data.shape = c(64, 64, 30),
  batch.size = 1
)

# Custom evaluation metric on CRPS.
mx.metric.CRPS <- mx.metric.custom("CRPS", function(label, pred) {
  pred <- as.array(pred)
  label <- as.array(label)
  for (i in 1:dim(pred)[2]) {
    for (j in 1:(dim(pred)[1] - 1)) {
      if (pred[j, i] > pred[j + 1, i]) {
        pred[j + 1, i] = pred[j, i]
      }
    }
  }
  return(sum((label - pred) ^ 2) / length(label))
})

# Training the stytole net
mx.set.seed(0)
stytole_model <- mx.model.FeedForward.create(
  X = data_train,
  ctx = mx.gpu(0),
  symbol = network,
  num.round = 65,
  learning.rate = 0.001,
  wd = 0.00001,
  momentum = 0.9,
  eval.metric = mx.metric.CRPS
)

# Predict stytole
stytole_prob = predict(stytole_model, data_validate)

# Training the diastole net
network = get.lenet()
batch_size = 32
data_train <-
  mx.io.CSVIter(
    data.csv = "./train-64x64-data.csv", data.shape = c(64, 64, 30),
    label.csv = "./train-diastole.csv", label.shape = 600,
    batch.size = batch_size
  )

diastole_model = mx.model.FeedForward.create(
  X = data_train,
  ctx = mx.gpu(0),
  symbol = network,
  num.round = 65,
  learning.rate = 0.001,
  wd = 0.00001,
  momentum = 0.9,
  eval.metric = mx.metric.CRPS
)

# Predict diastole
diastole_prob = predict(diastole_model, data_validate)

accumulate_result <- function(validate_lst, prob) {
  t <- read.table(validate_lst, sep = ",")
  p <- cbind(t[,1], t(prob))
  dt <- as.data.table(p)
  return(dt[, lapply(.SD, mean), by = V1])
}

stytole_result = as.data.frame(accumulate_result("./validate-label.csv", stytole_prob))
diastole_result = as.data.frame(accumulate_result("./validate-label.csv", diastole_prob))

train_csv <- read.table("./train-label.csv", sep = ',')

# we have 2 person missing due to frame selection, use udibr's hist result instead
doHist <- function(data) {
  res <- rep(0, 600)
  for (i in 1:length(data)) {
    for (j in round(data[i]):600) {
      res[j] = res[j] + 1
    }
  }
  return(res / length(data))
}

hSystole = doHist(train_csv[, 2])
hDiastole = doHist(train_csv[, 3])

res <- read.table("data/sample_submission_validate.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

submission_helper <- function(pred) {
  for (i in 2:length(pred)) {
    if (pred[i] < pred[i - 1]) {
      pred[i] = pred[i - 1]
    }
  }
  return(pred)
}

for (i in 1:nrow(res)) {
  key <- unlist(strsplit(res$Id[i], "_"))[1]
  target <- unlist(strsplit(res$Id[i], "_"))[2]
  if (key %in% stytole_result$V1) {
    if (target == 'Diastole') {
      res[i, 2:601] <- submission_helper(diastole_result[which(diastole_result$V1 == key), 2:601])
    } else {
      res[i, 2:601] <- submission_helper(stytole_result[which(stytole_result$V1 == key), 2:601])
    }
  } else {
    if (target == 'Diastole') {
      res[i, 2:601] <- hDiastole
    } else {
      res[i, 2:601] <- hSystole
    }
  }
}

write.table(res, file = "submission.csv", sep = ",", quote = FALSE, row.names = FALSE)

require("mxnet")

source("mx.io.bucket.iter.R")
source("rnn.train.R")

corpus_bucketed_train <- readRDS(file = "corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test <- readRDS(file = "corpus_bucketed_test_100_200_300_500_800_left.rds")

vocab <- length(corpus_bucketed_test$dic)

### Create iterators
batch.size <- 64

num.round <- 16

train.data <- mx.io.bucket.iter(buckets = corpus_bucketed_train$buckets, batch.size = batch.size, 
  data.mask.element = 0, shuffle = TRUE)

eval.data <- mx.io.bucket.iter(buckets = corpus_bucketed_test$buckets, batch.size = batch.size, 
  data.mask.element = 0, shuffle = FALSE)

mx.set.seed(0)
optimizer <- mx.opt.create("adadelta", rho = 0.92, epsilon = 1e-06, wd = 2e-04, clip_gradient = NULL, 
  rescale.grad = 1/batch.size)

model_sentiment_lstm <- mx.rnn.buckets(train.data = train.data, begin.round = 1, 
  num.round = num.round, ctx = mx.cpu(), metric = mx.metric.accuracy, optimizer = optimizer, 
  num.rnn.layer = 2, num.embed = 16, num.hidden = 24, num.label = 2, input.size = vocab, 
  initializer = mx.init.Xavier(rnd_type = "gaussian", factor_type = "in", magnitude = 2), 
  dropout = 0.25, config = "seq-to-one", batch.end.callback = mx.callback.log.train.metric(period = 50), 
  verbose = TRUE)

mx.model.save(model_sentiment_lstm, prefix = "model_sentiment_lstm", iteration = num.round)

source("rnn.infer.R")

model <- mx.model.load("model_sentiment_lstm", iteration = num.round)

pred <- mx.rnn.infer.buckets(infer_iter = eval.data, model, "seq-to-one", ctx = mx.cpu())

ypred <- max.col(t(as.array(pred)), tie = "first") - 1

packer <- mxnet:::mx.nd.arraypacker()

eval.data$reset()

while (eval.data$iter.next()) {
  packer$push(eval.data$value()$label)
}

ylabel <- as.array(packer$get())

acc <- sum(ylabel == ypred)/length(ylabel)

message(paste("Acc:", acc))

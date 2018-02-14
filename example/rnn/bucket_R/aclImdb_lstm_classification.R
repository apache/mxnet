# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require("mxnet")

corpus_bucketed_train <- readRDS(file = "data/corpus_bucketed_train.rds")
corpus_bucketed_test <- readRDS(file = "data/corpus_bucketed_test.rds")

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

bucket_list <- unique(c(train.data$bucket.names, eval.data$bucket.names))

symbol_buckets <- sapply(bucket_list, function(seq) {
  rnn.graph(config = "seq-to-one", cell_type = "lstm", 
            num_rnn_layer = 1, num_embed = 2, num_hidden = 6, 
            num_decode = 2, input_size = vocab, dropout = 0.5, 
            ignore_label = -1, loss_output = "softmax",
            output_last_state = F, masking = T)
})

model_sentiment_lstm <- mx.model.buckets(symbol = symbol_buckets,
                          train.data = train.data, eval.data = eval.data,
                          num.round = num.round, ctx = devices, verbose = FALSE,
                          metric = mx.metric.accuracy, optimizer = optimizer,  
                          initializer = initializer,
                          batch.end.callback = NULL, 
                          epoch.end.callback = epoch.end.callback)

mx.model.save(model_sentiment_lstm, prefix = "model_sentiment_lstm", iteration = num.round)
model <- mx.model.load("model_sentiment_lstm", iteration = num.round)

pred <- mx.infer.rnn(infer.data = eval.data, model = model, ctx = mx.cpu())

ypred <- max.col(t(as.array(pred)), tie = "first") - 1

packer <- mxnet:::mx.nd.arraypacker()

eval.data$reset()

while (eval.data$iter.next()) {
  packer$push(eval.data$value()$label)
}

ylabel <- as.array(packer$get())

acc <- sum(ylabel == ypred)/length(ylabel)

message(paste("Acc:", acc))

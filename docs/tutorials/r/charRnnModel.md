
# Character-level Language Model using RNN

This tutorial will demonstrate creating a language model using a character level RNN model using MXNet-R package. You will need the following R packages to run this tutorial -
 - readr
 - stringr
 - stringi
 - mxnet

We will use the [tinyshakespeare](https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare) dataset to build this model.


```R
library("readr")
library("stringr")
library("stringi")
library("mxnet")
```

## Preprocess and prepare the data

Download the data:


```R
download.data <- function(data_dir) {
    dir.create(data_dir, showWarnings = FALSE)
    if (!file.exists(paste0(data_dir,'input.txt'))) {
        download.file(url='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt',
                      destfile=paste0(data_dir,'input.txt'), method='wget')
    }
}
```

Next we transform the test into feature vectors that is fed into the RNN model. The `make_data` function reads the dataset, cleans it of any non-alphanumeric characters, splits it into individual characters and groups it into sequences of length `seq.len`.


```R
make_data <- function(path, seq.len = 32, dic=NULL) {
  
  text_vec <- read_file(file = path)
  text_vec <- stri_enc_toascii(str = text_vec)
  text_vec <- str_replace_all(string = text_vec, pattern = "[^[:print:]]", replacement = "")
  text_vec <- strsplit(text_vec, '') %>% unlist
  
  if (is.null(dic)) {
    char_keep <- sort(unique(text_vec))
  } else char_keep <- names(dic)[!dic == 0]
  
  # Remove terms not part of dictionary
  text_vec <- text_vec[text_vec %in% char_keep]
  
  # Build dictionary
  dic <- 1:length(char_keep)
  names(dic) <- char_keep
  
  # reverse dictionary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
  # Adjust by -1 to have a 1-lag for labels
  num.seq <- (length(text_vec) - 1) %/% seq.len
  
  features <- dic[text_vec[1:(seq.len * num.seq)]] 
  labels <- dic[text_vec[1:(seq.len*num.seq) + 1]]
  
  features_array <- array(features, dim = c(seq.len, num.seq))
  labels_array <- array(labels, dim = c(seq.len, num.seq))
  
  return (list(features_array = features_array, labels_array = labels_array, dic = dic, rev_dic = rev_dic))
}


seq.len <- 100
data_prep <- make_data(path = "input.txt", seq.len = seq.len, dic=NULL)
```

Fetch the features and labels for training the model, and split the data into training and evaluation in 9:1 ratio.


```R
X <- data_prep$features_array
Y <- data_prep$labels_array
dic <- data_prep$dic
rev_dic <- data_prep$rev_dic
vocab <- length(dic)

samples <- tail(dim(X), 1)
train.val.fraction <- 0.9

X.train.data <- X[, 1:as.integer(samples * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(samples * train.val.fraction))]

X.train.label <- Y[, 1:as.integer(samples * train.val.fraction)]
X.val.label <- Y[, -(1:as.integer(samples * train.val.fraction))]

train_buckets <- list("100" = list(data = X.train.data, label = X.train.label))
eval_buckets <- list("100" = list(data = X.val.data, label = X.val.label))

train_buckets <- list(buckets = train_buckets, dic = dic, rev_dic = rev_dic)
eval_buckets <- list(buckets = eval_buckets, dic = dic, rev_dic = rev_dic)
```

Create iterators for training and evaluation datasets.


```R
vocab <- length(eval_buckets$dic)

batch.size <- 32

train.data <- mx.io.bucket.iter(buckets = train_buckets$buckets, batch.size = batch.size, 
                                data.mask.element = 0, shuffle = TRUE)

eval.data <- mx.io.bucket.iter(buckets = eval_buckets$buckets, batch.size = batch.size,
                               data.mask.element = 0, shuffle = FALSE)
```

## Train the Model


This model is a multi-layer RNN for sampling from character-level language models. It has a one-to-one model configuration since for each character, we want to predict the next one. For a sequence of length 100, there are also 100 labels, corresponding the same sequence of characters but offset by a position of +1. The parameters output_last_state is set to TRUE in order to access the state of the RNN cells when performing inference.


```R
rnn_graph_one_one <- rnn.graph(num_rnn_layer = 3, 
                               num_hidden = 96,
                               input_size = vocab,
                               num_embed = 64, 
                               num_decode = vocab,
                               dropout = 0.2, 
                               ignore_label = 0,
                               cell_type = "lstm",
                               masking = F,
                               output_last_state = T,
                               loss_output = "softmax",
                               config = "one-to-one")

graph.viz(rnn_graph_one_one, type = "graph", direction = "LR", 
          graph.height.px = 180, shape=c(100, 64))

devices <- mx.cpu()

initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

optimizer <- mx.opt.create("adadelta", rho = 0.9, eps = 1e-5, wd = 1e-8,
                           clip_gradient = 5, rescale.grad = 1/batch.size)

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)
batch.end.callback <- mx.callback.log.train.metric(period = 50)

mx.metric.custom_nd <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value = (state[[2]] / state[[1]]))
  }
  ret <- (list(init = init, update = update, get = get))
  class(ret) <- "mx.metric"
  return(ret)
}

mx.metric.Perplexity <- mx.metric.custom_nd("Perplexity", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  label_probs <- as.array(mx.nd.choose.element.0index(pred, label))
  batch <- length(label_probs)
  NLL <- -sum(log(pmax(1e-15, as.array(label_probs)))) / batch
  Perplexity <- exp(NLL)
  return(Perplexity)
})

model <- mx.model.buckets(symbol = rnn_graph_one_one,
                          train.data = train.data, eval.data = eval.data, 
                          num.round = 20, ctx = devices, verbose = TRUE,
                          metric = mx.metric.Perplexity, 
                          initializer = initializer, optimizer = optimizer, 
                          batch.end.callback = NULL, 
                          epoch.end.callback = epoch.end.callback)

mx.model.save(model, prefix = "one_to_one_seq_model", iteration = 20)
```

    Start training with 1 devices
    [1] Train-Perplexity=13.7040474322178
    [1] Validation-Perplexity=7.94617194460922
    [2] Train-Perplexity=6.57039815554525
    [2] Validation-Perplexity=6.60806110658011
    [3] Train-Perplexity=5.65360504501481
    [3] Validation-Perplexity=6.18932770630876
    [4] Train-Perplexity=5.32547285727298
    [4] Validation-Perplexity=6.02198756798859
    [5] Train-Perplexity=5.14373631472579
    [5] Validation-Perplexity=5.8095658243407
    [6] Train-Perplexity=5.03077673487379
    [6] Validation-Perplexity=5.72582993567431
    [7] Train-Perplexity=4.94453383291536
    [7] Validation-Perplexity=5.6445258528126
    [8] Train-Perplexity=4.88635290100261
    [8] Validation-Perplexity=5.6730024536433
    [9] Train-Perplexity=4.84205646230548
    [9] Validation-Perplexity=5.50960780230982
    [10] Train-Perplexity=4.80441673535513
    [10] Validation-Perplexity=5.57002263750006
    [11] Train-Perplexity=4.77763413242626
    [11] Validation-Perplexity=5.55152143269169
    [12] Train-Perplexity=4.74937775290777
    [12] Validation-Perplexity=5.44968305351486
    [13] Train-Perplexity=4.72824849541467
    [13] Validation-Perplexity=5.50889348298234
    [14] Train-Perplexity=4.70980846981694
    [14] Validation-Perplexity=5.51473225859859
    [15] Train-Perplexity=4.69685776886122
    [15] Validation-Perplexity=5.45391985233811
    [16] Train-Perplexity=4.67837107034824
    [16] Validation-Perplexity=5.46636764997829
    [17] Train-Perplexity=4.66866961934873
    [17] Validation-Perplexity=5.44267086113492
    [18] Train-Perplexity=4.65611469144194
    [18] Validation-Perplexity=5.4290169469462
    [19] Train-Perplexity=4.64614689879405
    [19] Validation-Perplexity=5.44221549833917
    [20] Train-Perplexity=4.63764001963654
    [20] Validation-Perplexity=5.42114250842862


## Inference on the Model

We now use the saved model to do inference and sample text character by character that will look like the original training data.


```R
set.seed(0)
model <- mx.model.load(prefix = "one_to_one_seq_model", iteration = 20)

internals <- model$symbol$get.internals()
sym_state <- internals$get.output(which(internals$outputs %in% "RNN_state"))
sym_state_cell <- internals$get.output(which(internals$outputs %in% "RNN_state_cell"))
sym_output <- internals$get.output(which(internals$outputs %in% "loss_output"))
symbol <- mx.symbol.Group(sym_output, sym_state, sym_state_cell)

infer_raw <- c("Thou ")
infer_split <- dic[strsplit(infer_raw, '') %>% unlist]
infer_length <- length(infer_split)

infer.data <- mx.io.arrayiter(data = matrix(infer_split), label = matrix(infer_split),  
                              batch.size = 1, shuffle = FALSE)

infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol,
                          arg.params = model$arg.params,
                          aux.params = model$aux.params,
                          input.params = NULL, 
                          ctx = devices)

pred_prob <- as.numeric(as.array(mx.nd.slice.axis(
    infer$loss_output, axis = 0, begin = infer_length-1, end = infer_length)))
pred <- sample(length(pred_prob), prob = pred_prob, size = 1) - 1
predict <- c(predict, pred)

for (i in 1:200) {
  
  infer.data <- mx.io.arrayiter(data = as.matrix(pred), label = as.matrix(pred),  
                                batch.size = 1, shuffle = FALSE)
  
  infer <- mx.infer.rnn.one(infer.data = infer.data, 
                            symbol = symbol,
                            arg.params = model$arg.params,
                            aux.params = model$aux.params,
                            input.params = list(rnn.state = infer[[2]], 
                                                rnn.state.cell = infer[[3]]), 
                            ctx = devices)
  
  pred_prob <- as.numeric(as.array(infer$loss_output))
  pred <- sample(length(pred_prob), prob = pred_prob, size = 1, replace = T) - 1
  predict <- c(predict, pred)
}

predict_txt <- paste0(rev_dic[as.character(predict)], collapse = "")
predict_txt_tot <- paste0(infer_raw, predict_txt, collapse = "")
print(predict_txt_tot)
```

    [1] "Thou NAknowledge thee my Comfort and his late she.FRIAR LAURENCE:Nothing a groats waterd forth. The lend he thank that;When she I am brother draw London: and not hear that know.BENVOLIO:How along, makes your "


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

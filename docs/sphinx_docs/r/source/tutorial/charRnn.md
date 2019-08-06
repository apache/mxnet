# Character-level Language Model using RNN

This tutorial will demonstrate how to create a language model (character-level RNN) using the **mxnet** R package. 

We will use the [tinyshakespeare](https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare) dataset to build this model. 
First let's load some required packages:

```{.python .input .R  n=1}
require("readr")
require("stringr")
require("stringi")
require("mxnet")
```

## Preprocess and prepare the data

We define the following function to download the text data:

```{.python .input .R  n=2}
download.data <- function(data_dir) {
    dir.create(data_dir, showWarnings = FALSE)
    if (!file.exists(paste0(data_dir,'input.txt'))) {
        download.file(url='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt',
                      destfile=paste0(data_dir,'input.txt'), method='wget')
    }
}
```

Next we transform the text into feature vectors that are fed into the RNN model. The `make_data` function reads the dataset, cleans it of any non-alphanumeric characters, splits it into individual characters and groups it into sequences of length `seq.len`.

```{.python .input .R  n=3}
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
```

Now execute the previously-defined functions and download/process the text data into sequences of length 100:

```{.python .input  n=4}
seq.len <- 100
download.data("")
data_prep <- make_data(path = "input.txt", seq.len = seq.len, dic=NULL)
```

Fetch the features and labels for training the model, and split the data into training and evaluation in 9:1 ratio.

```{.python .input .R  n=5}
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

Create iterators for training and evaluation datasets. ``mx.io.bucket.iter`` will provide batches of 32 training examples that can be simultaneously processed by the RNN.

```{.python .input .R  n=6}
vocab <- length(eval_buckets$dic)
batch.size <- 32
train.data <- mx.io.bucket.iter(buckets = train_buckets$buckets, batch.size = batch.size, 
                                data.mask.element = 0, shuffle = TRUE)
eval.data <- mx.io.bucket.iter(buckets = eval_buckets$buckets, batch.size = batch.size,
                               data.mask.element = 0, shuffle = FALSE)
```

## Train the RNN Model


Our language model will be a 3-layer LSTM recurrent network that operates upon individual characters at each time step.  Upon encountering each additional character in the text data, our goal in this task is to predict the next one. Thus, for a sequence of length 100, there are also 100 target labels to predict, corresponding the same sequence of characters but offset by a position of +1. This type of configuration (in which a label is predicted at each sequence position) is specified as ``one-to-one`` in our call to ``rnn.graph`` below, which defines the RNN architecture to be used.

The parameter ``output_last_state`` is set to **TRUE** in order to access the state of the RNN cells when performing inference. ``ignore_label`` specifies that sequence-positions with the given label 0 are ignored during computation of the loss function that guides learning of the LSTM parameters. The RNN we define below employs dropout regularization and utilizes an embedding layer which embeds each character in the vocabulary into a distinct 64-dimensional vector of continuous values (``num_embed``).  

Calling ``graph.viz`` produces a visualization of the computations performed within our RNN architecture. 
We train this model for 5 epochs, which may be a bit slow on your machine.  You can change this number by specifying  ``num.training.epochs``; lowering the value will speed up the training process, raising it will slow things down but the resulting RNN-generated text is likely to look better (around 20 epochs seems to produce more reasonable-looking text).

```{.python .input .R  n=7}
num.training.epochs <- 5  # change this to alter time required for training model
mx.set.seed(0)
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
                          num.round = num.training.epochs, ctx = devices, verbose = TRUE,
                          metric = mx.metric.Perplexity, 
                          initializer = initializer, optimizer = optimizer, 
                          batch.end.callback = NULL, 
                          epoch.end.callback = epoch.end.callback)

mx.model.save(model, prefix = "one_to_one_seq_model", iteration = 20)
```

## Inference on the Model

Note that the previously executed code has saved our trained model to disk.  We now load the saved model from disk and use it to do inference, in which we sample new text character by character that should follow the distribution of  the original training data. The code below produces a language sample of 200 characters, but you can change this via ``sample.text.len``, and obtain different text samples by altering the random seed ``Rseed`` (Note that we actually draw the next character in R code, not MXNet code, which is we we use R's ``set.seed`` here rather than ``mx.set.seed``).

```{.python .input .R  n=8}
Rseed <- 0  # change this to sample different text
sample.text.len <- 200  # change this to sample a different number of characters.

set.seed(Rseed)
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

for (i in 1:sample.text.len) {
  
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

While the generated text may look quite questionable, remember this model has no knowledge of words and has learned everything it knows about language by simply reading the provided training text one character at a time.

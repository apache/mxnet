# Time Series Modeling with LSTM network

This tutorial shows how to use a LSTM recurrent neural network for predicting multivariate time series data in R's **mxnet** package.

We employ an open source pollution dataset, the [PM2.5 data of US Embassy in Beijing](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data), where the goal is to forecast air pollution levels  with data recorded  over five years at the US embassy in Beijing, China.
We use past PM2.5 concentration, dew point, temperature, pressure, wind speed, snow and rain to predict future PM2.5 concentration levels.


## Load and pre-process data

The first step is to download the data:

```{.python .input  n=13}
filename <- "pollution.csv"
if (!file.exists(filename)) {
    download.file(url='http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv',
                  destfile=filename, method='wget')
}
```

**Note:** The above command relies on ``wget``.  If the command fails, you can instead manually download the data 
from this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00381/). After downloading, rename the resulting CSV file to the name specified by ``filename`` and move the file into the current working directory of our R session (use ``getwd()`` command to print this directory from the current R notebook).

After we have the data files in the right place, let's load some required R packages and then preprocess the data:

```{.python .input  n=14}
require(mxnet)
if (!require(readr)) { install.packages('readr') }
if (!require(dplyr)) { install.packages('dplyr') }
if (!require(abind)) { install.packages('abind') }


Data <- read.csv(file = filename, header = TRUE, sep = ",")

## Extracting specific features from the dataset as variables for time series: 
## We extract pollution, temperature, pressue, windspeed, snowfall and rainfall information
df <- data.frame(Data$pm2.5, Data$DEWP, Data$TEMP,
                 Data$PRES, Data$Iws, Data$Is, Data$Ir)
df[is.na(df)] <- 0 # impute missing values as 0

## Now we normalise each of the feature set to a range(0,1)
df <- matrix(as.matrix(df), ncol = ncol(df), dimnames = NULL)
rangenorm <- function(x) {
    (x - min(x))/(max(x) - min(x))
}
df <- apply(df, 2, rangenorm)
df <- t(df)
dim(df)
```

For using multidimesional sequence data with **mxnet**, we need to convert training data to the form (*n_dim* x *seq_len* x *num_samples*). 
For one-to-one RNN variants (with a label to predict at each sequence-position), the labels should be of the form (*seq_len* x *num_samples*). 
For many-to-one RNN variants (with a single label to predict for the entire sequence), the labels should be of the form (1 x *num_samples*).
Note that the **mxnet** package in R currently supports only these two RNN variants.

We have used ``n_dim`` = 7, ``seq_len`` = 100, and ``num_samples`` = 430 because the dataset has 430 samples, each the length of 100 time-stamps, and there are 7 time series to be used as input features at each time step.

```{.python .input  n=15}
n_dim <- 7
seq_len <- 100
num_samples <- 430

## extract only required data from dataset
trX <- df[1:n_dim, 25:(24 + (seq_len * num_samples))]

## the label (next PM2.5 concentration) should be one time-step ahead of the current PM2.5 concentration
trY <- df[1, 26:(25 + (seq_len * num_samples))]

## Reshape the matrices into format accepted by MXNet RNNs:
trainX <- trX
dim(trainX) <- c(n_dim, seq_len, num_samples)
trainY <- trY
dim(trainY) <- c(seq_len, num_samples)
```

## Defining and training the network

```{.python .input  n=40}
batch.size <- 32

# take first 300 samples for training - remaining 100 for evaluation
train_ids <- 1:300
eval_ids <- 301:400

## Create data iterators. 
## The number of samples used for training and evaluation is arbitrary.  
## We have kept aside few samples for testing purposes.
train.data <- mx.io.arrayiter(data = trainX[, , train_ids, drop = F], 
                              label = trainY[, train_ids],
                              batch.size = batch.size, shuffle = TRUE)

eval.data <- mx.io.arrayiter(data = trainX[, , eval_ids, drop = F], 
                             label = trainY[, eval_ids],
                             batch.size = batch.size, shuffle = FALSE)

## Create the symbol for RNN
symbol <- rnn.graph(num_rnn_layer = 1,
                    num_hidden = 5,
                    input_size = NULL,
                    num_embed = NULL,
                    num_decode = 1,
                    masking = F, 
                    loss_output = "linear",
                    dropout = 0.2, 
                    ignore_label = -1, 
                    cell_type = "lstm", 
                    output_last_state = T,
                    config = "one-to-one")



mx.metric.mse.seq <- mx.metric.custom("MSE", function(label, pred) {
    label = mx.nd.reshape(label, shape = -1)
    pred = mx.nd.reshape(pred, shape = -1)
    res <- mx.nd.mean(mx.nd.square(label - pred))
    return(as.array(res))
})



ctx <- mx.cpu()

initializer <- mx.init.Xavier(rnd_type = "gaussian",
                              factor_type = "avg", 
                              magnitude = 3)

optimizer <- mx.opt.create("adadelta",
                           rho = 0.9, 
                           eps = 1e-05, 
                           wd = 1e-06, 
                           clip_gradient = 1, 
                           rescale.grad = 1/batch.size)

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 10, 
                                                   logger = logger)

## train the network
num.epoch <- 100
system.time(model <- mx.model.buckets(symbol = symbol, 
                                      train.data = train.data, 
                                      eval.data = eval.data,
                                      num.round = num.epoch, 
                                      ctx = ctx, 
                                      verbose = TRUE, 
                                      metric = mx.metric.mse.seq, 
                                      initializer = initializer,
                                      optimizer = optimizer, 
                                      batch.end.callback = NULL, 
                                      epoch.end.callback = epoch.end.callback))
```

We can see how the mean squared error over the training/validation data varies as learning progresses:

```{.python .input  n=46}
plot(1:num.epoch, logger$train, type ="l", xlab="Epochs",ylab="MSE", col='blue')
lines(1:num.epoch, logger$eval, col='red')
legend("topright", legend = c("Train","Validation"), fill = c("blue","red"))
```

## Inference on the network

Now that we have trained the network, let’s use it for inference.

```{.python .input  n=42}
## We extract the state symbols for RNN
internals <- model$symbol$get.internals()
sym_state <- internals$get.output(which(internals$outputs %in% "RNN_state"))
sym_state_cell <- internals$get.output(which(internals$outputs %in% "RNN_state_cell"))
sym_output <- internals$get.output(which(internals$outputs %in% "loss_output"))
symbol <- mx.symbol.Group(sym_output, sym_state, sym_state_cell)

## We will predict 100 timestamps for 401st sample (first sample from the test samples)
pred_length <- 100
predicted <- numeric()

## We pass the 400th sample through the network to get the weights and use it for predicting next
## 100 time stamps.
data <- mx.nd.array(trainX[, , 400, drop = F])
label <- mx.nd.array(trainY[, 400, drop = F])


## We create dataiterators for the input, please note that the label is required to create
## iterator and will not be used in the inference. You can use dummy values too in the label.
infer.data <- mx.io.arrayiter(data = data, 
                              label = label, 
                              batch.size = 1, 
                              shuffle = FALSE)

infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol, 
                          arg.params = model$arg.params,
                          aux.params = model$aux.params, 
                          input.params = NULL, 
                          ctx = ctx)
## Once we get the weights for the above time series, we try to predict the next 100 steps for
## this time series, which is technically our 401st time series.

actual <- trainY[, 401]

## Now we iterate one by one to generate each of the next timestamp pollution values

for (i in 1:pred_length) {

    data <- mx.nd.array(trainX[, i, 401, drop = F])
    label <- mx.nd.array(trainY[i, 401, drop = F])
    infer.data <- mx.io.arrayiter(data = data, 
                                  label = label, 
                                  batch.size = 1, 
                                  shuffle = FALSE)
    ## note that we use rnn state values from previous iterations here
    infer <- mx.infer.rnn.one(infer.data = infer.data,
                              symbol = symbol,
                              ctx = ctx, 
                              arg.params = model$arg.params,
                              aux.params = model$aux.params, 
                              input.params = list(rnn.state = infer[[2]], 
                                                  rnn.state.cell = infer[[3]]))

    pred <- infer[[1]]
    predicted <- c(predicted, as.numeric(as.array(pred)))

}
```

``predicted`` contains the 100 prediction values output by our model. We plot the actual vs predicted values below.

```{.python .input  n=51}
plot(1:pred_length, predicted, type ="l", xlab="Time Steps", ylab="Values", col='blue', 
     ylim=c(min(c(actual,predicted)),max(c(actual,predicted))))
lines(1:pred_length, actual, col='red')
legend("topleft", legend = c("Actual","Predicted"), fill = c("blue","red"))

```

**Note:** This tutorial is merely for demonstration purposes and the network architectures and training hyperparameters have not been tuned extensively for accuracy.

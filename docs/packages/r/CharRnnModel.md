Char RNN Example
=============================================

This example aims to show how to use lstm model to build a char level language model, and generate text from it. We use a tiny shakespeare text for demo purpose.

Data can be found at [here](https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare) 

Preface
-------
This tutorial is written in Rmarkdown.
- You can directly view the hosted version of the tutorial from [MXNet R Document](http://mxnet.readthedocs.org/en/latest/packages/r/CharRnnModel.html)
- You can find the download the Rmarkdown source from [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/CharRnnModel.Rmd)

Load Data 
---------
First of all, load in the data and preprocess it.

```r
require(mxnet)
```

```
## Loading required package: mxnet
```

```
## Loading required package: methods
```
Set basic network parameters.

```r
batch.size = 32
seq.len = 32
num.hidden = 16
num.embed = 16
num.lstm.layer = 1
num.round = 1
learning.rate= 0.1
wd=0.00001
clip_gradient=1
update.period = 1
```
download the data.

```r
download.data <- function(data_dir) {
    dir.create(data_dir, showWarnings = FALSE)
    if (!file.exists(paste0(data_dir,'input.txt'))) {
        download.file(url='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt',
                      destfile=paste0(data_dir,'input.txt'), method='wget')
    }
}
```
Make dictionary from text.

```r
make.dict <- function(text, max.vocab=10000) {
    text <- strsplit(text, '')
    dic <- list()
    idx <- 1
    for (c in text[[1]]) {
        if (!(c %in% names(dic))) {
            dic[[c]] <- idx
            idx <- idx + 1
        }
    }
    if (length(dic) == max.vocab - 1)
        dic[["UNKNOWN"]] <- idx
    cat(paste0("Total unique char: ", length(dic), "\n"))
    return (dic)
}
```
Transfer text into data feature.

```r
make.data <- function(file.path, seq.len=32, max.vocab=10000, dic=NULL) {
    fi <- file(file.path, "r")
    text <- paste(readLines(fi), collapse="\n")
    close(fi)

    if (is.null(dic))
        dic <- make.dict(text, max.vocab)
    lookup.table <- list()
    for (c in names(dic)) {
        idx <- dic[[c]]
        lookup.table[[idx]] <- c 
    }

    char.lst <- strsplit(text, '')[[1]]
    num.seq <- as.integer(length(char.lst) / seq.len)
    char.lst <- char.lst[1:(num.seq * seq.len)]
    data <- array(0, dim=c(seq.len, num.seq))
    idx <- 1
    for (i in 1:num.seq) {
        for (j in 1:seq.len) {
            if (char.lst[idx] %in% names(dic))
                data[j, i] <- dic[[ char.lst[idx] ]]-1
            else {
                data[j, i] <- dic[["UNKNOWN"]]-1
            }
            idx <- idx + 1
        }
    }
    return (list(data=data, dic=dic, lookup.table=lookup.table))
}
```
Move tail text.

```r
drop.tail <- function(X, batch.size) {
    shape <- dim(X)
    nstep <- as.integer(shape[2] / batch.size)
    return (X[, 1:(nstep * batch.size)])
}
```
get the label of X

```r
get.label <- function(X) {
    label <- array(0, dim=dim(X))
    d <- dim(X)[1]
    w <- dim(X)[2]
    for (i in 0:(w-1)) {
        for (j in 1:d) {
            label[i*d+j] <- X[(i*d+j)%%(w*d)+1]
        }
    }
    return (label)
}
```
get training data and eval data

```r
download.data("./data/")
ret <- make.data("./data/input.txt", seq.len=seq.len)
```

```
## Total unique char: 65
```

```r
X <- ret$data
dic <- ret$dic
lookup.table <- ret$lookup.table

vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]

X.train.data <- X[, 1:as.integer(size * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(size * train.val.fraction))]
X.train.data <- drop.tail(X.train.data, batch.size)
X.val.data <- drop.tail(X.val.data, batch.size)

X.train.label <- get.label(X.train.data)
X.val.label <- get.label(X.val.data)

X.train <- list(data=X.train.data, label=X.train.label)
X.val <- list(data=X.val.data, label=X.val.label)
```

Training Model
--------------
In `mxnet`, we have a function called `mx.lstm` so that users can build a general lstm model. 


```r
model <- mx.lstm(X.train, X.val, 
                 ctx=mx.cpu(),
                 num.round=num.round, 
                 update.period=update.period,
                 num.lstm.layer=num.lstm.layer, 
                 seq.len=seq.len,
                 num.hidden=num.hidden, 
                 num.embed=num.embed, 
                 num.label=vocab,
                 batch.size=batch.size, 
                 input.size=vocab,
                 initializer=mx.init.uniform(0.1), 
                 learning.rate=learning.rate,
                 wd=wd,
                 clip_gradient=clip_gradient)
```

```
## Epoch [31] Train: NLL=3.53787130224343, Perp=34.3936275728271
## Epoch [62] Train: NLL=3.43087958036949, Perp=30.903813186055
## Epoch [93] Train: NLL=3.39771238228587, Perp=29.8956319855751
## Epoch [124] Train: NLL=3.37581711716687, Perp=29.2481732041015
## Epoch [155] Train: NLL=3.34523331338447, Perp=28.3671933405139
## Epoch [186] Train: NLL=3.30756356274787, Perp=27.31848454823
## Epoch [217] Train: NLL=3.25642968403829, Perp=25.9566978956055
## Epoch [248] Train: NLL=3.19825967486207, Perp=24.4898727477925
## Epoch [279] Train: NLL=3.14013971549828, Perp=23.1070950525017
## Epoch [310] Train: NLL=3.08747601837462, Perp=21.9216781782189
## Epoch [341] Train: NLL=3.04015595674863, Perp=20.9085038031042
## Epoch [372] Train: NLL=2.99839339255659, Perp=20.0532932584534
## Epoch [403] Train: NLL=2.95940091012609, Perp=19.2864139984503
## Epoch [434] Train: NLL=2.92603311380224, Perp=18.6534872738302
## Epoch [465] Train: NLL=2.89482756896395, Perp=18.0803835531869
## Epoch [496] Train: NLL=2.86668230478397, Perp=17.5786009078994
## Epoch [527] Train: NLL=2.84089368534943, Perp=17.1310684830416
## Epoch [558] Train: NLL=2.81725862932279, Perp=16.7309220880514
## Epoch [589] Train: NLL=2.79518870141492, Perp=16.3657166956952
## Epoch [620] Train: NLL=2.77445683225304, Perp=16.0299176962855
## Epoch [651] Train: NLL=2.75490970113174, Perp=15.719621374694
## Epoch [682] Train: NLL=2.73697900634351, Perp=15.4402696117257
## Epoch [713] Train: NLL=2.72059739336781, Perp=15.1893935780915
## Epoch [744] Train: NLL=2.70462837571585, Perp=14.948760335793
## Epoch [775] Train: NLL=2.68909904683828, Perp=14.7184093476224
## Epoch [806] Train: NLL=2.67460054451836, Perp=14.5065539595711
## Epoch [837] Train: NLL=2.66078997776751, Perp=14.3075873113043
## Epoch [868] Train: NLL=2.6476781639279, Perp=14.1212134100373
## Epoch [899] Train: NLL=2.63529039846876, Perp=13.9473621677371
## Epoch [930] Train: NLL=2.62367693518974, Perp=13.7863219168709
## Epoch [961] Train: NLL=2.61238282674384, Perp=13.6314936713501
## Iter [1] Train: Time: 10301.6818172932 sec, NLL=2.60536539345356, Perp=13.5361704272949
## Iter [1] Val: NLL=2.26093848746227, Perp=9.59208699731232
```

Inference from model
--------------------
helper function for random sample.

```r
cdf <- function(weights) {
    total <- sum(weights)
    result <- c()
    cumsum <- 0
    for (w in weights) {
        cumsum <- cumsum+w
        result <- c(result, cumsum / total)
    }
    return (result)
}

search.val <- function(cdf, x) {
    l <- 1
    r <- length(cdf) 
    while (l <= r) {
        m <- as.integer((l+r)/2)
        if (cdf[m] < x) {
            l <- m+1
        } else {
            r <- m-1
        }
    }
    return (l)
}
choice <- function(weights) {
    cdf.vals <- cdf(as.array(weights))
    x <- runif(1)
    idx <- search.val(cdf.vals, x)
    return (idx)
}
```
we can use random output or fixed output by choosing largest probability.

```r
make.output <- function(prob, sample=FALSE) {
    if (!sample) {
        idx <- which.max(as.array(prob))
    }
    else {
        idx <- choice(prob)
    }
    return (idx)

}
```

In `mxnet`, we have a function called `mx.lstm.inference` so that users can build a inference from lstm model and then use function `mx.lstm.forward` to get forward output from the inference.
Build inference from model.

```r
infer.model <- mx.lstm.inference(num.lstm.layer=num.lstm.layer,
                                 input.size=vocab,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=vocab,
                                 arg.params=model$arg.params,
                                 ctx=mx.cpu())
```
generate a sequence of 75 chars using function `mx.lstm.forward`.
```r
start <- 'a'
seq.len <- 75
random.sample <- TRUE

last.id <- dic[[start]]
out <- "a"
for (i in (1:(seq.len-1))) {
    input <- c(last.id-1)
    ret <- mx.lstm.forward(infer.model, input, FALSE)
    infer.model <- ret$model
    prob <- ret$prob
    last.id <- make.output(prob, random.sample)
    out <- paste0(out, lookup.table[[last.id]])
}
cat (paste0(out, "\n"))
```
The result:
```
ah not a drobl greens
Settled asing lately sistering sounted to their hight
```

Other RNN models
----------------
In `mxnet`, other RNN models like custom RNN and gru is also provided.
- For **custom RNN model**, you can replace `mx.lstm` with `mx.rnn` to train rnn model. Also, you can replace `mx.lstm.inference` and `mx.lstm.forward` with `mx.rnn.inference` and `mx.rnn.forward` to inference from rnn model and get forward result from the inference model.
- For **GRU model**, you can replace `mx.lstm` with `mx.gru` to train gru model. Also, you can replace `mx.lstm.inference` and `mx.lstm.forward` with `mx.gru.inference` and `mx.gru.forward` to inference from gru model and get forward result from the inference model.

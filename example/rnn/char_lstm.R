# Char LSTM Example.

# This example aims to show how to use lstm to build a char level language model, and generate text from it. We use a tiny shakespeare text for demo purpose.
# Data can be found at https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare. 

# If running for the first time, download the data by running the following commands: sh get_ptb_data.sh
 
require(mxnet)
source("lstm.R")

# Set basic network parameters.
batch.size = 32
seq.len = 32
num.hidden = 256
num.embed = 256
num.lstm.layer = 2
num.round = 21
learning.rate= 0.01
wd=0.00001
clip_gradient=1
update.period = 1

# Make dictionary from text
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

# Transfer text into data batch
make.batch <- function(file.path, batch.size=32, seq.lenth=32, max.vocab=10000, dic=NULL) {
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
    num.batch <- as.integer(length(char.lst) / batch.size)
    char.lst <- char.lst[1:(num.batch * batch.size)]
    data <- array(0, dim=c(batch.size, num.batch))
    idx <- 1
    for (j in 1:batch.size) {
        for (i in 1:num.batch) {
            if (char.lst[idx] %in% names(dic))
                data[j, i] <- dic[[ char.lst[idx] ]]
            else {
                data[j, i] <- dic[["UNKNOWN"]]
            }
            idx <- idx + 1
        }
    }
    return (list(data=data, dic=dic, lookup.table=lookup.table))
}

# Move tail text
drop.tail <- function(X, seq.len) {
    shape <- dim(X)
    nstep <- as.integer(shape[2] / seq.len)
    return (X[, 1:(nstep * seq.len)])
}

ret <- make.batch("./data/input.txt", batch.size=batch.size, seq.lenth=seq.len)
X <- ret$data
dic <- ret$dic
lookup.table <- ret$lookup.table

vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]
X.train <- X[, 1:as.integer(size * train.val.fraction)]
X.val <- X[, -(1:as.integer(size * train.val.fraction))]
X.train <- drop.tail(X.train, seq.len)
X.val <- drop.tail(X.val, seq.len)

# Set up LSTM model
model <- setup.rnn.model(ctx=mx.gpu(0),
                         num.lstm.layer=num.lstm.layer,
                         seq.len=seq.len,
                         num.hidden=num.hidden,
                         num.embed=num.embed,
                         num.label=vocab,
                         batch.size=batch.size,
                         input.size=vocab,
                         initializer=mx.init.uniform(0.1),
                         dropout=0.)

# Train LSTM model
train.lstm(model, X.train, X.val,
                num.round=num.round,
                half.life=3,
                update.period=update.period,
                learning.rate=learning.rate,
                wd=wd,
                clip_gradient=clip_gradient)

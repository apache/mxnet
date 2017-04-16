# PennTreeBank Language Model using lstm, you can replace mx.lstm by mx.gru/ mx.rnn to use gru/rnn model
# The data file can be found at:
# https://github.com/dmlc/web-data/tree/master/mxnet/ptb
require(hash)
require(mxnet)
require(stringr

    )

load.data <- function(path, dic=NULL) {
    fi <- file(path, "r")
    content <- paste(readLines(fi), collapse="<eos>")
    close(fi)
    #cat(content)
    content <- str_split(content, ' ')[[1]]
    cat(paste0("Loading ", path, ", size of data = ", length(content), "\n"))
    X <- array(0, dim=c(length(content)))
    #cat(X)
    if (is.null(dic))
        dic <- hash()
    idx <- 1
    for (i in 1:length(content)) {
        word <- content[i]
        if (str_length(word) > 0) {
            if (!has.key(word, dic)) {
                dic[[word]] <- idx
                idx <- idx + 1
            }
            X[i] <- dic[[word]]
        }
    }
    cat(paste0("Unique token: ", length(dic), "\n"))
    return (list(X=X, dic=dic))
}


replicate.data <- function(X, seq.len) {
    num.seq <- as.integer(length(X) / seq.len)
    X <- X[1:(num.seq*seq.len)]
    print
    dim(X) = c(seq.len, num.seq)
    return (X)
}

drop.tail <- function(X, batch.size) {
    shape <- dim(X)
    nstep <- as.integer(shape[2] / batch.size)
    return (X[, 1:(nstep * batch.size)])
}

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

batch.size = 20
seq.len = 35
num.hidden = 200
num.embed = 200
num.lstm.layer = 2
num.round = 15
learning.rate= 0.1
wd=0.00001
update.period = 1


train <- load.data("./data/ptb.train.txt")
X.train <- train$X
dic <- train$dic
val <- load.data("./data/ptb.valid.txt", dic)
X.val <- val$X
dic <- val$dic
X.train.data <- replicate.data(X.train, seq.len)
X.val.data <- replicate.data(X.val, seq.len)
vocab <- length(dic)
cat(paste0("Vocab=", vocab, "\n"))

X.train.data <- drop.tail(X.train.data, batch.size)
X.val.data <- drop.tail(X.val.data, batch.size)
X.train.label <- get.label(X.train.data)
X.val.label <- get.label(X.val.data)
X.train <- list(data=X.train.data, label=X.train.label)
X.val <- list(data=X.val.data, label=X.val.label)

model <- mx.lstm(X.train, X.val, 
                 ctx=mx.gpu(0),
                 num.round=num.round, 
                 update.period=update.period,
                 num.lstm.layer=num.lstm.layer, 
                 seq.len=seq.len,
                 num.hidden=num.hidden, 
                 num.embed=num.embed, 
                 num.label=vocab,
                 batch.size=batch.size, 
                 input.size=vocab,
                 initializer=mx.init.uniform(0.01), 
                 learning.rate=learning.rate,
                 wd=wd)


require(mxnet)

context("lstm models")

get.nll <- function(s) {
    pat <- ".*\\NLL=(.+), Perp=.*"
    nll <- sub(pat, "\\1", s)
    return (as.numeric(nll))
} 

test_that("training error decreasing", {

    # Set basic network parameters.
    batch.size = 2
    seq.len = 2
    num.hidden = 1
    num.embed = 2
    num.lstm.layer = 2
    num.round = 5
    learning.rate= 0.1
    wd=0.00001
    clip_gradient=1
    update.period = 1
    vocab=17

    X.train <- list(data=array(1:16, dim=c(2,8)), label=array(2:17, dim=c(2,8)))

    s <- capture.output(model <- mx.lstm( X.train, 
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
                                          initializer=mx.init.uniform(0.01), 
                                          learning.rate=learning.rate,
                                          wd=wd,
                                          clip_gradient=clip_gradient))

    prev.nll <- 10000000.0
    for (r in s) {
        nll <- get.nll(r)
        expect_true(prev.nll >= nll)
        prev.nll <- nll

    }

})
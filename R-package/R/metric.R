# create a customized metric based on feval(label, pred)
mx.metric.custom <-function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value=(state[[2]]/state[[1]]))
  }
  ret <- (list(init=init, update=update, get=get))
  class(ret) <- "mx.metric"
  return(ret)
}

#' Accuracy metric
#'
#' @export
mx.metric.accuracy <- mx.metric.custom("accuracy", function(label, pred) {
  ypred = max.col(as.array(pred), tie="first")
  return(sum((as.array(label) + 1) == ypred) / length(label))
})

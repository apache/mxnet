#' Helper function to create a customized metric
#'
#' @export
mx.metric.custom <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(as.array(label), as.array(pred))
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

#' Accuracy metric for classification
#'
#' @export
mx.metric.accuracy <- mx.metric.custom("accuracy", function(label, pred) {
  ypred = max.col(t(as.array(pred)), tie="first")
  return(sum((as.array(label) + 1) == ypred) / length(label))
})

#' RMSE (Root Mean Squared Error) metric for regression
#'
#' @export
mx.metric.rmse <- mx.metric.custom("rmse", function(label, pred) {
  res <- sqrt(mean((label-pred)^2))
  return(res)
})

#' MAE (Mean Absolute Error) metric for regression
#'
#' @export
mx.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  res <- mean(abs(label-pred))
  return(res)
})

#' RMSLE (Root Mean Squared Logarithmic Error) metric for regression
#'
#' @export
mx.metric.rmsle <- mx.metric.custom("rmsle", function(label, pred) {
  res <- sqrt(mean((log(pred + 1) - log(label + 1))^2))
  return(res)
})

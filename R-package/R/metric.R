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

#' Helper function for top-k accuracy
is.num.in.vect <- function(vect, num){
  resp <- any(is.element(vect, num))
  return(resp)
}

#' Top-k accuracy metric for classification
#'
#' @export
mx.metric.top_k_accuracy <- mx.metric.custom("top_k_accuracy", function(label, pred, top_k = 5) {
  if(top_k == 1){
    return(mx.metric.accuracy(label,pred))
  } else{
    ypred <- apply(pred,2,function(x) order(x, decreasing=TRUE)[seq_len(top_k)])
    ans <- apply(ypred, 2, is.num.in.vect, num = as.array(label + 1))
    acc <- sum(ans)/length(label)  
    return(acc)
  }
})

#' MSE (Mean Squared Error) metric for regression
#'
#' @export
mx.metric.mse <- mx.metric.custom("mse", function(label, pred) {
  res <- mean((label-pred)^2)
  return(res)
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

#' Perplexity metric for language model
#'
#' @export
mx.metric.Perplexity <- mx.metric.custom("Perplexity", function(label, pred) {
  label_probs <- as.array(mx.nd.choose.element.0index(pred, label))
  batch <- length(label_probs)
  NLL <- -sum(log(pmax(1e-15, as.array(label_probs)))) / batch
  Perplexity <- exp(NLL)
  return(Perplexity)
})

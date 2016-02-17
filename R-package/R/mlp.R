#' Convenience interface for multiple layer perceptron
#' 
#' @param data the input matrix. Only mx.io.DataIter and R array/matrix types supported.
#' @param label the training label. Only R array type supported.
#' @param hidden_node a vector containing number of hidden nodes on each hidden layer as well as the output layer.
#' @param out_node the number of nodes on the output layer.
#' @param dropout a number in [0,1) containing the dropout ratio from the last hidden layer to the output layer.
#' @param activation either a single string or a vector containing the names of the activation functions.
#' @param out_activation a single string containing the name of the output activation function.
#' @param device whether train on cpu (default) or gpu.
#' @param eval_metric the evaluation metric/
#' @param ... other parameters passing to \code{mx.model.FeedForward.create}/
#' 
#' @examples
#' 
#' require(mlbench)
#' data(Sonar, package="mlbench")
#' Sonar[,61] = as.numeric(Sonar[,61])-1
#' train.ind = c(1:50, 100:150)
#' train.x = data.matrix(Sonar[train.ind, 1:60])
#' train.y = Sonar[train.ind, 61]
#' test.x = data.matrix(Sonar[-train.ind, 1:60])
#' test.y = Sonar[-train.ind, 61]
#' model = mx.mlp(train.x, train.y, hidden_node = 10, out_node = 2, out_activation = "softmax", 
#'                learning.rate = 0.1)
#' preds = predict(model, test.x)
#' 
#' @export
mx.mlp <- function(data, label, hidden_node = 1, out_node, dropout = NULL, 
                   activation = "tanh", out_activation = "softmax",
                   device=mx.ctx.default(), ...) {
  
  m <- length(hidden_node)
  if (!is.null(dropout)) {
    if (length(dropout) != 1) {
      stop("only accept dropout ratio of length 1.")
    }
    dropout = max(0,min(dropout, 1-1e-7))
  }
  
  # symbol construction
  act <- mx.symbol.Variable("data")
  if (length(activation) == 1) {
    activation <- rep(activation, m)
  } else {
    if (length(activation) != m) {
      stop(paste("Length of activation should be",m))
    }
  }
  for (i in 1:m) {
    fc <- mx.symbol.FullyConnected(act, num_hidden=hidden_node[i])
    act <- mx.symbol.Activation(fc, act_type=activation[i])
    if (i == m && !is.null(dropout)) {
      act <- mx.symbol.Dropout(act, p = dropout)
    }
  }
  fc <- mx.symbol.FullyConnected(act, num_hidden=out_node)
  if (out_activation == "rmse") {
    out <- mx.symbol.LinearRegressionOutput(fc)
  } else if (out_activation == "softmax") {
    out <- mx.symbol.SoftmaxOutput(fc)
  } else if (out_activation == "logistic") {
    out <- mx.symbol.LogisticRegressionOutput(fc)
  } else {
    stop("Not supported yet.")
  }
  model <- mx.model.FeedForward.create(out, X=data, y=label, ctx=device, ...)
  return(model)
}

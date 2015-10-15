# Log training metric each period
#' @export
mx.callback.log.train.metric <- function(period) {
  function(iteration, nbatch, env) {
    if (nbatch %% period == 0) {
      result <- env$metric$get(env$train.metric)
      cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
    }
  }
}


#
#' @export
mx.callback.save.checkpoint <- function(prefix, period=1) {
  function(iteration, nbatch, env) {
    if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      cat(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
  }
}

# Log training metric each period
mx.callback.log.train.metric <- function(period) {
  function(iteration, nbatch, env) {
    if (nbatch %% period == 0) {
      result <- env$metric$get(env$train.metric)
      cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
    }
  }
}

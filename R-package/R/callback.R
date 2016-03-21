#' @export mx.metric.logger
mx.metric.logger <- setRefClass("mx.metric.logger", fields = list(train = "numeric", eval="numeric"))

#' Log training metric each period
#' @export
mx.callback.log.train.metric <- function(period, logger=NULL) {
  function(iteration, nbatch, env, verbose=TRUE) {
    if (nbatch %% period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0 & verbose)
        cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
      if (!is.null(logger)) {
        if (class(logger) != "mx.metric.logger") {
          stop("Invalid mx.metric.logger.")
        }
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if (nbatch != 0 & verbose)
            cat(paste0("Batch [", nbatch, "] Validation-", result$name, "=", result$value, "\n"))
          logger$eval <- c(logger$eval, result$value)
        }
      }
    }
    return(TRUE)
  }
}


#' Save checkpoint to files each period iteration.
#'
#' @param prefix The prefix of the model checkpoint.
#'
#' @export
mx.callback.save.checkpoint <- function(prefix, period=1) {
  function(iteration, nbatch, env, verbose=TRUE) {
    if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      if(verbose) cat(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
  }
}

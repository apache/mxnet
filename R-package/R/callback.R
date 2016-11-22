#' @export mx.metric.logger
mx.metric.logger <- setRefClass("mx.metric.logger", fields = list(train = "numeric", eval="numeric"))

#' Log training metric each period
#'
#' @param period The number of batch to log the training evaluation metric
#' @param logger The logger class
#'
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

#' Calculate the training speed 
#'
#' @param batch_size The batch size
#' @param frequency The frequency of the training speed update
#'
#' @export
mx.callback.log.speedometer <- function(batch.size, frequency=50){
  function(iteration, nbatch, env, verbose=TRUE) {
    count <- nbatch
    if(is.null(env$count)) env$count <- 0
    if(is.null(env$init)) env$init <- FALSE
    if (env$count > count) env$init <- FALSE
    env$count = count
    if(env$init){
      if (count %% frequency == 0 && !is.null(env$metric)){
        time <- as.double(difftime(Sys.time(), env$tic, units = "secs"))
        speed <- frequency*batch.size/time
        result <- env$metric$get(env$train.metric)
        if (nbatch != 0 & verbose)
          cat(paste0("Batch [", nbatch, "] Speed: ", speed, " samples/sec Train-",
                     result$name, "=", result$value, "\n"))
        env$tic = Sys.time()
      }      
    } else {
      env$init <- TRUE
      env$tic <- Sys.time()
    }
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


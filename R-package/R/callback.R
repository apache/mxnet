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
      if (nbatch != 0 && verbose)
        message("Batch [", nbatch, "] Train-", result$name, "=", result$value)
      if (!is.null(logger)) {
        if (class(logger) != "mx.metric.logger") {
          stop("Invalid mx.metric.logger.")
        }
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if (nbatch != 0 && verbose)
            message("Batch [", nbatch, "] Validation-", result$name, "=", result$value)
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
        if (nbatch != 0 && verbose)
          message("Batch [", nbatch, "] Speed: ", speed, " samples/sec Train-",
                     result$name, "=", result$value)
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
      if(verbose) message(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
  }
}

#' Early stop with different conditions
#' 
#' Early stopping applying different conditions: hard thresholds or epochs number from the best score. Tested with "epoch.end.callback" function.
#' 
#' @param train.metric Numeric. Hard threshold for the metric of the training data set (optional)
#' @param eval.metric Numeric. Hard threshold for the metric of the evaluating data set (if set, optional) 
#' @param bad.steps Integer. How much epochs should gone from the best score? Use this option with evaluation data set
#' @param maximize Logical. Do your model use maximizing or minimizing optimization?
#' @param verbose Logical
#' 
#' @export
#' 
mx.callback.early.stop <- function(train.metric = NULL, eval.metric = NULL, bad.steps = NULL, maximize = FALSE, verbose = FALSE) {
  
  function(iteration, nbatch, env, verbose = verbose) {
    
    # hard threshold for train metric
    if (!is.null(env$metric)) {
      if (!is.null(train.metric)) {
        result <- env$metric$get(env$train.metric)
        if ((! maximize && result$value < train.metric) || (maximize && result$value > train.metric)) {
          return(FALSE)
        }
      }
      
      # hard threshold for test metric
      if (!is.null(eval.metric)) {
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if ((!maximize && result$value < eval.metric) || (maximize && result$value > eval.metric)) {
            return(FALSE)
          }
        }
      }
    }
    
    # not worse than previous X steps
    if (!is.null(bad.steps)) {
      
      # set / reset iteration variables
      # it may be not the best practice to use global variables,
      # but let's not touch "model.r" file
      if (iteration == 1){
        # reset iterator
        mx.best.iter <<- 1
        
        # reset best score
        if (maximize) {
          mx.best.score <<- 0
        }
        else {
          mx.best.score <<- Inf
        }
      }
      
      # test early stop round
      if (!is.null(env$eval.metric)) {
        
        result <- env$metric$get(env$eval.metric)
        
        if ((! maximize && result$value > mx.best.score) || (maximize && result$value < mx.best.score)) {
          
          if (mx.best.iter == bad.steps) {
            if (verbose) {
              message("Best score=", mx.best.score, ", iteration [", iteration - bad.steps, "]")
            }
            return(FALSE)
          } else {
            mx.best.iter <<- mx.best.iter + 1
          }
          
        } else {
          mx.best.score <<- result$value
          mx.best.iter <<- 1
        }
      }
    }
    
    return(TRUE)
  }
}

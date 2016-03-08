
#' @export
FactorScheduler <- function(step, factor_val) {
  function(optimizerEnv){
    num_update <- optimizerEnv$num_update
    count      <- optimizerEnv$count
    lr         <- optimizerEnv$lr

    if (num_update > count + step){
      count = count + step
      lr = lr * factor_val
      cat(paste0("Update[", num_update, "]: learning rate is changed to ", lr, "\n"))
      optimizerEnv$lr <- lr
      optimizerEnv$count <- count
    }
    # return(optimizerEnv)
  }
}


#' @export
FactorScheduler <- function(step, factor_val, stop_factor_lr=1e-8, verbose=TRUE) {
  if(step < 1) stop("Schedule step must be greater or equal than 1 round")
  if(factor_val > 1) stop("Factor must be no more than 1 to make lr reduce")
  function(optimizerEnv){
    num_update <- optimizerEnv$num_update
    count      <- optimizerEnv$count
    lr         <- optimizerEnv$lr
    if (num_update > count + step){
      count <- count + step
      lr    <- lr * factor_val
      if(lr < stop_factor_lr){
        lr <- stop_factor_lr
        if(verbose) cat(paste0("Update[", num_update, 
                               "]: now learning rate arrived at ", lr, 
                               "will not change in the future\n"))
      } else{
        if(verbose) cat(paste0("Update[", num_update, 
                               "]: learning rate is changed to ", lr, "\n"))
      }
      optimizerEnv$lr    <- lr
      optimizerEnv$count <- count      
      
    }
  }
}


BucketIter <- setRefClass("BucketIter", fields = c("buckets", "bucket.names", "batch.size", 
                                                   "data.mask.element", "shuffle", "bucket.plan", "bucketID", "epoch", "batch", "batch.per.bucket", 
                                                   "last.batch.pad", "batch.per.epoch", "seed"), 
                          methods = list(
                            initialize = function(buckets, 
                                                  batch.size, data.mask.element = 0, shuffle = FALSE, seed = 123) {
                              .self$buckets <- buckets
                              .self$bucket.names <- names(.self$buckets)
                              .self$batch.size <- batch.size
                              .self$data.mask.element <- data.mask.element
                              .self$epoch <- 0
                              .self$batch <- 0
                              .self$shuffle <- shuffle
                              .self$batch.per.bucket <- 0
                              .self$batch.per.epoch <- 0
                              .self$bucket.plan <- NULL
                              .self$bucketID <- NULL
                              .self$seed <- seed
                              .self
                            }, reset = function() {
                              buckets_nb <- length(bucket.names)
                              buckets_id <- seq_len(buckets_nb)
                              buckets.size <- sapply(.self$buckets, function(x) {
                                tail(dim(x$data), 1)
                              })
                              .self$batch.per.bucket <- ceiling(buckets.size/.self$batch.size)
                              .self$last.batch.pad <- .self$batch.size - buckets.size %% .self$batch.size
                              .self$last.batch.pad[.self$last.batch.pad == .self$batch.size] <- 0
                              
                              .self$batch.per.epoch <- sum(.self$batch.per.bucket)
                              # Number of batches per epoch given the batch.size
                              .self$batch.per.epoch <- sum(.self$batch.per.bucket)
                              .self$epoch <- .self$epoch + 1
                              .self$batch <- 0
                              
                              if (.self$shuffle) {
                                set.seed(.self$seed)
                                bucket_plan_names <- sample(rep(names(.self$batch.per.bucket), times = .self$batch.per.bucket))
                                .self$bucket.plan <- ave(bucket_plan_names == bucket_plan_names, bucket_plan_names, 
                                                         FUN = cumsum)
                                names(.self$bucket.plan) <- bucket_plan_names
                                # Return first BucketID at reset for initialization of the model
                                .self$bucketID <- .self$bucket.plan[1]
                                
                                .self$buckets <- lapply(.self$buckets, function(x) {
                                  shuffle_id <- sample(tail(dim(x$data), 1))
                                  if (length(dim(x$label)) == 0) {
                                    list(data = x$data[, shuffle_id], label = x$label[shuffle_id])
                                  } else {
                                    list(data = x$data[, shuffle_id], label = x$label[, shuffle_id])
                                  }
                                })
                              } else {
                                bucket_plan_names <- rep(names(.self$batch.per.bucket), times = .self$batch.per.bucket)
                                .self$bucket.plan <- ave(bucket_plan_names == bucket_plan_names, bucket_plan_names, 
                                                         FUN = cumsum)
                                names(.self$bucket.plan) <- bucket_plan_names
                              }
                            }, iter.next = function() {
                              .self$batch <- .self$batch + 1
                              .self$bucketID <- .self$bucket.plan[batch]
                              return(.self$batch <= .self$batch.per.epoch)
                            }, value = function() {
                              # bucketID is a named integer: the integer indicates the batch id for the given
                              # bucket (used to fetch appropriate samples within the bucket) the name is a
                              # character containing the sequence length of the bucket (used to unroll the rnn
                              # to appropriate sequence length)
                              idx <- (.self$bucketID - 1) * (.self$batch.size) + seq_len(batch.size)
                              
                              # Reuse first idx for padding
                              if (bucketID == .self$batch.per.bucket[names(.self$bucketID)] & !.self$last.batch.pad[names(.self$bucketID)] == 0) {
                                idx <- c(idx[seq_len(.self$batch.size - .self$last.batch.pad[names(.self$bucketID)])], seq_len(.self$last.batch.pad[names(.self$bucketID)]))
                              }
                              
                              data <- .self$buckets[[names(.self$bucketID)]]$data[, idx, drop = F]
                              seq.mask <- as.integer(names(bucketID)) - apply(data==.self$data.mask.element, 2, sum)
                              if (length(dim(.self$buckets[[names(.self$bucketID)]]$label)) == 0) {
                                label <- .self$buckets[[names(.self$bucketID)]]$label[idx]
                              } else {
                                label <- .self$buckets[[names(.self$bucketID)]]$label[, idx, drop = F]
                              }
                              return(list(data = mx.nd.array(data), seq.mask = mx.nd.array(seq.mask), 
                                          label = mx.nd.array(label)))
                            }, num.pad = function() {
                              if (bucketID == .self$batch.per.bucket[names(.self$bucketID)] & !.self$last.batch.pad[names(.self$bucketID)] == 0){
                                return(.self$last.batch.pad[names(.self$bucketID)])
                              } else return(0)
                            }, finalize = function() {
                            }))

# 
#' Create Bucket Iter
#'
#' @param buckets The data array.
#' @param batch.size The batch size used to pack the array.
#' @param data.mask.element The element to mask
#' @param shuffle Whether shuffle the data
#' @param seed The random seed
#'
#' @export
mx.io.bucket.iter <- function(buckets, batch.size, data.mask.element = 0, shuffle = FALSE, 
                              seed = 123) {
  return(BucketIter$new(buckets = buckets, batch.size = batch.size, data.mask.element = data.mask.element, 
                        shuffle = shuffle, seed = seed))
}
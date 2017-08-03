BucketIter <- setRefClass("BucketIter", fields = c("buckets", "bucket.names", "batch.size", 
  "data.mask.element", "shuffle", "bucket.plan", "bucketID", "epoch", "batch", 
  "batch.per.epoch", "seed"), contains = "Rcpp_MXArrayDataIter", methods = list(initialize = function(buckets, 
  batch.size, data.mask.element = 0, shuffle = FALSE, seed = 123) {
  .self$buckets <- buckets
  .self$bucket.names <- names(.self$buckets)
  .self$batch.size <- batch.size
  .self$data.mask.element <- data.mask.element
  .self$epoch <- 0
  .self$batch <- 0
  .self$shuffle <- shuffle
  .self$batch.per.epoch <- 0
  .self$bucket.plan <- NULL
  .self$bucketID <- NULL
  .self$seed <- seed
  .self
}, reset = function() {
  buckets_nb <- length(bucket.names)
  buckets_id <- 1:buckets_nb
  buckets_size <- sapply(.self$buckets, function(x) {
    dim(x$data)[length(dim(x$data))]
  })
  batch_per_bucket <- floor(buckets_size/.self$batch.size)
  # Number of batches per epoch given the batch_size
  .self$batch.per.epoch <- sum(batch_per_bucket)
  .self$epoch <- .self$epoch + 1
  .self$batch <- 0
  
  if (.self$shuffle) {
    set.seed(.self$seed)
    bucket_plan_names <- sample(rep(names(batch_per_bucket), times = batch_per_bucket))
    .self$bucket.plan <- ave(bucket_plan_names == bucket_plan_names, bucket_plan_names, 
      FUN = cumsum)
    names(.self$bucket.plan) <- bucket_plan_names
    ### Return first BucketID at reset for initialization of the model
    .self$bucketID <- .self$bucket.plan[1]
    
    .self$buckets <- lapply(.self$buckets, function(x) {
      shuffle_id <- sample(ncol(x$data))
      if (length(dim(x$label)) == 0) {
        list(data = x$data[, shuffle_id], label = x$label[shuffle_id])
      } else {
        list(data = x$data[, shuffle_id], label = x$label[, shuffle_id])
      }
    })
  } else {
    bucket_plan_names <- rep(names(batch_per_bucket), times = batch_per_bucket)
    .self$bucket.plan <- ave(bucket_plan_names == bucket_plan_names, bucket_plan_names, 
      FUN = cumsum)
    names(.self$bucket.plan) <- bucket_plan_names
  }
}, iter.next = function() {
  .self$batch <- .self$batch + 1
  .self$bucketID <- .self$bucket.plan[batch]
  if (.self$batch > .self$batch.per.epoch) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}, value = function() {
  # bucketID is a named integer: the integer indicates the batch id for the given
  # bucket (used to fetch appropriate samples within the bucket) the name is the a
  # character containing the sequence length of the bucket (used to unroll the rnn
  # to appropriate sequence length)
  idx <- (.self$bucketID - 1) * (.self$batch.size) + (1:batch.size)
  data <- .self$buckets[[names(.self$bucketID)]]$data[, idx, drop = F]
  data_mask_array <- (!data == 0)
  if (length(dim(.self$buckets[[names(.self$bucketID)]]$label)) == 0) {
    label <- .self$buckets[[names(.self$bucketID)]]$label[idx]
  } else {
    label <- .self$buckets[[names(.self$bucketID)]]$label[, idx, drop = F]
  }
  return(list(data = mx.nd.array(data), data.mask.array = mx.nd.array(data_mask_array), 
    label = mx.nd.array(label)))
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

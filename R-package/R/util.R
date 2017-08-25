# filter out null, keep the names
mx.util.filter.null <- function(lst) {
  lst[!sapply(lst, is.null)]
}

#' Internal function to generate mxnet_generated.R
#' Users do not need to call this function.
#' @param path The path to the root of the package.
#'
#' @export
mxnet.export <- function(path) {
  mxnet.internal.export(path.expand(path))
}

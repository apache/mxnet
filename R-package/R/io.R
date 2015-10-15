is.MXDataIter <- function(x) {
  inherits(x, "Rcpp_MXNativeDataIter") ||
  inherits(x, "Rcpp_MXArrayDataIter")
}

#' Extract a certain field from DataIter.
#' @export
mx.io.extract <- function(iter, field) {
  packer <- mx.nd.arraypacker.create()
  iter$reset()
  while (iter$iter.next()) {
    dlist <- iter$value()
    padded <- iter$num.pad()
    data <- dlist[[field]]
    oshape <- dim(data)
    packer$push(mx.nd.slice(data, 0, oshape[[1]] - padded))
  }
  return(packer$get())
}

#' Create MXDataIter compatible iterator from R's array
#'
#' @param data The data array.
#' @param label The label array.
#' @param batch.size The batch size used to pack the array.
#' @param shuffle Whether shuffle the data
#'
#' @export
mx.io.ArrayIter <- function(data, label=NULL,
                            batch.size=128,
                            shuffle=FALSE) {
  mx.io.internal.ArrayIter.create(as.array(data),
                                  as.array(label),
                                  batch.size,
                                  shuffle)
}

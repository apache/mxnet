# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#' Judge if an object is mx.dataiter
#'
#' @return Logical indicator
#'
#' @export
is.mx.dataiter <- function(x) {
  inherits(x, "Rcpp_MXNativeDataIter") ||
  inherits(x, "Rcpp_MXArrayDataIter")
}

#' Extract a certain field from DataIter.
#'
#' @export
mx.io.extract <- function(iter, field) {
  packer <- mx.nd.arraypacker()
  iter$reset()
  while (iter$iter.next()) {
    dlist <- iter$value()
    padded <- iter$num.pad()
    data <- dlist[[field]]
    oshape <- dim(data)
    ndim <- length(oshape)
    packer$push(mx.nd.slice(data, 0, oshape[[ndim]] - padded))
  }
  iter$reset()
  return(packer$get())
}

#
#' Create MXDataIter compatible iterator from R's array
#'
#' @param data The data array.
#' @param label The label array.
#' @param batch.size The batch size used to pack the array.
#' @param shuffle Whether shuffle the data
#'
#' @export
mx.io.arrayiter <- function(data, label,
                            batch.size=128,
                            shuffle=FALSE) {
  if (shuffle) {
    shape <- dim(data)
    if (is.null(shape)) {
      num.data <- length(data)
    } else {
      ndim <- length(shape)
      num.data <- shape[[ndim]]
    }
    unif.rnds <- as.array(mx.runif(c(num.data), ctx=mx.cpu()));
  } else {
    unif.rnds <- as.array(0)
  }
  mx.io.internal.arrayiter(as.array(data),
                           as.array(label),
                           unif.rnds,
                           batch.size,
                           shuffle)
}

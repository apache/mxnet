require(methods)

loadModule("mxnet", TRUE)
setOldClass("mx.NDArray")
setMethod("+", signature(e1="mx.NDArray", e2="numeric"), function(e1, e2) {
  mx.nd.internal.plus.scalar(e1, e2)
})

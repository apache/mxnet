require(methods)

.onLoad <- function(libname, pkgname) {
  loadModule("mxnet", TRUE)
  init.ndarray.methods()
}

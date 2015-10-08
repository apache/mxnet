
.onLoad <- function(libname, pkgname) {
  library.dynam("libmxnet", pkgname, libname, local=FALSE)
  library.dynam("mxnet", pkgname, libname)
  loadModule("mxnet", TRUE)
  init.ndarray.methods()
  init.symbol.methods()
}

.onUnload <- function(libpath) {
  unloadModule("mxnet")
  library.dynam.unload("mxnet", libpath)
  library.dynam.unload("libmxnet", libpath)
}

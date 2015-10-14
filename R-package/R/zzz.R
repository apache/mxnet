.onLoad <- function(libname, pkgname) {
  library.dynam("libmxnet", pkgname, libname, local=FALSE)
  library.dynam("mxnet", pkgname, libname)
  loadModule("mxnet", TRUE)
  init.symbol.methods()
  init.context.default()
}

.onUnload <- function(libpath) {
  mx.internal.notify.shutdown()
  unloadModule("mxnet")
  library.dynam.unload("mxnet", libpath)
  library.dynam.unload("libmxnet", libpath)
}

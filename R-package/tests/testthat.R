library(testthat)
library(mxnet)

if (as.integer(Sys.getenv("R_GPU_ENABLE")) == 1) {
  mx.ctx.default(new = mx.gpu())
  message("Using GPU for testing.")
}

test_check("mxnet")

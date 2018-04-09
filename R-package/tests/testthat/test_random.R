require(mxnet)

context("random")

test_that("mx.runif", {
  X <- mx.runif(shape=50000, min=0, max=1, ctx=mx.ctx.default())
  expect_equal(X>=0, mx.nd.ones(50000))
  expect_equal(X<=1, mx.nd.ones(50000))
  sample_mean = mean(as.array(X))
  expect_equal(sample_mean, 0.5, tolerance=1e-2)
})

test_that("mx.rnorm", {
  X <- mx.rnorm(shape=50000, mean=5, sd=0.1, ctx=mx.ctx.default())
  sample_mean = mean(as.array(X))
  sample_sd = sd(as.array(X))
  expect_equal(sample_mean, 5, tolerance=1e-2)
  expect_equal(sample_sd, 0.1, tolerance=1e-2)
})

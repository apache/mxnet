require(mxnet)

context("ndarray")

if (Sys.getenv("R_GPU_ENABLE") != "" & as.integer(Sys.getenv("R_GPU_ENABLE")) == 1) {
  mx.ctx.default(new = mx.gpu())
  message("Using GPU for testing.")
}

test_that("element-wise calculation for vector", {
  x = 1:10
  mat = mx.nd.array(as.array(x), mx.ctx.default())
  expect_equal(x, as.array(mat))
  expect_equal(x + 1, as.array(mat + 1))
  expect_equal(x - 10, as.array(mat - 10))
  expect_equal(x * 20, as.array(mat * 20))
  expect_equal(x / 3, as.array(mat / 3), tolerance = 1e-5)
  expect_equal(-1 - x, as.array(-1 - mat))
  expect_equal(-5 / x, as.array(-5 / mat), tolerance = 1e-5)
  expect_equal(x + x, as.array(mat + mat))
  expect_equal(x / x, as.array(mat / mat))
  expect_equal(x * x, as.array(mat * mat))
  expect_equal(x - x, as.array(mat - mat))
  expect_equal(as.array(1 - mat), as.array(1 - mat))
  
  x <- runif(10,-10, 10)
  nd = mx.nd.array(as.array(x))
  expect_equal(sqrt(abs(x)), as.array(mx.nd.sqrt(mx.nd.abs(nd))), tolerance = 1e-6)
  expect_equal(x ^ 2, as.array(mx.nd.square(nd)), tolerance = 1e-6)
})

test_that("element-wise calculation for matrix", {
  x = matrix(1:4, 2, 2)
  mat = mx.nd.array(as.array(x), mx.ctx.default())
  expect_equal(x, as.array(mat))
  expect_equal(x + 1, as.array(mat + 1))
  expect_equal(x - 10, as.array(mat - 10))
  expect_equal(x * 20, as.array(mat * 20))
  expect_equal(x / 3, as.array(mat / 3), tolerance = 1e-5)
  expect_equal(-1 - x, as.array(-1 - mat))
  expect_equal(-5 / x, as.array(-5 / mat), tolerance = 1e-5)
  expect_equal(x + x, as.array(mat + mat))
  expect_equal(x / x, as.array(mat / mat))
  expect_equal(x * x, as.array(mat * mat))
  expect_equal(x - x, as.array(mat - mat))
  expect_equal(as.array(1 - mat), as.array(1 - mat))
})

test_that("ndarray ones, zeros, save and load", {
  expect_equal(rep(0, 10), as.array(mx.nd.zeros(10)))
  expect_equal(matrix(0, 10, 5), as.array(mx.nd.zeros(c(10, 5))))
  expect_equal(rep(1, 10), as.array(mx.nd.ones(10)))
  expect_equal(matrix(1, 10, 5), as.array(mx.nd.ones(c(10, 5))))
  mat = mx.nd.array(1:20)
  mx.nd.save(mat, 'temp.mat')
  mat2 = mx.nd.load('temp.mat')
  expect_true(is.mx.ndarray(mat2[[1]]))
  expect_equal(as.array(mat), as.array(mat2[[1]]))
  file.remove('temp.mat')
})

test_that("ndarray concatenate", {
  shapes <- matrix(c(2, 3, 4, 2, 2, 2, 4, 2, 2, 1, 4, 2), nrow = 3, byrow = TRUE)
  array_r <- apply(shapes, 2, function(s) { runif(s, -10, 10) })
  array_nd <- apply(array_r, 1, function(s) { mx.nd.array(matrix(s, nrow = 1)) })
  array_nd_concat <- mx.nd.concat(data = array_nd, num_args = 3, dim = 1)
  expect_equal(array_r, as.matrix(array_nd_concat), tolerance = 1e-6)
  
  x1 <- mx.nd.array(c(1:24))
  x2 <- mx.nd.array(c(25:48))
  x3 <- mx.nd.concat(data = c(x1, x2), num_args = 2, dim = 0)
  expect_equal(c(1:48), as.array(x3))
  expect_equal(dim(x3), 48)
  
  x1 <- array(1:24, dim = c(4, 3, 2))
  x2 <- array(25:48, dim = c(4, 3, 2))
  x3 <- c(1:4, 25:28, 5:8, 29:32, 9:12, 33:36, 13:16, 37:40, 17:20, 41:44, 21:24, 45:48)
  y1 <- mx.nd.array(x1)
  y2 <- mx.nd.array(x2)
  y3 <- mx.nd.concat(data = c(y1, y2), num_args = 2, dim = 2)
  expect_equal(dim(y3), c(8, 3, 2))
  expect_equal(as.array(y3), array(x3, dim = c(8, 3, 2)))
})

test_that("ndarray clip", {
  nd <- mx.nd.array(runif(10,-10, 10))
  nd2 <- mx.nd.clip(nd,-2, 3)
  arr <- as.array(nd2)
  expect_equal(arr >= -2 | arr <= 3, rep(TRUE, length(arr)))
})

test_that("ndarray dot", {
  a <- matrix(runif(12), nrow = 3)
  b <- matrix(runif(20), nrow = 4)
  c <- a %*% b
  
  A <- mx.nd.array(t(a))
  B <- mx.nd.array(t(b))
  C <- mx.nd.dot(A, B)
  
  expect_equal(c, t(as.matrix(C)), tolerance = 1e-6)
})

test_that("ndarray crop", {
  x <- mx.nd.ones(c(2, 3, 4))
  y <- mx.nd.crop(x, begin = c(0, 0, 0), end = c(2, 1, 3))
  expect_equal(array(1, dim = c(2, 1, 3)), as.array(y))
  
  z <- mx.nd.zeros(c(2, 1, 3))
  x <- mxnet:::mx.nd.internal.crop.assign(x, z, begin = c(0, 0, 0), end = c(2, 1, 3))
  arr_x <- array(1, dim = dim(x))
  arr_x[c(1:2), 1 , c(1:3)] <- 0
  
  expect_equal(as.array(x), arr_x)
})

test_that("ndarray negate", {
  arr <- array(runif(24, -10, 10), dim = c(2, 3, 4))
  nd <- mx.nd.array(arr)
  
  expect_equal(arr, as.array(nd), tolerance = 1e-6)
  expect_equal(-arr, as.array(-nd), tolerance = 1e-6)
  expect_equal(arr, as.array(nd), tolerance = 1e-6)
})

test_that("ndarray equal", {
  x <- mx.nd.zeros(c(2, 3))
  y <- mx.nd.ones(c(2, 3))
  z = x == y
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = 0 == x
  expect_equal(as.array(z), array(1, c(2,3)))
})

test_that("ndarray not equal", {
  x <- mx.nd.zeros(c(2, 3))
  y <- mx.nd.ones(c(2, 3))
  z = x != y
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = 0 != x
  expect_equal(as.array(z), array(0, c(2,3)))
})

test_that("ndarray greater", {
  x <- mx.nd.zeros(c(2, 3))
  y <- mx.nd.ones(c(2, 3))
  z = x > y
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = y > 0
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = 0 > y
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = x >= y
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = y >= 0
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = 0 >= y
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = y >= 1
  expect_equal(as.array(z), array(1, c(2,3)))
})

test_that("ndarray lesser", {
  x <- mx.nd.zeros(c(2, 3))
  y <- mx.nd.ones(c(2, 3))
  z = x < y
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = y < 0
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = 0 < y
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = x <= y
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = y <= 0
  expect_equal(as.array(z), array(0, c(2,3)))
  
  z = 0 <= y
  expect_equal(as.array(z), array(1, c(2,3)))
  
  z = y <= 1
  expect_equal(as.array(z), array(1, c(2,3)))
})
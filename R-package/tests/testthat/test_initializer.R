require(mxnet)

context("initializer")

testthat("mx.init.uniform", {
  uniform_init <- mx.init.uniform(scale = 1)
  expect_equal(typeof(uniform_init), "closure")
  
  X_bias <- uniform_init("X_bias", c(1, 10000), ctx = mx.ctx.default())
  expect_equal(X_bias, mx.nd.zeros(c(1, 10000)))
  
  X_weight <- uniform_init("X_weight", c(5, 10, 10000), ctx = mx.ctx.default())
  expect_equal(X_weight >= -1, mx.nd.ones(c(5, 10, 10000)))
  expect_equal(X_weight <= 1, mx.nd.ones(c(5, 10, 10000)))
  mean_weight <- mean(as.array(X_weight))
  expect_equal(mean_weight, 0, tolerance = 0.01)
})

testthat("mx.init.normal", {
  normal_init <- mx.init.normal(sd = 0.1)
  expect_equal(typeof(normal_init), "closure")
  
  X_bias <- normal_init("X_bias", c(1, 10000), ctx = mx.ctx.default())
  expect_equal(X_bias, mx.nd.zeros(c(1, 10000)))
  
  X_weight <- normal_init("X_weight", c(5, 10, 10000), ctx = mx.ctx.default())
  weight_mean <- mean(as.array(X_weight))
  weight_sd <- sd(as.array(X_weight))
  expect_equal(weight_mean, 0, tolerance = 0.01)
  expect_equal(weight_sd, 0.1, tolerance = 0.01)
})

testthat("mx.init.Xavier", {
  xavier_init <- mx.init.Xavier()
  expect_equal(typeof(xavier_init), "closure")
  
  # default parameters
  shape <- c(2, 3, 324, 324)
  fan_out <- shape[length(shape)]
  fan_in <- prod(shape[-length(shape)])
  
  X_bias <- xavier_init("X_bias", shape = shape, ctx = mx.ctx.default())
  expect_equal(X_bias, mx.nd.zeros(shape))
  
  X_weight <- xavier_init("X_weight", shape = shape, ctx = mx.ctx.default())
  scale <- sqrt(3/((fan_in + fan_out)/2))
  expect_equal(X_weight >= -scale, mx.nd.ones(shape))
  expect_equal(X_weight <= scale, mx.nd.ones(shape))
  weight_mean <- mean(as.array(X_weight))
  expect_equal(weight_mean, 0, tolerance = 0.01)
  
  for (dist_type in c("gaussian", "uniform")) {
    for (factor_type in c("in", "out", "avg")) {
      xavier_init <- mx.init.Xavier(rnd_type = dist_type, factor_type = factor_type, 
        magnitude = 200)
      expect_equal(typeof(xavier_init), "closure")
      
      X_weight <- xavier_init("X_weight", shape = shape, ctx = mx.ctx.default())
      factor_val <- switch(factor_type, avg = (fan_in + fan_out)/2, `in` = fan_in, 
        out = fan_out)
      scale <- sqrt(200/factor_val)
      
      if (dist_type == "gaussian") {
        weight_mean <- mean(as.array(X_weight))
        weight_sd <- sd(as.array(X_weight))
        expect_equal(weight_mean, 0, tolerance = 0.01)
        expect_equal(weight_sd, scale, tolerance = 0.01)
      } else {
        expect_equal(X_weight >= -scale, mx.nd.ones(shape))
        expect_equal(X_weight <= scale, mx.nd.ones(shape))
        weight_mean <- mean(as.array(X_weight))
        expect_equal(weight_mean, 0, tolerance = 0.01)
      }
    }
  }
})

testthat("mx.init.internal.default", {
  sample_bias <- mxnet:::mx.init.internal.default("X_bias", c(5, 10, 100), ctx = mx.ctx.default())
  expect_equal(sample_bias, mx.nd.zeros(c(5, 10, 100)))
  
  sample_gamma <- mxnet:::mx.init.internal.default("X_gamma", c(5, 10, 100), ctx = mx.ctx.default())
  expect_equal(sample_gamma, mx.nd.ones(c(5, 10, 100)))
  
  sample_beta <- mxnet:::mx.init.internal.default("X_beta", c(5, 10, 100), ctx = mx.ctx.default())
  expect_equal(sample_beta, mx.nd.zeros(c(5, 10, 100)))
  
  sample_moving_mean <- mxnet:::mx.init.internal.default("X_moving_mean", c(5, 
    10, 100), ctx = mx.ctx.default())
  expect_equal(sample_moving_mean, mx.nd.zeros(c(5, 10, 100)))
  
  sample_moving_var <- mxnet:::mx.init.internal.default("X_moving_var", c(5, 10, 
    100), ctx = mx.ctx.default())
  expect_equal(sample_moving_var, mx.nd.ones(c(5, 10, 100)))
  
  expect_error(mxnet:::mx.init.internal.default("X", c(5, 10, 100), ctx = mx.ctx.default()), 
    "Unkown initialization pattern for  X")
})

testthat("mx.init.create", {
  uniform_init <- mx.init.uniform(scale = 1)
  expect_equal(typeof(uniform_init), "closure")
  arrs <- setNames(as.list(c(50000, 100)), c("X_weight", "X_bias"))
  arr_init <- mx.init.create(uniform_init, arrs, ctx = mx.ctx.default())
  
  X_bias <- arr_init$X_bias
  expect_equal(X_bias, mx.nd.zeros(c(100)))
  
  X_weight <- arr_init$X_weight
  expect_equal(X_weight >= -1, mx.nd.ones(c(50000)))
  expect_equal(X_weight <= 1, mx.nd.ones(c(50000)))
  mean_weight <- mean(as.array(X_weight))
  expect_equal(mean_weight, 0, tolerance = 0.01)
})

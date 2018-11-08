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

require("imager")
require("dplyr")
require("readr")
require("mxnet")

source("iterators.R")

### Data import and preperation 
# First download MNIST train data at Kaggle:
# https://www.kaggle.com/c/digit-recognizer/data

train <- read_csv("data/train.csv")
train <- data.matrix(train)

train_data <- train[, -1]
train_data <- t(train_data/255 * 2 - 1)
train_label <- as.integer(train[, 1])

dim(train_data) <- c(28, 28, 1, ncol(train_data))

### Model parameters
random_dim <- 96
gen_features <- 96
dis_features <- 32
image_depth <- 1
fix_gamma <- T
no_bias <- T
eps <- 1e-05 + 1e-12
batch_size <- 64


### Generator Symbol
data <- mx.symbol.Variable("data")

gen_rand <- mx.symbol.normal(loc = 0, scale = 1, shape = c(1, 1, random_dim, batch_size), 
  name = "gen_rand")
gen_concat <- mx.symbol.concat(data = list(data, gen_rand), num.args = 2, name = "gen_concat")

g1 <- mx.symbol.Deconvolution(gen_concat, name = "g1", kernel = c(4, 4), num_filter = gen_features * 
  4, no_bias = T)
gbn1 <- mx.symbol.BatchNorm(g1, name = "gbn1", fix_gamma = fix_gamma, eps = eps)
gact1 <- mx.symbol.Activation(gbn1, name = "gact1", act_type = "relu")

g2 <- mx.symbol.Deconvolution(gact1, name = "g2", kernel = c(3, 3), stride = c(2, 
  2), pad = c(1, 1), num_filter = gen_features * 2, no_bias = no_bias)
gbn2 <- mx.symbol.BatchNorm(g2, name = "gbn2", fix_gamma = fix_gamma, eps = eps)
gact2 <- mx.symbol.Activation(gbn2, name = "gact2", act_type = "relu")

g3 <- mx.symbol.Deconvolution(gact2, name = "g3", kernel = c(4, 4), stride = c(2, 
  2), pad = c(1, 1), num_filter = gen_features, no_bias = no_bias)
gbn3 <- mx.symbol.BatchNorm(g3, name = "gbn3", fix_gamma = fix_gamma, eps = eps)
gact3 <- mx.symbol.Activation(gbn3, name = "gact3", act_type = "relu")

g4 <- mx.symbol.Deconvolution(gact3, name = "g4", kernel = c(4, 4), stride = c(2, 
  2), pad = c(1, 1), num_filter = image_depth, no_bias = no_bias)
G_sym <- mx.symbol.Activation(g4, name = "G_sym", act_type = "tanh")


### Discriminator Symbol
data <- mx.symbol.Variable("data")
dis_digit <- mx.symbol.Variable("digit")
label <- mx.symbol.Variable("label")

dis_digit <- mx.symbol.Reshape(data = dis_digit, shape = c(1, 1, 10, batch_size), 
  name = "digit_reshape")
dis_digit <- mx.symbol.broadcast_to(data = dis_digit, shape = c(28, 28, 10, batch_size), 
  name = "digit_broadcast")

data_concat <- mx.symbol.concat(list(data, dis_digit), num.args = 2, dim = 1, name = "dflat_concat")

d1 <- mx.symbol.Convolution(data = data_concat, name = "d1", kernel = c(3, 3), stride = c(1, 
  1), pad = c(0, 0), num_filter = 24, no_bias = no_bias)
dbn1 <- mx.symbol.BatchNorm(d1, name = "dbn1", fix_gamma = fix_gamma, eps = eps)
dact1 <- mx.symbol.LeakyReLU(dbn1, name = "dact1", act_type = "elu", slope = 0.25)
pool1 <- mx.symbol.Pooling(data = dact1, name = "pool1", pool_type = "max", kernel = c(2, 
  2), stride = c(2, 2), pad = c(0, 0))

d2 <- mx.symbol.Convolution(pool1, name = "d2", kernel = c(3, 3), stride = c(2, 2), 
  pad = c(0, 0), num_filter = 32, no_bias = no_bias)
dbn2 <- mx.symbol.BatchNorm(d2, name = "dbn2", fix_gamma = fix_gamma, eps = eps)
dact2 <- mx.symbol.LeakyReLU(dbn2, name = "dact2", act_type = "elu", slope = 0.25)

d3 <- mx.symbol.Convolution(dact2, name = "d3", kernel = c(3, 3), stride = c(1, 1), 
  pad = c(0, 0), num_filter = 64, no_bias = no_bias)
dbn3 <- mx.symbol.BatchNorm(d3, name = "dbn3", fix_gamma = fix_gamma, eps = eps)
dact3 <- mx.symbol.LeakyReLU(dbn3, name = "dact3", act_type = "elu", slope = 0.25)

d4 <- mx.symbol.Convolution(dact2, name = "d3", kernel = c(4, 4), stride = c(1, 1), 
  pad = c(0, 0), num_filter = 64, no_bias = no_bias)
dbn4 <- mx.symbol.BatchNorm(d4, name = "dbn4", fix_gamma = fix_gamma, eps = eps)
dact4 <- mx.symbol.LeakyReLU(dbn4, name = "dact4", act_type = "elu", slope = 0.25)

# pool4 <- mx.symbol.Pooling(data=dact3, name='pool4', pool_type='avg',
# kernel=c(4,4), stride=c(1,1), pad=c(0,0))

dflat <- mx.symbol.Flatten(dact4, name = "dflat")

dfc <- mx.symbol.FullyConnected(data = dflat, name = "dfc", num_hidden = 1, no_bias = F)
D_sym <- mx.symbol.LogisticRegressionOutput(data = dfc, label = label, name = "D_sym")


### Graph
input_shape_G <- c(1, 1, 10, batch_size)
input_shape_D <- c(28, 28, 1, batch_size)

graph.viz(G_sym, type = "graph", direction = "LR")
graph.viz(D_sym, type = "graph", direction = "LR")


### Training module for GAN

# Change this to mx.gpu() when running on gpu machine.
devices <- mx.cpu()

data_shape_G <- c(1, 1, 10, batch_size)
data_shape_D <- c(28, 28, 1, batch_size)
digit_shape_D <- c(10, batch_size)

mx.metric.binacc <- mx.metric.custom("binacc", function(label, pred) {
  res <- mean(label == round(pred))
  return(res)
})

mx.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
  res <- mean(label * log(pred) + (1 - label) * log(1 - pred))
  return(res)
})

### Define iterators
iter_G <- G_iterator(batch_size = batch_size)
iter_D <- D_iterator(batch_size = batch_size)

exec_G <- mx.simple.bind(symbol = G_sym, data = data_shape_G, ctx = devices, grad.req = "write")
exec_D <- mx.simple.bind(symbol = D_sym, data = data_shape_D, digit = digit_shape_D, 
  ctx = devices, grad.req = "write")

### initialize parameters - To Do - personalise each layer
initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

arg_param_ini_G <- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, 
  data = data_shape_G)$arg.shapes, ctx = devices)
aux_param_ini_G <- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, 
  data = data_shape_G)$aux.shapes, ctx = devices)

arg_param_ini_D <- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, 
  data = data_shape_D, digit = digit_shape_D)$arg.shapes, ctx = devices)

aux_param_ini_D <- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, 
  data = data_shape_D, digit = digit_shape_D)$aux.shapes, ctx = devices)

mx.exec.update.arg.arrays(exec_G, arg_param_ini_G, match.name = TRUE)
mx.exec.update.aux.arrays(exec_G, aux_param_ini_G, match.name = TRUE)

mx.exec.update.arg.arrays(exec_D, arg_param_ini_D, match.name = TRUE)
mx.exec.update.aux.arrays(exec_D, aux_param_ini_D, match.name = TRUE)

input_names_G <- mxnet:::mx.model.check.arguments(G_sym)
input_names_D <- mxnet:::mx.model.check.arguments(D_sym)


### initialize optimizers
optimizer_G <- mx.opt.create(name = "adadelta", rho = 0.92, epsilon = 1e-06, wd = 0, 
  rescale.grad = 1/batch_size, clip_gradient = 1)

updater_G <- mx.opt.get.updater(optimizer = optimizer_G, weights = exec_G$ref.arg.arrays, 
  ctx = devices)

optimizer_D <- mx.opt.create(name = "adadelta", rho = 0.92, epsilon = 1e-06, wd = 0, 
  rescale.grad = 1/batch_size, clip_gradient = 1)

updater_D <- mx.opt.get.updater(optimizer = optimizer_D, weights = exec_D$ref.arg.arrays, 
  ctx = devices)

### initialize metric
metric_G <- mx.metric.binacc
metric_G_value <- metric_G$init()

metric_D <- mx.metric.binacc
metric_D_value <- metric_D$init()

iteration <- 1
iter_G$reset()
iter_D$reset()


for (iteration in 1:2400) {
  
  iter_G$iter.next()
  iter_D$iter.next()
  
  ### Random input to Generator to produce fake sample
  G_values <- iter_G$value()
  G_data <- G_values[input_names_G]
  mx.exec.update.arg.arrays(exec_G, arg.arrays = G_data, match.name = TRUE)
  mx.exec.forward(exec_G, is.train = T)
  
  ### Feed Discriminator with Concatenated Generator images and real images Random
  ### input to Generator
  D_data_fake <- exec_G$ref.outputs$G_sym_output
  D_digit_fake <- G_values$data %>% mx.nd.Reshape(shape = c(-1, batch_size))
  
  D_values <- iter_D$value()
  D_data_real <- D_values$data
  D_digit_real <- D_values$digit
  
  ### Train loop on fake
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data = D_data_fake, digit = D_digit_fake, 
    label = mx.nd.array(rep(0, batch_size))), match.name = TRUE)
  mx.exec.forward(exec_D, is.train = T)
  mx.exec.backward(exec_D)
  update_args_D <- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null = TRUE)
  
  metric_D_value <- metric_D$update(label = as.array(mx.nd.array(rep(0, batch_size))), 
    pred = as.array(exec_D$ref.outputs[["D_sym_output"]]), metric_D_value)
  
  ### Train loop on real
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data = D_data_real, digit = D_digit_real, 
    label = mx.nd.array(rep(1, batch_size))), match.name = TRUE)
  mx.exec.forward(exec_D, is.train = T)
  mx.exec.backward(exec_D)
  update_args_D <- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null = TRUE)
  
  metric_D_value <- metric_D$update(label = as.array(mx.nd.array(rep(1, batch_size))), 
    pred = as.array(exec_D$ref.outputs[["D_sym_output"]]), metric_D_value)
  
  ### Update Generator weights - use a seperate executor for writing data gradients
  exec_D_back <- mxnet:::mx.symbol.bind(symbol = D_sym, arg.arrays = exec_D$arg.arrays, 
    aux.arrays = exec_D$aux.arrays, grad.reqs = rep("write", length(exec_D$arg.arrays)), 
    ctx = devices)
  mx.exec.update.arg.arrays(exec_D_back, arg.arrays = list(data = D_data_fake, 
    digit = D_digit_fake, label = mx.nd.array(rep(1, batch_size))), match.name = TRUE)
  mx.exec.forward(exec_D_back, is.train = T)
  mx.exec.backward(exec_D_back)
  D_grads <- exec_D_back$ref.grad.arrays$data
  mx.exec.backward(exec_G, out_grads = D_grads)
  
  update_args_G <- updater_G(weight = exec_G$ref.arg.arrays, grad = exec_G$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_G, update_args_G, skip.null = TRUE)
  
  ### Update metrics metric_G_value <- metric_G$update(values[[label_name]],
  ### exec_G$ref.outputs[[output_name]], metric_G_value)
  
  if (iteration%%25 == 0) {
    D_metric_result <- metric_D$get(metric_D_value)
    cat(paste0("[", iteration, "] ", D_metric_result$name, ": ", D_metric_result$value, 
      "\n"))
  }
  
  if (iteration == 1 | iteration%%100 == 0) {
    
    metric_D_value <- metric_D$init()
    
    par(mfrow = c(3, 3), mar = c(0.1, 0.1, 0.1, 0.1))
    for (i in 1:9) {
      img <- as.array(exec_G$ref.outputs$G_sym_output)[, , , i]
      plot(as.cimg(img), axes = F)
    }
    
    print(as.numeric(as.array(G_values$digit)))
    print(as.numeric(as.array(D_values$label)))
    
  }
}

ifelse(!dir.exists(file.path(".", "models")), dir.create(file.path(".", "models")), 
  "Folder already exists")
mx.symbol.save(D_sym, filename = "models/D_sym_model_v1.json")
mx.nd.save(exec_D$arg.arrays, filename = "models/D_aux_params_v1.params")
mx.nd.save(exec_D$aux.arrays, filename = "models/D_aux_params_v1.params")

mx.symbol.save(G_sym, filename = "models/G_sym_model_v1.json")
mx.nd.save(exec_G$arg.arrays, filename = "models/G_arg_params_v1.params")
mx.nd.save(exec_G$aux.arrays, filename = "models/G_aux_params_v1.params")


### Inference
G_sym <- mx.symbol.load("models/G_sym_model_v1.json")
G_arg_params <- mx.nd.load("models/G_arg_params_v1.params")
G_aux_params <- mx.nd.load("models/G_aux_params_v1.params")

digit <- mx.nd.array(rep(9, times = batch_size))
data <- mx.nd.one.hot(indices = digit, depth = 10)
data <- mx.nd.reshape(data = data, shape = c(1, 1, -1, batch_size))

exec_G <- mx.simple.bind(symbol = G_sym, data = data_shape_G, ctx = devices, grad.req = "null")
mx.exec.update.arg.arrays(exec_G, G_arg_params, match.name = TRUE)
mx.exec.update.arg.arrays(exec_G, list(data = data), match.name = TRUE)
mx.exec.update.aux.arrays(exec_G, G_aux_params, match.name = TRUE)

mx.exec.forward(exec_G, is.train = F)

par(mfrow = c(3, 3), mar = c(0.1, 0.1, 0.1, 0.1))
for (i in 1:9) {
  img <- as.array(exec_G$ref.outputs$G_sym_output)[, , , i]
  plot(as.cimg(img), axes = F)
}

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

#####################################################
### Training module for GAN
#####################################################

devices<- mx.cpu()

data_shape_G<- c(1, 1, 10, batch_size)
data_shape_D<- c(28, 28, 1, batch_size)
digit_shape_D<- c(10, batch_size)

mx.metric.binacc <- mx.metric.custom("binacc", function(label, pred) {
  res <- mean(label==round(pred))
  return(res)
})

mx.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
  res <- mean(label*log(pred)+(1-label)*log(1-pred))
  return(res)
})

##############################################
### Define iterators
iter_G<- G_iterator(batch_size = batch_size)
iter_D<- D_iterator(batch_size = batch_size)

exec_G<- mx.simple.bind(symbol = G_sym, data=data_shape_G, ctx = devices, grad.req = "write")
exec_D<- mx.simple.bind(symbol = D_sym, data=data_shape_D, digit=digit_shape_D, ctx = devices, grad.req = "write")

### initialize parameters - To Do - personalise each layer
initializer<- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

arg_param_ini_G<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, data=data_shape_G)$arg.shapes, ctx = mx.cpu())
aux_param_ini_G<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(G_sym, data=data_shape_G)$aux.shapes, ctx = mx.cpu())

arg_param_ini_D<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, data=data_shape_D, digit=digit_shape_D)$arg.shapes, ctx = mx.cpu())
aux_param_ini_D<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(D_sym, data=data_shape_D, digit=digit_shape_D)$aux.shapes, ctx = mx.cpu())

mx.exec.update.arg.arrays(exec_G, arg_param_ini_G, match.name=TRUE)
mx.exec.update.aux.arrays(exec_G, aux_param_ini_G, match.name=TRUE)

mx.exec.update.arg.arrays(exec_D, arg_param_ini_D, match.name=TRUE)
mx.exec.update.aux.arrays(exec_D, aux_param_ini_D, match.name=TRUE)

input_names_G <- mxnet:::mx.model.check.arguments(G_sym)
input_names_D <- mxnet:::mx.model.check.arguments(D_sym)


###################################################
#initialize optimizers
optimizer_G<-mx.opt.create(name = "adadelta",
                           rho=0.92, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)

updater_G<- mx.opt.get.updater(optimizer = optimizer_G, weights = exec_G$ref.arg.arrays)

optimizer_D<-mx.opt.create(name = "adadelta",
                           rho=0.92, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)
updater_D<- mx.opt.get.updater(optimizer = optimizer_D, weights = exec_D$ref.arg.arrays)

####################################
#initialize metric
metric_G<- mx.metric.binacc
metric_G_value<- metric_G$init()

metric_D<- mx.metric.binacc
metric_D_value<- metric_D$init()

iteration<- 1
iter_G$reset()
iter_D$reset()


for (iteration in 1:2400) {
  
  iter_G$iter.next()
  iter_D$iter.next()
  
  ### Random input to Generator to produce fake sample
  G_values <- iter_G$value()
  G_data <- G_values[input_names_G]
  mx.exec.update.arg.arrays(exec_G, arg.arrays = G_data, match.name=TRUE)
  mx.exec.forward(exec_G, is.train=T)
  
  ### Feed Discriminator with Concatenated Generator images and real images
  ### Random input to Generator
  D_data_fake <- exec_G$ref.outputs$G_sym_output
  D_digit_fake <- G_values$data %>% mx.nd.Reshape(shape=c(-1, batch_size))
  
  D_values <- iter_D$value()
  D_data_real <- D_values$data
  D_digit_real <- D_values$digit
  
  ### Train loop on fake
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_data_fake, digit=D_digit_fake, label=mx.nd.array(rep(0, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(label = mx.nd.array(rep(0, batch_size)), exec_D$ref.outputs[["D_sym_output"]], metric_D_value)
  
  ### Train loop on real
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_data_real, digit=D_digit_real, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(mx.nd.array(rep(1, batch_size)), exec_D$ref.outputs[["D_sym_output"]], metric_D_value)
  
  ### Update Generator weights - use a seperate executor for writing data gradients
  exec_D_back<- mxnet:::mx.symbol.bind(symbol = D_sym, arg.arrays = exec_D$arg.arrays, aux.arrays = exec_D$aux.arrays, grad.reqs = rep("write", length(exec_D$arg.arrays)), ctx = devices)
  mx.exec.update.arg.arrays(exec_D_back, arg.arrays = list(data=D_data_fake, digit=D_digit_fake, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D_back, is.train=T)
  mx.exec.backward(exec_D_back)
  D_grads<- exec_D_back$ref.grad.arrays$data
  mx.exec.backward(exec_G, out_grads=D_grads)
  
  update_args_G<- updater_G(weight = exec_G$ref.arg.arrays, grad = exec_G$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_G, update_args_G, skip.null=TRUE)
  
  ### Update metrics
  #metric_G_value <- metric_G$update(values[[label_name]], exec_G$ref.outputs[[output_name]], metric_G_value)
  
  if (iteration %% 25==0){
    D_metric_result <- metric_D$get(metric_D_value)
    cat(paste0("[", iteration, "] ", D_metric_result$name, ": ", D_metric_result$value, "\n"))
  }
  
  if (iteration==1 | iteration %% 100==0){
    
    metric_D_value<- metric_D$init()
    
    par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
    for (i in 1:9) {
      img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
      plot(as.cimg(img), axes=F)
    }

    print(as.numeric(as.array(G_values$digit)))
    print(as.numeric(as.array(D_values$label)))
    
  }
}

mx.symbol.save(D_sym, filename = "models/D_sym_model_v1.json")
mx.nd.save(exec_D$arg.arrays, filename = "models/D_aux_params_v1.params")
mx.nd.save(exec_D$aux.arrays, filename = "models/D_aux_params_v1.params")

mx.symbol.save(G_sym, filename = "models/G_sym_model_v1.json")
mx.nd.save(exec_G$arg.arrays, filename = "models/G_arg_params_v1.params")
mx.nd.save(exec_G$aux.arrays, filename = "models/G_aux_params_v1.params")


### Inference
G_sym<- mx.symbol.load("models/G_sym_model_v1.json")
G_arg_params<- mx.nd.load("models/G_arg_params_v1.params")
G_aux_params<- mx.nd.load("models/G_aux_params_v1.params")

digit<- mx.nd.array(rep(9, times=batch_size))
data<- mx.nd.one.hot(indices = digit, depth = 10)
data<- mx.nd.reshape(data = data, shape = c(1,1,-1, batch_size))

exec_G<- mx.simple.bind(symbol = G_sym, data=data_shape_G, ctx = devices, grad.req = "null")
mx.exec.update.arg.arrays(exec_G, G_arg_params, match.name=TRUE)
mx.exec.update.arg.arrays(exec_G, list(data=data), match.name=TRUE)
mx.exec.update.aux.arrays(exec_G, G_aux_params, match.name=TRUE)

mx.exec.forward(exec_G, is.train=F)

par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:9) {
  img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
  plot(as.cimg(img), axes=F)
}

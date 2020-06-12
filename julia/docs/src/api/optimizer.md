<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Optimizers

Says, you have the parameter `W` inited for your model and
got its gradient stored as `∇` (perhaps from AutoGrad APIs).
Here is minimal snippet of getting your parameter `W` baked by `SGD`.

```@repl
using MXNet

opt = SGD(η = 10)
decend! = getupdater(opt)

W = NDArray(Float32[1, 2, 3, 4]);
∇ = NDArray(Float32[.1, .2, .3, .4]);

decend!(1, ∇, W)
```

```@autodocs
Modules = [MXNet.mx, MXNet.mx.LearningRate, MXNet.mx.Momentum]
Pages = ["optimizer.jl"]
```

## Built-in optimizers

### Stochastic Gradient Descent
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/sgd.jl"]
```

### ADAM
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/adam.jl"]
```

### AdaGrad
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/adagrad.jl"]
```

### AdaDelta
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/adadelta.jl"]
```

### AdaMax
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/adamax.jl"]
```

### RMSProp
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/rmsprop.jl"]
```

### Nadam
```@autodocs
Modules = [MXNet.mx]
Pages = ["optimizers/nadam.jl"]
```

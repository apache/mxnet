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

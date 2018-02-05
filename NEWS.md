# v0.4.0 (#TBD)

* Following material from `mx` module got exported (#TBD):
    * `NDArray`
        * `clip()`
        * `clip!()`
        * `context()`
        * `empty()`
        * `expand_dims()`
        * `@inplace`
        * `σ()`
        * `sigmoid()`
        * `relu()`
        * `softmax()`
        * `log_softmax()`
        * `broadcast_to()`
        * `broadcast_axis()`
        * `broadcast_axes()`

    * `SymbolicNode`
        * `Variable`
        * `@var`

    * `Context`
        * `cpu()`
        * `gpu()`

    * `AbstractModel`
        * `FeedForward`
        * `predict()`

    * `MLP`

    * `Executor`
        * `bind()`
        * `simple_bind()`
        * `forward()`
        * `backward()`

    * `AbstractEvalMetric`
        * `ACE`
        * `Accuracy`
        * `MSE`
        * `MultiACE`
        * `MultiMetric`
        * `NMSE`
        * `SeqMetric`

    * `KVStore`
        * `init!()`
        * `push!()`
        * `pull!()`
        * `barrier()`
        * `set_updater()`
        * `set_optimizer()`

    * `AbstractInitializer`
        * `UniformInitializer`
        * `NormalInitializer`
        * `XavierInitializer`

    * `AbstractOptimizer`
        * `AdaDelta`
        * `AdaGrad`
        * `ADAM`
        * `AdaMax`
        * `Nadam`
        * `RMSProp`
        * `SGD`
        * `getupdater()`
        * `normgrad!()`
        * `update!()`

    * `AbstractDataProvider`
        * `AbstractDataBatch`
        * `ArrayDataProvider`
        * `ArrayDataBatch`

    * `to_graphviz()`

## New APIs

### `SymbolicNode`

* `mx.get_children` for exploring the graph programmatically. (#TBD)

* A handy macro `@mx.var` for creating `mx.Variable`. (#TBD)

  ```julia
  julia> x = @mx.var x
  MXNet.mx.SymbolicNode x

  julia> x, y, z = @mx.var x y z
  (MXNet.mx.SymbolicNode x, MXNet.mx.SymbolicNode y, MXNet.mx.SymbolicNode z)
  ```

### `NDArray`

* A handy constructor: `NDArray(Type, AbstractArray)` is added. (#TBD)

  E.g.
  ```julia
  julia> NDArray([1, 2, 3])
  3-element mx.NDArray{Int64,1} @ CPU0:
   1
   2
   3

  julia> NDArray(Float32, [1, 2, 3])
  3-element mx.NDArray{Float32,1} @ CPU0:
   1.0
   2.0
   3.0
  ```

* A port of Python's `autograd` for `NDArray` (#274)

* `size(x, dims...)` is supported now. (#TBD)

  ```julia
  julia> x = mx.NDArray([1 2; 3 4; 5 6])
  3×2 mx.NDArray{Int64,2} @ CPU0:
   1  2
   3  4
   5  6

  julia> size(x, 1, 2, 3, 4)
  (3, 2, 1, 1)

  ```

* `copy(AbstractArray, context)` is implemented now. (#TBD)

  ```julia
  julia> copy(1:4, mx.cpu())
  4 mx.NDArray{Int64,1} @ CPU0:
   1
   2
   3
   4

  julia> copy(1.:4, mx.cpu())
  4 mx.NDArray{Float64,1} @ CPU0:
   1.0
   2.0
   3.0
   4.0
  ```

* `copy!(NDArray, AbstractArray)` is implemented now. (#TBD)

  ```julia
  julia> x = mx.zeros(3)
  3-element mx.NDArray{Float32,1} @ CPU0:
   0.0
   0.0
   0.0

  julia> copy!(x, 3:5)
  3-element mx.NDArray{Float32,1} @ CPU0:
   3.0
   4.0
   5.0
  ```

* `Base.ones(x::NDArray)` for creating an one-ed `NDArray`. (#TBD)

* `Base.zeros(x::NDArray)` for creating a zero-ed `NDArray`. (#TBD)

* Modulo operator. (#TBD)

  ```julia
  x = NDArray(...)
  y = NDArray(...)

  x .% y
  x .% 2
  2 .% x
  ```

* Inplace modulo operator, `mod_from!` and `rmod_from!`. (#TBD)

  ```julia
  mod_from!(x, y)
  mod_from!(x, 2)
  rmod_from!(2, x)
  ```

* `cat`, `vcat`, `hcat` is implemented. (#TBD)

  E.g. `hcat`
  ```julia
  julia> x
  4 mx.NDArray{Float64,1} @ CPU0:
   1.0
   2.0
   3.0
   4.0

  julia> y
  4 mx.NDArray{Float64,1} @ CPU0:
   2.0
   4.0
   6.0
   8.0

  julia> [x y]
  4×2 mx.NDArray{Float64,2} @ CPU0:
   1.0  2.0
   2.0  4.0
   3.0  6.0
   4.0  8.0
  ```

* Transposing a column `NDArray` to a row `NDArray` is supported now. (#TBD)

  ```julia
  julia> x = NDArray(Float32[1, 2, 3, 4])
  4 mx.NDArray{Float32,1} @ CPU0:
   1.0
   2.0
   3.0
   4.0

  julia> x'
  1×4 mx.NDArray{Float32,2} @ CPU0:
   1.0  2.0  3.0  4.0
  ```

* Matrix/tensor multiplication is supported now. (#TBD)

  ```julia
  julia> x
  2×3 mx.NDArray{Float32,2} @ CPU0:
   1.0  2.0  3.0
   4.0  5.0  6.0

  julia> y
  3 mx.NDArray{Float32,1} @ CPU0:
   -1.0
   -2.0
   -3.0

  julia> x * y
  2 mx.NDArray{Float32,1} @ CPU0:
   -14.0
   -32.0
  ```

## API Changes

### `NDArray`

* Broadcasting along dimension supported on following operators,
  and the original `mx.broadcast_*` APIs are deprecated
  (#401) (#402) (#403):

    * `+`
    * `-`
    * `*`
    * `/`
    * `%`
    * `^`
    * `==`
    * `!=`
    * `>`
    * `>=`
    * `<`
    * `<=`
    * `max`
    * `min`

    ```julia
    julia> x = NDArray([1 2 3;
                        4 5 6])
    2×3 mx.NDArray{Int64,2} @ CPU0:
     1  2  3
     4  5  6

    julia> y = NDArray([1;
                        10])
    2-element mx.NDArray{Int64,1} @ CPU0:
      1
     10

    julia> x .+ y
    2×3 mx.NDArray{Int64,2} @ CPU0:
      2   3   4
     14  15  16
    ```

* Please use dot-call on following trigonometric functions.
  Also, the `arc*` has been renamed to keep consistent with `Base`.
  (#TBD)

    * `sin.(x)`
    * `cos.(x)`
    * `tan.(x)`
    * `arcsin(x)` -> `asin.(x)`
    * `arccos(x)` -> `acos.(x)`
    * `arctan(x)` -> `atan.(x)`

* Please use dot-call on following hyperbolic functions.
  Also, the `arc*` has been renamed to keep consistent with `Base`.
  (#TBD)

    * `sinh.(x)`
    * `cosh.(x)`
    * `tanh.(x)`
    * `arcsinh(x)` -> `asinh.(x)`
    * `arccosh(x)` -> `acosh.(x)`
    * `arctanh(x)` -> `atanh.(x)`

* Please use dot-call on following activation functions.
  And the `dim` of `softmax` and `log_softmax` has been fixed
  as Julia column-based style.
  (#TBD)

    * `σ.(x)`
    * `relu.(x)`
    * `softmax.(x, [dim = ndims(x)])`
    * `log_softmax.(x, [dim = ndims(x)])`

* `rand`, `rand!`, `randn`, `randn!` is more Base-like now (#TBD).

  ```julia
  julia> mx.rand(2, 3)
  2×3 mx.NDArray{Float32,2} @ CPU0:
   0.631961  0.324175  0.0762663
   0.285366  0.395292  0.074995

  julia> mx.rand(2, 3; low = 1, high = 10)
  2×3 mx.NDArray{Float32,2} @ CPU0:
   7.83884  7.85793  7.64791
   7.68646  8.56082  8.42189
  ```

  ```julia
  julia> mx.randn(2, 3)
  2×3 mx.NDArray{Float32,2} @ CPU0:
   0.962853  0.424535  -0.320123
   0.478113  1.72886    1.72287

  julia> mx.randn(2, 3, μ = 100)
  2×3 mx.NDArray{Float32,2} @ CPU0:
   99.5635  100.483   99.888
   99.9889  100.533  100.072
  ```

* Signature of `clip` changed, it doesn't require any keyword argument now.
  (#TBD)

  Before: `clip(x, a_min = -4, a_max = 4)`
  After: `clip(x, -4, 4)`

### Optimizer

We overhauled the optimizer APIs, introducing breaking changes.
There are tons of renaming, and we try to increase the flexibility.
Making it decouples from some high-level, so user can use it without
understand some detail implementations of `fit!`.

See #396.

* All the keyword argument of optimizers have been renamed.
  Now we have more elegant keyword arguments than Python's,
  thanks to well Unicode support on Julia's REPL and editor plugin.
  *These are breaking changes, no deprecation warning.*

    | old                       | new       | comment                        |
    |---------------------------|-----------|--------------------------------|
    | `opts.lr`                 | `η`       | type `\eta<tab>` in REPL       |
    | `opts.momentum`           | `μ`       | type `\mu<tab>` in REPL        |
    | `opts.grad_clip`          | `clip`    | type `\nabla<tab>c` in REPL    |
    | `opts.weight_decay`       | `λ`       | type `\lambda<tab>` in REPL    |
    | `opts.lr_schedular`       | `η_sched` | type `\eta<tab>_sched` in REPL |
    | `opts.momentum_schedular` | `μ_sched` | type `\mu<tab>_sched` in REPL  |

  For instance, one accessed the learning via `SGD().opts.lr`,
  but now, it's `SGD().η`.

* New keyword argument `scale` for gradient rescaling.

  Docstring:
  ```
  If != 0, multiply the gradient with `∇r` before updating.
  Often choose to be `1.0 / batch_size`.
  If leave it default, high-level API like `fit!` will set it to
  `1.0 / batch_size`, since `fit!` knows the `batch_size`.
  ```

* Keyword arguments of `NadamScheduler` has been renamed.
  *This is a breaking change, no deprecation warning.*

    * Before

      ```julia
      NadamScheduler(; mu0 = 0.99, delta = 0.004, gamma = 0.5, alpha = 0.96)
      ```

    * After

      ```julia
      NadamScheduler(; μ = 0.99, δ = 0.004, γ = 0.5, α = 0.96)
      ```

* The attribute `optimizer.state` is removed.
  `OptimizationState` is only used by high-level abstraction, like `fit!`.

* `LearningRate` scheduler API changes:

    * `get_learning_rate` is removed.
      Please use `Base.get` to get learning rate.

      ```julia
      julia> sched = mx.LearningRate.Exp(.1)
      MXNet.mx.LearningRate.Exp(0.1, 0.9, 0)

      julia> get(sched)
      0.1

      julia> update!(sched);

      julia> get(sched)
      0.09000000000000001
      ```

    * `update!` to bump counter of `Scheduler.t`
      ```julia
      julia> sched.t
      1

      julia> update!(sched);

      julia> sched.t
      2

      julia> update!(sched);

      julia> sched.t
      3
      ```

* `Momentum` module API changes:

    * `get_momentum_scheduler` is removed. Please use `Base.get` instead.

      ```julia
      julia> get(mx.Momentum.Fixed(.9))
      0.9
      ```

----

# v0.3.0 (2017.11.16)

* Update `libmxnet` to
    * On Windows: v0.12.0.
    (See https://github.com/apache/incubator-mxnet/releases/tag/0.12.0)

    * On Linux/macOS: v0.12.1.
    (See https://github.com/apache/incubator-mxnet/releases/tag/0.12.1)

* Drop 0.5 support. ([#300][300])

## New API

### `SymbolicNode`

* Debugging print support. ([#276][276])

### `NDArray`

* `deepcopy` for `NDArray` ([#273][273])

* `scalar ./ NDArray` is available now. ([#292][292])

* `fill` and `fill!` for `NDArray`. ([#297][297], [#311][311])

  An API correspond to Python's `mx.nd.full()`

    * `fill(x, dims, ctx=cpu())`
    * `fill(x, dims...)`
    * `fill!(arr::NDArray, x)`

* Matrix (2D `NDArray`) multiplication is available now. ([#300][300])

    ```julia
    julia> x
    1x2 mx.NDArray{Float64} @ CPU0:
     1.0  2.0

    julia> x' * x
    2x2 mx.NDArray{Float64} @ CPU0:
     1.0  2.0
     2.0  4.0
    ```

* `NDArray` `getindex`/`setindex!` linear indexing support and `first` for
  extracting scalar value. ([#294][294])

  ```julia
  julia> x = mx.zeros(2, 5)

  julia> x[5] = 42  # do synchronization and set the value
  ```

  ```julia
  julia> y = x[5]  # actually, getindex won't do synchronization, but REPL's showing did it for you
  1 mx.NDArray{Float32} @ CPU0:
   42.0

  julia> first(y)  # do sync and get the value
  42.0f0

  julia> y[]  # this is available, also
  42.0f0
  ```
* Elementwise power of `NDArray`. ([#293][293])

    * `x.^2`
    * `2.^x`
    * `x.^y`
    * where `x` and `y` are `NDArray`s.

* Elementwise power of irrational and `NDArray`. ([#310][310])

    * `e.^x`
    * `x.^e`
    * `π.^x`

## API Changes

### `SymbolicNode`

* `reshape` of `SymbolicNode` shares the same interface with Base
  and additional keyword argument. ([#279][279])

    * `reshape(SymbolicNode, dim; reverse=false, name)`
    * `reshape(SymbolicNode, dim...; reverse=false, name)`
    * `Reshape` is deprecated.

* `mx.forward(x)` will return `x.outputs` now. ([#312][312])

### `NDArray`

* `reshape` of `NDArray` shares the same interface with Base. ([#272][272])

    * `reshape(NDArray, dim; reverse=false)`
    * `reshape(NDArray, dim...; reverse=false)`
    * `Reshape` is deprecated.

* `srand!` deprecated, please use `srand`. ([#282][282])

* `mean` and `sum` of `NDArray` share the same interface with Base
  and fix the `axis` indexing. ([#303][303])

    * This is a breaking change; no deprecated warning.
    * Before: `mean(arr, axis=0)`
    * After: `mean(arr, 1)`

* `max` and `min` of `NDArray` renamed to `maximum` and `minimum` and share the
  same interface with Base. The `axis` indexing is fixed, also. ([#303][303])

    * This is a breaking change; no deprecated warning.
    * Before: `mx.max(arr, axis=0)` or `mx.max_axis(arr, axis=0)`
    * After: `maximum(arr, 1)`

* `mx.transpose` for high dimension `NDArray` has been renamed to `permutedims`
  and shares the same interface with Base. ([#303][303])

    * This is a breaking changes; no deprecated warning.
    * Before: `mx.transpose(A, axis=[2, 1, 3])`
    * After: `permutedims(A, [2, 1, 3])`

* `prod` of `NDArray` shares the same interface with Base and fix the `axis`
  indexing. ([#303][303])

    * This is a breaking change; no deprecated warning.
    * Before: `prod(arr, axis=-1)`
    * After: `prod(arr, 1)`

## Bugfix

* Broadcasting operation on same variable is back. ([#300][300], [#314][314])
  ```julia
  x = mx.NDArray(...)
  x .* x
  ```

  ```julia
  y = mx.Variable(:y)
  y .* y
  ```

[272]: https://github.com/dmlc/MXNet.jl/pull/272
[273]: https://github.com/dmlc/MXNet.jl/pull/273
[276]: https://github.com/dmlc/MXNet.jl/pull/276
[279]: https://github.com/dmlc/MXNet.jl/pull/279
[282]: https://github.com/dmlc/MXNet.jl/pull/282
[292]: https://github.com/dmlc/MXNet.jl/pull/292
[293]: https://github.com/dmlc/MXNet.jl/pull/293
[294]: https://github.com/dmlc/MXNet.jl/pull/294
[297]: https://github.com/dmlc/MXNet.jl/pull/297
[300]: https://github.com/dmlc/MXNet.jl/pull/300
[303]: https://github.com/dmlc/MXNet.jl/pull/303
[310]: https://github.com/dmlc/MXNet.jl/pull/310
[311]: https://github.com/dmlc/MXNet.jl/pull/311
[312]: https://github.com/dmlc/MXNet.jl/pull/312
[314]: https://github.com/dmlc/MXNet.jl/pull/314

# v0.2.2 (2017.05.14)
* Updated supported version of MXNet to 0.9.4.
* Improved build-system with support for auto-detecting GPU support.
* Several updates to Metrics.
* CI for Windows.
* Verbosity option for `predict` (@rdeits)

# v0.2.1 (2017.01.29)
* Bugfix release for Windows

# v0.2.0 (2017.01.26)
* Drop support for Julia v0.4.
* Added support for NVVM.
* Updated supported version of MXNet to 0.9.2
* New optimizers (@Arkoniak).

# v0.1.0 (2016.09.08)

* Track specific libmxnet version for each release.
* Migrated documentation system to `Documenter.jl` (@vchuravy)
* Simplified building by using Julia's OpenBlas (@staticfloat)
* Freezing parameters (@vchuravy)
* Support `DType` for `NDArray` (@vchuravy)

# v0.0.8 (2016.02.08)

* Fix compatability with Julia v0.5.
* Fix seg-faults introduced by upstream API changes.

# v0.0.7 (2015.12.14)

* Fix compatability with Julia v0.4.2 (@BigEpsilon)
* Metrics in epoch callbacks (@kasiabozek)

# v0.0.6 (2015.12.02)

* Variants of Xaiver initializers (@vchuravy)
* More arithmetic operators on symbolic nodes
* Basic interface for symbolic node attributes (@vchuravy)

# v0.0.5 (2015.11.14)

* char-lstm example.
* Network visualization via GraphViz.
* NN-factory for common models.
* Convenient `@nd_as_jl` macro to work with `NDArray` as Julia Arrays.
* Refactoring: `Symbol` -> `SymbolicNode`.
* More evaluation metrics (@vchuravy, @Andy-P)

# v0.0.4 (2015.11.09)

* ADAM optimizer (@cbecker)
* Improved data provider API.
* More documentation.
* Fix a bug in array data iterator (@vchuravy)

# v0.0.3 (2015.10.27)

* Model prediction API.
* Model checkpoint loading and saving.
* IJulia Notebook example of using pre-trained imagenet model as classifier.
* Symbol saving and loading.
* NDArray saving and loading.
* Optimizer gradient clipping.
* Model training callback APIs, default checkpoint and speedometer callbacks.
* Julia Array / NDArray data iterator.
* Sphinx documentation system and documents for dynamically imported libmxnet APIs.

# v0.0.2 (2015.10.23)

* Fix a bug in build script that causes Julia REPL to exit.

# v0.0.1 (2015.10.23)

Initial release.

* Basic libmxnet API.
* Basic documentation, overview and MNIST tutorial.
* Working MNIST and cifar-10 examples, with multi-GPU training.
* Automatic building of libmxnet with BinDeps.jl.


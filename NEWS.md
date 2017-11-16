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


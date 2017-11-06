# v0.3.0 (TBD)

## New API

* `deepcopy` for NDArray (#273)
* `scalar ./ NDArray` is available now. (#292)
* `fill` and `fill!` for NDArray (#TBD)
  An API correspond to Python's `mx.nd.full()`

    * `fill(x, dims, ctx=cpu())`
    * `fill(x, dims...)`
    * `fill!(x, arr::NDArray)`

## API Changes

* `reshape` of NDArray shares the same interface with Base (#272).
    * `reshape(NDArray, dim; reverse=false)`
    * `reshape(NDArray, dim...; reverse=false)`
    * `Reshape` deprecated.

* `reshape` of SymbolicNode shares the same interface with Base
  and additional keyword argument (#279).

    * `reshape(SymbolicNode, dim; reverse=false, name)`
    * `reshape(SymbolicNode, dim...; reverse=false, name)`
    * `Reshape` deprecated.

* `srand!` deprecated, please use `srand` (#282)

* `mean` and `sum` of NDArray share the same interface with Base
  and fix the `axis` indexing (#TBD).

    * This is a breaking change; no deprecated warning.
    * Before: `mean(arr, axis=0)`
    * After: `mean(arr, 1)`

* `max` and `min` of NDArray renamed to `maximum` and `minimum` and share the
  same interface with Base. The `axis` indexing is fixed, also. (#TBD)

    * This is a breaking change; no deprecated warning.
    * Before: `mx.max(arr, axis=0)` or `mx.max_axis(arr, axis=0)`
    * After: `maximum(arr, 1)`

* `mx.transpose` for high dimension NDArray has been renamed to `permutedims`
  and shares the same interface with Base. (#TBD)

    * This is a breaking changes; no deprecated warning.
    * Before: `mx.transpose(A, axis=[2, 1, 3])`
    * After: `permutedims(A, [2, 1, 3])`

* `prod` of `NDArray` shares the same interface with Base and fix
  the `axis` indexing. (#TBD).

    * This is a breaking change; no deprecated warning.
    * Before: `prod(arr, axis=-1)`
    * After: `prod(arr, 1)`

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


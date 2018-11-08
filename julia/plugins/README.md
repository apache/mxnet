# Plugins of MXNet.jl

This directory contains *plugins* of MXNet.jl. A plugin is typically a component that could be part of MXNet.jl, but excluded from the `mx` namespace. The plugins are included here primarily for two reasons:

* To minimize the dependency of MXNet.jl on other optional packages.
* To serve as examples on how to extend some components of MXNet.jl.

The most straightforward way to use a plugin is to `include` the code. For example

```julia
include(joinpath(Pkg.dir("MXNet"), "plugins", "io", "svmlight.jl"))

provider = SVMLightProvider("/path/to/dataset", 100)
```

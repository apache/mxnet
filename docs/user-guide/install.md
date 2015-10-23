# Automatic Installation

To install MXNet.jl, simply type
```jl
Pkg.add("MXNet")
```
in the Julia REPL. MXNet.jl is built on top of [libmxnet](https://github.com/dmlc/mxnet). Upon installation, Julia will try to automatically download and build libmxnet.

The libmxnet source is downloaded to `Pkg.dir("MXNet")/deps/src/mxnet`. The automatic build is using default configurations, with OpenCV, CUDA disabled.
If the compilation failed due to unresolved dependency, or if you want to customize the build, it is recommended to compile and install libmxnet manually. Please see [below](#manual-compilation) for more details.

To use the latest git version of MXNet.jl, use the following command instead
```jl
Pkg.checkout("MXNet")
```

# Manual Compilation

It is possible to compile libmxnet separately and point MXNet.jl to a the existing library in case automatic compilation fails due to unresolved dependencies in an un-standard environment; Or when one want to work with a seperate, maybe customized libmxnet.

To build libmxnet, please refer to [the installation guide of libmxnet](http://mxnet.readthedocs.org/en/latest/build.html). After successfully installing libmxnet, set the `MXNET_HOME` environment variable to the location of libmxnet. In other words, the compiled `libmxnet.so` should be found in `$MXNET_HOME/lib`.

When the `MXNET_HOME` environment variable is detected and the corresponding `libmxnet.so` could be loaded successfully, MXNet.jl will skip automatic building during installation and use the specified libmxnet instead.

Basically, MXNet.jl will search `libmxnet.so` or `libmxnet.dll` in the following paths (and in that order):

* `$MXNET_HOME/lib`: customized libmxnet builds
* `Pkg.dir("MXNet")/deps/usr/lib`: automatic builds
* Any system wide library search path

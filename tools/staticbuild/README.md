# MXNet Static Build

This folder contains the core script used to build the static library. This README provides information on how to use the scripts in this folder. Please be aware, all of the scripts are designed to be run under the root folder.

## `build.sh`
This script is a wrapper around `build_lib.sh. It simplifies the things by automatically identifing the system version, number of cores, and all environment variable settings. Here are examples you can run with this script:

```
tools/staticbuild/build.sh cu92 maven
```
This would build the mxnet package based on CUDA9.2 and Maven (Scala) build setttings.
```
tools/staticbuild/build.sh mkl pip
```
This would build the mxnet package based on MKLDNN and and pypi configuration settings.

As the result, users would have a complete static dependencies in `/staticdeps` in the root folder as well as a static-linked `libmxnet.so` file lives in `lib`. You can build your language binding by using the `libmxnet.so`.

## `build_lib.sh`
This script clones the most up-to-date master and builds the MXNet backend with a static library. In order to run the static library, you must set the the following environment variables:

- `DEPS_PATH` Path to your static dependencies
- `STATIC_BUILD_TARGET` Either `pip` or `maven` as your publish platform
- `PLATFORM` linux, darwin
- `VARIANT` cpu, cu*, cu*mkl, mkl

It is not recommended to run this file alone since there are a bunch of variables need to be set.

After running this script, you would have everything you need ready in the `/lib` folder.

## `build_wheel.sh`
This script builds the python package. It also runs a sanity test.
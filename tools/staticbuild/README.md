# MXNet Static build

This folder contains the core script used to build the static library. This README would bring you the information and usages of the script in here. Please be aware, all of the scripts are designed to be run under the root folder.

## `build.sh`
This script is a wrapper around `build_lib.sh` aimed to simplify the usage of it. It would automatically identify the system version, number of cores and all environment variable settings. Here are the examples you can run this script:

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
This script would clone the most up-to-date master and build the MXNet backend with static library. In order to run that, you should have prepare the the following environment variable:

- `DEPS_PATH` Path to your static dependencies
- `STATIC_BUILD_TARGET` Either `pip` or `maven` as your publish platform
- `PLATFORM` linux, darwin
- `VARIANT` cpu, cu*, cu*mkl, mkl

It is not recommended to run this file alone since there are a bunch of variables need to be set.

After running this script, you would have everything you need ready in the `/lib` folder.

## `build_wheel.sh`
This script is used to build the python package as well as running a sanity test
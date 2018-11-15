# Overview

This folder contains scripts for building the dependencies from source. The static libraries from
the build artifacts can be used to create self-contained shared object for mxnet through static
linking.

# Settings

The scripts use the following environment variables for setting behavior:

`DEPS_PATH`: the location in which the libraries are downloaded, built, and installed.
`PLATFORM`: name of the OS in lower case such as 'linux' and 'darwin'. For example, it can be set using `export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')`.

It also expects the following build tools in path: make, cmake

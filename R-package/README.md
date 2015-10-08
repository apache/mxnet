MXNet R-Package
===============
This is an on-going effort to support mxnet in R, stay tuned.

Bleeding edge Installation
- First build ```../lib/libmxnet.so``` by following [Build Instruction](../doc/build.md)
- Type ```R CMD INSTALL R-package``` in the root folder.

Contributing Style Guide
------------------------
- Most C++ of R package heavily relies on [Rcpp](https://github.com/RcppCore/Rcpp).
- We follow Google's C++ Style guide on C++ code.
  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.
- You can check the style of the code by typing the following command at root folder.
```bash
make rcpplint
```
- When needed, you can disable the linter warning of certain line with ```// NOLINT(*)``` comments.

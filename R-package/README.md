MXNet R-Package
===============
This is an on-going effort to support mxnet in R, stay tuned.

Bleeding edge Installation
- First build ```../lib/libmxnet.so``` by following [Build Instruction](../doc/build.md)
- Type ```R CMD INSTALL R-package``` in the root folder.

Contributor Guide for R
-----------------------
### Code Style
- Most C++ of R package heavily relies on [Rcpp](https://github.com/RcppCore/Rcpp).
- We follow Google's C++ Style guide on C++ code.
  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.
- You can check the style of the code by typing the following command at root folder.
```bash
make rcpplint
```
- When needed, you can disable the linter warning of certain line with ```// NOLINT(*)``` comments.

### Auto Generated API
- Many mxnet API are exposed from Rcpp side in a dynamic way.
- The [mx_generated.R](R/mx_generated.R) is auto generated API and documents for these functions.
- You can remake the file by typing the following command at root folder
```bash
make rcppexport
```
- This only need to be done periodically when there is update on dynamic functions.

### Document
- The document is generated using roxygen2
- You can type the following command to remake the documents at root folder.
```bash
make roxygen
```

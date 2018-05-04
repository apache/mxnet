<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnetR.png width=155/> Deep Learning for R
==========================

You have found MXNet R Package! The MXNet R packages brings flexible and efficient GPU
computing and state-of-the-art deep learning to R.

- It enables you to write seamless tensor/matrix computation with multiple GPUs in R.
- It also enables you to construct and customize state-of-the-art deep learning models in R,
  and apply them to tasks such as image classification and data science challenges.

Sounds exciting? This page contains links to all the related documentation of the R package.


Installation
------------

We provide pre-built binary packages for Windows/OSX users.
You can install the CPU package directly from the R console:

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

To use the GPU version or to use it on Linux, please follow [Installation Guide](http://mxnet.io/install/index.html)

License
-------
MXNet R-package is licensed under [Apache-2.0](./LICENSE) license.

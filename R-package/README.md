<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnetR.png width=155/> Deep Learning for R
==========================
[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.readthedocs.org/en/latest/packages/r/index.html)

You have found MXNet R Package! The MXNet R packages brings flexible and efficient GPU
computing and state-of-art deep learning to R.

- It enables you to write seamless tensor/matrix computation with multiple GPUs in R.
- It also enables you to construct and customize the state-of-art deep learning models in R,
  and apply them to tasks such as image classification and data science challenges.

Sounds exciting? This page contains links to all the related documents on R package.

Resources
---------
* [MXNet R Package Document](http://mxnet.readthedocs.org/en/latest/packages/r/index.html)
  - Check this out for detailed documents, examples, installation guides.

Installation
------------

For Windows/Mac users, we provide pre-built binary package using CPU.
You can install weekly updated package directly in R console:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

To use GPU version or use it on Linux, please follow [Installation Guide](http://mxnet.readthedocs.org/en/latest/how_to/build.html)

License
-------
MXNet R-package is licensed under [BSD](./LICENSE) license.

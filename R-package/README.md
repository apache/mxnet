<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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

To use the GPU version or to use it on Linux, please follow [Installation Guide](https://mxnet.io/install/index.html)

License
-------
MXNet R-package is licensed under [Apache-2.0](./LICENSE) license.

<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Captcha

This is the clojure version of [captcha recognition](https://github.com/xlvector/learning-dl/tree/master/mxnet/ocr)
example by xlvector and mirrors the R captcha example. It can be used as an
example of multi-label training. For the following captcha example, we consider it as an
image with 4 labels and train a CNN over the data set.

![captcha example](captcha_example.png)

## Installation

Before you run this example, make sure that you have the clojure package
installed. In the main clojure package directory, do `lein install`.
Then you can run `lein install` in this directory.

## Usage

### Training

First the OCR model needs to be trained based on [labeled data](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/captcha_example.zip).
The training can be started using the following:
```
$ lein train [:cpu|:gpu] [num-devices]
```
This downloads the training/evaluation data using the `get_data.sh` script
before starting training.

It is possible that you will encounter some out-of-memory issues while training using :gpu on Ubuntu
linux (18.04). However, the command `lein train` (training on one CPU) may resolve the issue.

The training runs for 10 iterations by default and saves the model with the
prefix `ocr-`. The model achieved an exact match accuracy of ~0.954 and
~0.628 on training and validation data respectively.

### Inference

Once the model has been saved, it can be used for prediction. This can be done
by running:
```
$ lein infer
INFO  MXNetJVM: Try loading mxnet-scala from native path.
INFO  MXNetJVM: Try loading mxnet-scala-linux-x86_64-gpu from native path.
INFO  MXNetJVM: Try loading mxnet-scala-linux-x86_64-cpu from native path.
WARN  MXNetJVM: MXNet Scala native library not found in path. Copying native library from the archive. Consider installing the library somewhere in the path (for Windows: PATH, for Linux: LD_LIBRARY_PATH), or specifying by Java cmd option -Djava.library.path=[lib path].
WARN  org.apache.mxnet.DataDesc: Found Undefined Layout, will use default index 0 for batch axis
INFO  org.apache.mxnet.infer.Predictor: Latency increased due to batchSize mismatch 8 vs 1
WARN  org.apache.mxnet.DataDesc: Found Undefined Layout, will use default index 0 for batch axis
WARN  org.apache.mxnet.DataDesc: Found Undefined Layout, will use default index 0 for batch axis
CAPTCHA output: 6643
INFO  org.apache.mxnet.util.NativeLibraryLoader: Deleting /tmp/mxnet6045308279291774865/libmxnet.so
INFO  org.apache.mxnet.util.NativeLibraryLoader: Deleting /tmp/mxnet6045308279291774865/mxnet-scala
INFO  org.apache.mxnet.util.NativeLibraryLoader: Deleting /tmp/mxnet6045308279291774865
```
The model runs on `captcha_example.png` by default.

It can be run on other generated captcha images as well. The script
`gen_captcha.py` generates random captcha images for length 4.
Before running the python script, you will need to install the [captcha](https://pypi.org/project/captcha/)
library using `pip3 install --user captcha`. The captcha images are generated
in the `images/` folder and we can run the prediction using
`lein infer images/7534.png`.

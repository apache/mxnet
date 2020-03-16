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

Prerequisites
-------------
This package supports Linux and Windows platforms. You may also want to check:
- [mxnet-cu102](https://pypi.python.org/pypi/mxnet-cu101/) with CUDA-10.2 support.
- [mxnet-cu92](https://pypi.python.org/pypi/mxnet-cu92/) with CUDA-9.2 support.
- [mxnet-cu90](https://pypi.python.org/pypi/mxnet-cu90/) with CUDA-9.0 support.
- [mxnet-cu80](https://pypi.python.org/pypi/mxnet-cu80/) with CUDA-8.0 support.
- [mxnet-cu75](https://pypi.python.org/pypi/mxnet-cu75/) with CUDA-7.5 support.
- [mxnet](https://pypi.python.org/pypi/mxnet/).

To download CUDA, check [CUDA download](https://developer.nvidia.com/cuda-downloads). For more instructions, check [CUDA Toolkit online documentation](http://docs.nvidia.com/cuda/index.html).

To install for other platforms (e.g. Windows, Raspberry Pi/ARM) or other versions, check [Installing MXNet](https://mxnet.apache.org/versions/master/install/index.html) for instructions on building from source.

Installation
------------
To install:
```bash
pip install mxnet-cu100
```

Nightly Builds
--------------
To install the latest nightly build, use:
```bash
pip install --pre mxnet-cu100 -f https://dist.mxnet.io/python
```

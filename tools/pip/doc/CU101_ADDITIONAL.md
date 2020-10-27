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
- [mxnet-cu110](https://pypi.python.org/pypi/mxnet-cu110/) with CUDA-11.0 support.
- [mxnet-cu102](https://pypi.python.org/pypi/mxnet-cu102/) with CUDA-10.2 support.
- [mxnet-cu100](https://pypi.python.org/pypi/mxnet-cu101/) with CUDA-10.0 support.
- [mxnet-cu92](https://pypi.python.org/pypi/mxnet-cu92/) with CUDA-9.2 support.
- [mxnet](https://pypi.python.org/pypi/mxnet/).

To download CUDA, check [CUDA download](https://developer.nvidia.com/cuda-downloads). For more instructions, check [CUDA Toolkit online documentation](http://docs.nvidia.com/cuda/index.html).

To use this package on Linux you need the `libquadmath.so.0` shared library. On
Debian based systems, including Ubuntu, run `sudo apt install libquadmath0` to
install the shared library. On RHEL based systems, including CentOS, run `sudo
yum install libquadmath` to install the shared library. As `libquadmath.so.0` is
a GPL library and MXNet part of the Apache Software Foundation, MXNet must not
redistribute `libquadmath.so.0` as part of the Pypi package and users must
manually install it.

To install for other platforms (e.g. Windows, Raspberry Pi/ARM) or other versions, check [Installing MXNet](https://mxnet.incubator.apache.org/versions/master/install/index.html) for instructions on building from source.

Installation
------------
To install:
```bash
pip install mxnet-cu101
```

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

MXNet Python Package
====================
This directory and nested files contain MXNet Python package and language binding.

## Installation
To install MXNet Python package, visit MXNet [Install Instruction](http://mxnet.incubator.apache.org/install/index.html)


## Running the unit tests

For running unit tests, you will need the [nose PyPi package](https://pypi.python.org/pypi/nose). To install:
```bash
pip install --upgrade nose
```

Once ```nose``` is installed, run the following from MXNet root directory (please make sure the installation path of ```nosetests``` is included in your ```$PATH``` environment variable):
```
nosetests tests/python/unittest
nosetests tests/python/train

```

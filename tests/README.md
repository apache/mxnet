<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# Testing MXNET

## Running CPP Tests

1. Install [cmake](https://cmake.org/install/)
1. Create a build directory in the root of the mxnet project
    ```
    mkdir build
    cd build
    ```
1. Generate your Makefile and build along with the tests with cmake (specify appropraite flags)
    ```
    cmake -DUSE_CUDNN=ON -DUSE_CUDA=ON -DUSE_ONEDNN=ON -DUSE_BLAS=Open -DCMAKE_BUILD_TYPE=Debug .. && make
    ```
1.  Run tests
    ```
    ctest --verbose
    ```

1. The following will run all the tests the in `cpp` directory. To run just your test file replace the following in your `tests/CMakeLists.txt`
    ```
    file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/*.cc" "cpp/*.h")
    ```
    with
    ```
    file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/test_main.cc" "cpp/{RELATIVE_PATH_TO_TEST_FILE}")
    ```

### Building with Ninja

Ninja is a build tool (like make) that prioritizes building speed. If you will be building frequently, we recommend you use ninja

1. Download Ninja via package manager or directly from [source](https://github.com/ninja-build/ninja)
    ```
    apt-get install ninja-build
    ```
1. When running cmake, add the `-GNinja` flag to specify cmake to generate a Ninja build file
    ```
    cmake -DUSE_CUDNN=ON -DUSE_CUDA=ON -DUSE_ONEDNN=ON -DUSE_BLAS=Open -GNinja -DCMAKE_BUILD_TYPE=Debug ..
    ```
1. Run the ninja build file with
    ```
    ninja
    ```
    
## Runing Python Tests Within Docker

To run tests inside docker, you first need to install `docker` and `docker-compose` on your machine.

On Ubuntu you may install them via `sudo apt-get install docker.io docker-compose python3-docker`
and set them up via `sudo usermod $(whoami) -G docker -a`.

Then, to run tests inside docker run the following command

```
ci/build.py --platform {PLATFORM} /work/runtime_functions.sh {RUNTIME_FUNCTION}
```

An example for running python tests on Ubuntu with a CPU would be

```
ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_python3_cpu
```

See [Continuous Integration](../ci/README.md)

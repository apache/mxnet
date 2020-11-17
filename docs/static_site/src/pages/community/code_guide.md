---
layout: page
title: Code Guide and Tips
subtitle: Tips in MXNet codebase for reviewers and contributors.
action: Contribute
action_url: /community/index
permalink: /community/code_guide
---
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

Code Guide and Tips
===================

This is a document used to record tips in MXNet codebase for reviewers and
contributors. Most of them are summarized through lessons during the
contributing and process.

C++ Code Styles
---------------

-   Use the [Google C/C++ style](https://google.github.io/styleguide/cppguide.html).
-   The public facing functions are documented in [doxygen](https://www.doxygen.nl/manual/docblocks.html) format.
-   Favor concrete type declaration over `auto` as long as it is short.
-   Favor passing by const reference (e.g. `const Expr&`) over passing
    by value. Except when the function consumes the value by copy
    constructor or move, pass by value is better than pass by const
    reference in such cases.
-   Favor `const` member function when possible.
-   Use [RAII](https://en.cppreference.com/w/cpp/language/raii) to manage resources, including smart pointers like shared_ptr and unique_ptr as well as allocating in constructors and deallocating in destructors. Avoid explicit calls to new and delete when possible. Use make_shared and make_unique instead.

We use [`cpplint`](https://github.com/cpplint/cpplint) to enforce the code style. Because
different version of `cpplint` might change by its version, it is
recommended to use the same version of the `cpplint` as the master.
You can also use the following command via docker.

```bash
ci/build.py -R --docker-registry mxnetci --platform ubuntu_cpu --docker-build-retries 3 --shm-size 500m /work/runtime_functions.sh sanity_cpp
```

`cpplint` is also not perfect, when necessary, you can use disable
`cpplint` on certain code regions.

Python Code Styles
------------------

-   The functions and classes are documented in
    [numpydoc](https://numpydoc.readthedocs.io/en/latest/) format.
-   Check your code style using `make pylint`
-   Stick to language features as in `python 3.6` and above.

Testing
-------

Our tests are maintained in the [/tests](https://github.com/apache/incubator-mxnet/tree/master/tests) folder. We use the following testing tools:
-   For Python, we use [pytest](https://pytest.org).
    -   An example of setting up and running tests (tested on MacOS with Python 3.6):
        -   follow the [build from source](https://mxnet.apache.org/get_started/build_from_source) guide to build MXNet
        -   install python libraries
            ```
            python3 -m pip install opencv-python
            python3 -m pip install -r ci/docker/install/requirements
            ```
        -   install MXNet Python bindings:
            ```
            python3 -m pip install -e ./python
            ```
        -   run tests in a specific module
            ```
            python3 -m pytest tests/python/unittest/test_smoke.py
            ```
        -   or run a specific test in a module
            ```
            python3 -m pytest tests/python/unittest/test_smoke.py::test_18927
            ```
        -   or run all the Python unittests
            ```
            python3 -m pytest tests/python/unittest/
            ```
-   For C++, we use [gtest](https://github.com/google/googletest).

Our CI pipelines check for a wide variety of configuration on all platforms. To locate and reproduce
a test issue in PR, you can refer to the process described in [#18723](https://github.com/apache/incubator-mxnet/issues/18723)

<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>

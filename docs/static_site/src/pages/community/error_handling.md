---
layout: page
title: Error Handling Guide
subtitle: Utilize structured error types in MXNet for modern cross-language error handling.
action: Contribute
action_url: /community/index
permalink: /community/error_handling
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

Error Handling Guide
====================

MXNet contains structured error classes to indicate specific types of
error. Please raise a specific error type when possible, so that users
can write code to handle a specific error category if necessary. You can
directly raise the specific error object in python. In other languages
like c++, you simply add `<ErrorType>:` prefix to the error message(see
below).

{% include note.html content="Please refer to [/python/mxnet/error.py](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/error.py) for the list of errors." %}


Raise a Specific Error in C++
-----------------------------

You can add `<ErrorType>:` prefix to your error message to raise an
error of the corresponding type. Note that you do not have to add a new
type `mxnet.base.MXNetError` will be
raised by default when there is no error type prefix in the message.
This mechanism works for both `LOG(FATAL)` and `CHECK` macros. The
following code gives an example on how to do so.

```cpp
// Python frontend receives the following error type:
// ValueError: Check failed: x == y (0 vs. 1) : expect x and y to be equal.
CHECK_EQ(0, 1) << "ValueError: expect x and y to be equal."


// Python frontend receives the following error type:
// InternalError: cannot reach here
LOG(FATAL) << "InternalError: cannot reach here";
```

As you can see in the above example, MXNet's ffi system combines both the
python and C++'s stacktrace into a single message, and generate the
corresponding error class automatically.

How to choose an Error Type
---------------------------

You can go through the error types are listed below, try to use common
sense and also refer to the choices in the existing code. We try to keep
a reasonable amount of error types. If you feel there is a need to add a
new error type, do the following steps:

-   Send a RFC proposal with a description and usage examples in the
    current codebase.
-   Add the new error type to `mxnet.error` with clear documents.
-   Update the list in this file to include the new error type.
-   Change the code to use the new error type.

We also recommend to use less abstraction when creating the short error
messages. The code is more readable in this way, and also opens path to
craft specific error messages when necessary.

```python
def preferred():
    # Very clear about what is being raised and what is the error message.
    raise OpNotImplemented("Operator relu is not implemented in the MXNet frontend")

def _op_not_implemented(op_name):
    return OpNotImplemented("Operator {} is not implemented.").format(op_name)

def not_preferred():
    # Introduces another level of indirection.
    raise _op_not_implemented("relu")
```

If we need to introduce a wrapper function that constructs multi-line
error messages, please put wrapper in the same file so other developers
can look up the implementation easily.

Signal Handling
---------------

When not careful, some errors can occur in the form of a [signal](https://en.wikipedia.org/wiki/Signal_(IPC)),
which is handled by the OS kernel. In MXNet, you can choose to handle certain signals in the form of
a catchable exception. This can be combined with the error type selection above so that it can be
caught in the Python frontend. Currently, the following signals are handled this way:

-   `SIGFPE`: throws `FloatingPointError`
-   `SIGBUS`: throws `IOError`

To extend this to other signals, you can modify the signal handler registration in
[/src/initialize.cc](https://github.com/apache/incubator-mxnet/blob/72eff9b66ecc683c3e7f9ad2c0ba69efa8dd423b/src/initialize.cc#L347-L376).

<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>

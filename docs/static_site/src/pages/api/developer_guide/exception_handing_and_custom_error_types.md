---
layout: page_category
title:  Exception handing and custom error types
category: Developer Guide
permalink: /api/dev-guide/exception_handing_and_custom_error_types
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

# Exception handing and custom error types


Apache MXNet v1.7 has added the custom error type support and as a result `MXNetError` is inherited from `RuntimeError` so it is possible to register a custom error type in the backend and prepend its error message. Then in the frontend, one can throw the exception of the registered error type. 

For example, we want the `transpose` operator defined in the C++ backend to throw `ValueError` type in the Python frontend. Therefore, in the C++ backend we can add this check:

```
CHECK_EQ(axes_set.size(), axes.ndim()) << "ValueError: Repeated axis in transpose."
                                       << " param.axes = "
                                       << param.axes;
```

so that on the frontend, when a problematic `transpose` call is made such as:

```
from mxnet import np

dat = np.random.normal(0, 1, (3, 4, 5))
dat.transpose((0, 0, 1))
```

the following traceback will be produced:


```
ValueError                                Traceback (most recent call last)
<ipython-input-3-3ad259b4e371> in <module>
----> 1 dat.transpose((0, 0, 1))

~/mxnet-distro/mxnet-build/python/mxnet/numpy/multiarray.py in transpose(self, *axes)
   1460             elif axes[0] is None:
   1461                 axes = None
-> 1462         return _mx_np_op.transpose(self, axes=axes)
   1463
   1464     def flip(self, *args, **kwargs):
~/mxnet-distro/mxnet-build/python/mxnet/ndarray/register.py in transpose(a, axes, out, name, **kwargs)

~/mxnet-distro/mxnet-build/python/mxnet/_ctypes/ndarray.py in _imperative_invoke(handle, ndargs, keys, vals, out, is_np_op, output_is_list)
    105         c_str_array(keys),
    106         c_str_array([str(s) for s in vals]),
--> 107         ctypes.byref(out_stypes)))
    108
    109     create_ndarray_fn = _np_ndarray_cls if is_np_op else _ndarray_cls
    
~/mxnet-distro/mxnet-build/python/mxnet/base.py in check_call(ret)
    271     """
    272     if ret != 0:
--> 273         raise get_last_ffi_error()
    274
    275
ValueError: Traceback (most recent call last):
  File "src/operator/numpy/np_matrix_op.cc", line 77
  
ValueError: Check failed: axes_set.size() == axes.ndim() (2 vs. 3) : Repeated axis in transpose. param.axes = [0,0,1]
```


Note that as of writing this document, the following Python error types are supported:


* `ValueError`
* `TypeError`
* `AttributeError`
* `IndexError`
* `NotImplementedError`

Check [this](https://github.com/apache/mxnet/blob/master/python/mxnet/error.py) resource for more details
about Python supported error types that MXNet supports.

## How to register a custom error type

Here is the way to register a custom error type in Python frontend:


```
import mxnet as mx

@mx.error.register
class MyError(mx.MXNetError):
    def __init__(self, msg):
        super().__init__(msg)
```

Then in the C++ backend, you can refer to `MyError` via:

`LOG(FATAL) << "MyError: this is a custom error message"`

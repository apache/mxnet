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

# ONNX Export Support for MXNet

### Overview
[ONNX](https://onnx.ai/), or Open Neural Network Exchange, is an open source deep learning model format that acts as a framework neutral graph representation between DL frameworks or between training and inference. With the ability to export models to the ONNX format, MXNet users can enjoy faster inference and a wider range of deployment device choices, including edge and mobile devices where MXNet installation may be constrained. Popular hardware-accelerated and/or cross-platform ONNX runtime frameworks include Nvidia [TensorRT](https://github.com/onnx/onnx-tensorrt), Microsoft [ONNXRuntime](https://github.com/microsoft/onnxruntime), Apple [CoreML](https://github.com/onnx/onnx-coreml), etc.

### ONNX Versions Supported
ONNX 1.7 & 1.8

### Installation
From MXNet 1.9 release and on, the ONNX export module has become an offical, built-in feature in MXNet. You can access the module at `mxnet.onnx`.

If you are a user of earlier MXNet versions and do not want to upgrade MXNet, you can still enjoy the latest ONNX support by pulling the MXNet source code and building the wheel for only the mx2onnx module. Just do `cd python/mxnet/onnx` and then build the wheel with `python3 -m build`. You should be able to find the wheel under `python/mxnet/onnx/dist/mx2onnx-0.0.0-py3-none-any.whl` and install it with `pip install mx2onnx-0.0.0-py3-none-any.whl`. You can then access the module with `import mx2onnx`. The `mx2onnx` namespace is equivalent to `mxnet.onnx`.

### APIs
The main API is `export_model`, which, as the name suggests, exports an MXNet model to the ONNX format.

```python
mxnet.onnx.export_model(sym, params, in_shapes=None, in_types=np.float32,
                 onnx_file_path='model.onnx', verbose=False, dynamic=False,
                 dynamic_input_shapes=None, run_shape_inference=False, input_type=None,
                 input_shape=None)
```

Parameters:

    sym : str or symbol object
        Path to the MXNet json file or Symbol object
    params : str or dict or list of dict
        str - Path to the MXNet params file
        dict - MXNet params dictionary (Including both arg_params and aux_params)
        list - list of length 2 that contains MXNet arg_params and aux_params
    in_shapes : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    in_types : data type or list of data types
        Input data type e.g. np.float32, or [np.float32, np.int32]
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If True will print logs of the model conversion
    dynamic: Boolean
        If True will allow for dynamic input shapes to the model
    dynamic_input_shapes: list of tuple
        Specifies the dynamic input_shapes. If None then all dimensions are set to None
    run_shape_inference : Boolean
        If True will run shape inference on the model
    input_type : data type or list of data types
        This is the old name of in_types. We keep this parameter name for backward compatibility
    input_shape : List of tuple
        This is the old name of in_shapes. We keep this parameter name for backward compatibility
    large_model : Boolean
        Whether to export a model that is larger than 2 GB. If true will save param tensors in separate
        files along with .onnx model file. This feature is supported since onnx 1.8.0

Returns:

    onnx_file_path : str
        Onnx file path

#### Model with Multiple Input
When the model has multiple inputs, all the input shapes and dtypes must be provided with `in_shapes` and `in_dtypes`. Note that the shape/dtype in `in_shapes`/`in_dtypes` must follow the same order as in the MXNet model symbol file. If `in_dtypes` is provided as a single data type, then that type will be applied to all input nodes.

#### Dynamic Shape Input
We can set `dynamic=True` to turn on support for dynamic input shapes. Note that even with dynamic shapes, a set of static input shapes still need to be specified in `in_shapes`; on top of that, we'll also need to specify which dimensions of the input shapes are dynamic in `dynamic_input_shapes`. We can simply set the dynamic dimensions as `None`, e.g. `(1, 3, None, None)`, or use strings in place of the `None`'s for better understandability in the exported onnx graph, e.g. `(1, 3, 'Height', 'Width')`

```python
# The batch dimension will be dynamic in this case
in_shapes = [(1, 3, 224, 224)]
dynamic_input_shapes = [(None, 3, 224, 224)]
mx.onnx.export_model(mx_sym, mx_params, in_shapes, in_types, onnx_file,
                     dynamic=True, dynamic_input_shapes=dynamic_input_shapes)
```

#### Export Large Model
Users can set `large_model=True` to export models that are larger than 2GB. In this case, all parameter tensors will be saved into separate files along with the .onnx model file.

### Operator Support Matrix
We have implemented export logics for a wide range of MXNet operators, and thus supported most CV and NLP use cases. Below is our most up-to-date operator support matrix.

|MXNet Op|ONNX Version|
|:-|:-:|
|TODO|TODO|

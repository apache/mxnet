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
[ONNX](https://onnx.ai/), or Open Neural Network Exchange, is an open source deep learning model format that acts as a framework neutral graph representation between DL frameworks or between training and inference. With the ability to export models to the ONNX format, MXNet users can enjoy faster inference and a wider range of deployment device choices, including edge and mobile devices where MXNet installation may be constrained. Popular hardware-accelerated and/or cross-platform ONNX runtime frameworks include Nvidia [TensorRT](https://github.com/onnx/onnx-tensorrt), Microsoft [ONNXRuntime](https://github.com/microsoft/onnxruntime), Apple [CoreML](https://github.com/onnx/onnx-coreml) and [TVM](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html), etc. 

### ONNX Versions Supported
ONNX 1.7 -- Fully Supported
ONNX 1.8 -- Work in Progress

### Installation
From the 1.9 release and on, the ONNX export module has become an offical, built-in module in MXNet. You can access the module at `mxnet.onnx`. 

If you are a user of earlier MXNet versions and do not want to upgrade MXNet, you can still enjoy the latest ONNX suppor by pulling the MXNet source code and building the wheel for only the mx2onnx module. Just do `cd python/mxnet/onnx` and then build the wheel with `python3 -m build`. You should be able to find the wheel under `python/mxnet/onnx/dist/mx2onnx-0.0.0-py3-none-any.whl` and install it with `pip install mx2onnx-0.0.0-py3-none-any.whl`. You should be able to access the module with `import mx2onnx` then.

### APIs

### Operator Support Matrix - ONNX 1.7

### GluonCV Pretrained Model Support Matrix

### GluonNLP Pretrained Model Support Matrix

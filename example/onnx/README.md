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

# MXNet ONNX Export Examples

This folder contains examples that use mx2onnx module to export MXNet models to ONNX format.

Please refer to [this link](https://github.com/apache/mxnet/tree/v1.x/python/mxnet/onnx#onnx-export-support-for-mxnet)
for more details.

- cv_model_inference.py.
    - This is an end-to-end exapmle of onnx export and inference for CV models.
    - It downloads a pretrained CV model, exports the model to ONNX format, prepares input images and performs inference with ONNXRuntime.



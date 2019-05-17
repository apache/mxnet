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

# MKL-DNN Operator list

MXNet MKL-DNN backend provides optimized implementations for various operators covering a broad range of applications including image classification, object detection, natural language processing. 

To help users understanding MKL-DNN backend better, the following table summarizes the list of supported operators, data types and functionalities.  A subset of operators support faster training and inference by using a lower precision version. Refer to the following table's `INT8 Inference` column to see which operators are supported.

| Operator           | Function                   | FP32 Training (backward) | FP32 Inference | INT8 Inference |
| ---                | ---                        | ---                      | ---            | ---            |
| **Convolution**    | 1D Convolution             | Y                        | Y              | N              |
|                    | 2D Convolution             | Y                        | Y              | Y              |
|                    | 3D Convolution             | Y                        | Y              | N              |
| **Deconvolution**  | 2D Deconvolution           | Y                        | Y              | N              |
|                    | 3D Deconvolution           | Y                        | Y              | N              |
| **FullyConnected** | 1D-4D input, flatten=True  | N                        | Y              | Y              |
|                    | 1D-4D input, flatten=False | N                        | Y              | Y              |
| **Pooling**        | 2D max Pooling             | Y                        | Y              | Y              |
|                    | 2D avg pooling             | Y                        | Y              | Y              |
| **BatchNorm**      | 2D BatchNorm               | Y                        | Y              | N              |
| **LRN**            | 2D LRN                     | Y                        | Y              | N              |
| **Activation**     | ReLU                       | Y                        | Y              | Y              |
|                    | Tanh                       | Y                        | Y              | N              |
|                    | SoftReLU                   | Y                        | Y              | N              |
|                    | Sigmoid                    | Y                        | Y              | N              |
| **softmax**        | 1D-4D input                | Y                        | Y              | N              |
| **Softmax_output** | 1D-4D input                | N                        | Y              | N              |
| **Transpose**      | 1D-4D input                | N                        | Y              | N              |
| **elemwise_add**   | 1D-4D input                | Y                        | Y              | Y              |
| **Concat**         | 1D-4D input                | Y                        | Y              | Y              |
| **slice**          | 1D-4D input                | N                        | Y              | N              |
| **Quantization**   | 1D-4D input                | N                        | N              | Y              |
| **Dequantization** | 1D-4D input                | N                        | N              | Y              |
| **Requantization** | 1D-4D input                | N                        | N              | Y              |

Besides direct operator optimizations, we also provide graph fusion passes listed in the table below. Users can choose to enable or disable these fusion patterns through environmental variables.

For example, you can enable all FP32 fusion passes in the following table by:

```
export MXNET_SUBGRAPH_BACKEND=MKLDNN
```

And disable `Convolution + Activation` fusion by:

```
export MXNET_DISABLE_MKLDNN_FUSE_CONV_RELU=1
```

When generating the corresponding INT8 symbol, users can enable INT8 operator fusion passes as following:

```
# get qsym after model quantization
qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
qsym.save(symbol_name) # fused INT8 operators will be save into the symbol JSON file
```

| Fusion pattern                                            | Disable                             |
| ---                                                       | ---                                 |
| Convolution + Activation                                  | MXNET_DISABLE_MKLDNN_FUSE_CONV_RELU |
| Convolution + elemwise_add                                | MXNET_DISABLE_MKLDNN_FUSE_CONV_SUM  |
| Convolution + BatchNorm                                   | MXNET_DISABLE_MKLDNN_FUSE_CONV_BN   |
| Convolution + Activation + elemwise_add                   |                                     |
| Convolution + BatchNorm + Activation + elemwise_add       |                                     |
| FullyConnected + Activation(ReLU)                         | MXNET_DISABLE_MKLDNN_FUSE_FC_RELU   |
| Convolution (INT8) + re-quantization                      |                                     |
| FullyConnected (INT8) + re-quantization                   |                                     |
| FullyConnected (INT8) + re-quantization + de-quantization |                                     |


To install MXNet MKL-DNN backend, please refer to [MKL-DNN backend readme](MKLDNN_README.md)

For performance numbers, please refer to [performance on Intel CPU](../../faq/perf.md#intel-cpu)

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file nnpack_ops-inl.h
 * \brief
 * \author David Braude
*/

#ifndef MXNET_OPERATOR_NN_NNPACK_NNPACK_OPS_INL_H_
#define MXNET_OPERATOR_NN_NNPACK_NNPACK_OPS_INL_H_

#if MXNET_USE_NNPACK == 1

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <vector>
#include <nnpack.h>

// TODO:
// Convolutional layer
//     Inference-optimized forward propagation (nnp_convolution_inference)
//     Training-optimized forward propagation (nnp_convolution_output)
//     Training-optimized backward input gradient update (nnp_convolution_input_gradient)
//     Training-optimized backward kernel gradient update (nnp_convolution_kernel_gradient)
// Fully-connected layer
//     Inference-optimized forward propagation (nnp_fully_connected_inference and nnp_fully_connected_inference_f16f32 version for FP16 weights)
//     Training-optimized forward propagation (nnp_fully_connected_output)
// Max pooling layer
//     Forward propagation, both for training and inference, (nnp_max_pooling_output)
// ReLU layer (with parametrized negative slope)
//     Forward propagation, both for training and inference, optionally in-place, (nnp_relu_output)
//     Backward input gradient update (nnp_relu_input_gradient)

namespace mxnet {
namespace op {

/* For softmax */
void NNPACKSoftmaxForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const NDArray &in_data, const OpReqType &req,
                          const NDArray &out_data);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1

#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_

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
 * \file mkldnn_ops-inl.h
 * \brief
 * \author Da Zheng
*/

#include <mkldnn.hpp>
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

/* For fully connected. */
void MKLDNNFC_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data);
void MKLDNNFC_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &outputs);

/* For convolution. */
void MKLDNNConvolution_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data);
void MKLDNNConvolution_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs);

/* For deconvolution */
void MKLDNNDeconvolution_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data);
void MKLDNNDeconvolution_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs);

/* For softmax */
void MKLDNNSoftmax_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const NDArray &in_data, const OpReqType &req, const NDArray &out_data);

/* For sum */
void MKLDNNSum_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &inputs, const OpReqType &req, const NDArray &out_data);

/* For copy */
void MKLDNNCopy(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                const NDArray &in_data, const OpReqType &req,
                const NDArray &out_data);

}
}
#endif  // MXNET_USE_MKLDNN == 1

#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_

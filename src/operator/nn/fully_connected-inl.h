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
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <limits>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../linalg.h"
#include "../../common/utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpResource {kTempSpace};
enum FullyConnectedOpOutputs {kOut};
enum FullyConnectedGradGradOutputs {
  k_o_y_grad,
  k_x_grad_grad,
  k_w_grad_grad
};
enum Inputs {
  k_o_x_grad,
  k_o_w_grad,
};
enum InputsBias {
  k_o_b_grad = 2,
  k_o_y_bias,
};
enum InputsNoBias {
  k_o_y = 2,
};
}  // namespace fullc

namespace quantized_fullc {
enum QuantizedFCInputMinMax {kDataMin, kDataMax, kWeightMin, kWeightMax, kBiasMin, kBiasMax};
enum QuantizedFCOutputs {kOut, kOutMin, kOutMax};
}  // quantized_fullc


struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");
  }
  bool operator==(const FullyConnectedParam& other) const {
    return this->num_hidden == other.num_hidden &&
           this->no_bias == other.no_bias &&
           this->flatten == other.flatten;
  }
};

/**
 * Flatten additional dimensions after the first
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional Tensor with upper shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DTail(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape[0], shape.ProdShape(1, shape.ndim())), stream);
}

/**
 * Flatten dimensions except last
 * @tparam xpu
 * @tparam DType
 * @param tblob
 * @param ctx
 * @return 2 Dimensional tensor with front shapes collapsed
 */
template<typename xpu, typename DType>
Tensor<xpu, 2, DType> FlattenAs2DHead(const TBlob& tblob, const OpContext& ctx) {
  const TShape& shape = tblob.shape_;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  return tblob.get_with_shape<xpu, 2, DType>(
      Shape2(shape.ProdShape(0, shape.ndim()-1), shape[shape.ndim()-1]), stream);
}

template<typename xpu, typename DType>
void FCForward(const OpContext &ctx, const FullyConnectedParam &param,
               const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[fullc::kOut] == kNullOp) return;
  CHECK_EQ(req[fullc::kOut], kWriteTo);
  // TODO(bing): check the BLAS Handle, be careful
  // maybe need blas handle from context
  // TODO(bing): judge shape to remove flatten op
  Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
  CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> data, out;
  if (!param.flatten) {
    data = FlattenAs2DHead<xpu, DType>(in_data[fullc::kData], ctx);
    out = FlattenAs2DHead<xpu, DType>(out_data[fullc::kOut], ctx);
  } else {
    data = FlattenAs2DTail<xpu, DType>(in_data[fullc::kData], ctx);
    out = FlattenAs2DTail<xpu, DType>(out_data[fullc::kOut], ctx);
  }

  CHECK_EQ(data.shape_[1], wmat.shape_[1])
    << "Incomplete weight tensor detected: weight.data().shape[1] != prod(data.data().shape[1:])."
       " This is not supported by FCForward. If weight is in row_sparse format,"
       " please make sure all row ids are present.";
  // Legacy approach shown here for comparison:
  //   out = dot(data, wmat.T());
  linalg_gemm(data, wmat, out, false, true, s);
  if (!param.no_bias) {
    Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get_with_shape<xpu, 1, DType>(
      Shape1(wmat.shape_[0]), s);
    CHECK_EQ(bias.shape_[0], wmat.shape_[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";
    out += repmat(bias, data.size(0));
  }
}


template<typename xpu, typename DType>
void FCBackward(const OpContext &ctx, const FullyConnectedParam &param,
                const std::vector<TBlob> &out_grad, const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req, const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  // TODO(bing): check the BLAS Handle, be careful
  //  maybe need blas handle from context
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(stream);
  Tensor<xpu, 2, DType> x, y_grad, x_grad;
  if (!param.flatten) {
    x = FlattenAs2DHead<xpu, DType>(in_data[fullc::kData], ctx);
    y_grad = FlattenAs2DHead<xpu, DType>(out_grad[fullc::kOut], ctx);
    x_grad = FlattenAs2DHead<xpu, DType>(in_grad[fullc::kData], ctx);
  } else {
    x = FlattenAs2DTail<xpu, DType>(in_data[fullc::kData], ctx);
    y_grad = FlattenAs2DTail<xpu, DType>(out_grad[fullc::kOut], ctx);
    x_grad = FlattenAs2DTail<xpu, DType>(in_grad[fullc::kData], ctx);
  }

#if defined(__CUDACC__)
  CHECK_EQ(stream->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif

  //  backprop
  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  // gradient of weight
  Tensor<xpu, 2, DType> w_grad = in_grad[fullc::kWeight].get<xpu, 2, DType>(stream);
  // Legacy approach shown here for comparison:
  //   out = Assign(w_grad, req[fullc::kWeight], dot(grad.T(), data));
  linalg_gemm(y_grad, x, w_grad, true, false, stream, req[fullc::kWeight]);
  // gradient of bias
  if (!param.no_bias) {
    Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(stream);
    TBlob grad_blob = TBlob(y_grad);
    TBlob gbias_blob = TBlob(gbias);
    mxnet::TShape axis(1, 0);
    mxnet::TShape small;
    if (shape_assign(&gbias_blob.shape_, Shape2(param.num_hidden, 1))) {
      small = gbias_blob.shape_;
    } else {
      small = ReduceAxesShapeImpl(grad_blob.shape_,
          dmlc::optional<mxnet::TShape>(axis), true, false);
    }
    ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, false,
        mshadow_op::identity>(ctx, {grad_blob}, {req[fullc::kBias]},
            {in_grad[fullc::kBias]}, small);
  }
  // gradient of data
  // Legacy approach shown here for comparison:
  //   Assign(x_grad, req[fullc::kData], dot(y_grad, wmat));
  linalg_gemm(y_grad, wmat, x_grad, false, false, stream, req[fullc::kData]);
}

template<typename xpu>
void FullyConnectedCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCForward<xpu, float>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat64:
    FCForward<xpu, double>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}

template<typename xpu>
void FullyConnectedGradCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), 3U);  // ograd_y, x, w
  CHECK_EQ(outputs.size(), out_expected);
  CHECK_EQ(req.size(), out_expected);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCBackward<xpu, float>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat64:
    FCBackward<xpu, double>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}



///
// Inputs are:
// o_x_grad : head gradient for x_grad
// o_w_grad : head gradient for w_grad
// o_b_grad : if param.no_bias is false
// o_y : head gradient of y
//
// outputs are:
// o_y_grad : gradient of o_y
// x_grad_grad : o_y *  o_w_grad
// w_grad_grad : o_y.T * o_x_grad
//
// For implementation details see this PR: https://github.com/apache/incubator-mxnet/pull/14779

/**
 * Second order gradient for Fully Connected
 * x_grad_grad = o_y * o_w_grad
 * w_grad_grad = o_y.T * o_x_grad
 *
 * @tparam xpu
 * @tparam DType
 * @param attrs
 * @param ctx
 * @param inputs
 * @param req
 * @param outputs
 */
template<typename xpu, typename DType>
void FullyConnectedGradGradCompute(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  using namespace std;
  using namespace fullc;
  Stream<xpu> *stream = ctx.get_stream<xpu>();
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const size_t num_inputs = param.no_bias ? 3U : 4U;
  CHECK_EQ(inputs.size(), num_inputs);  // o_x_grad, o_w_grad, o_y
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(req.size(), 3U);

  // inputs
  Tensor<xpu, 2, DType> o_x_grad;
  Tensor<xpu, 2, DType> o_w_grad;
  Tensor<xpu, 2, DType> o_y;
  Tensor<xpu, 2, DType> o_b_grad;

  // outputs
  Tensor<xpu, 2, DType> o_y_grad;
  Tensor<xpu, 2, DType> x_grad_grad;
  Tensor<xpu, 2, DType> w_grad_grad;
  size_t o_y_idx = std::numeric_limits<size_t>::max();
  if (param.no_bias)
    o_y_idx = k_o_y;
  else
    o_y_idx = k_o_y_bias;
  if (!param.flatten) {
    o_x_grad = FlattenAs2DHead<xpu, DType>(inputs[k_o_x_grad], ctx);
    o_w_grad = inputs[k_o_w_grad].get<xpu, 2, DType>(stream);
    o_y = FlattenAs2DHead<xpu, DType>(inputs[o_y_idx], ctx);
  } else {
    o_x_grad = FlattenAs2DTail<xpu, DType>(inputs[k_o_x_grad], ctx);
    o_w_grad = FlattenAs2DTail<xpu, DType>(inputs[k_o_w_grad], ctx);
    o_y = inputs[o_y_idx].get<xpu, 2, DType>(stream);
    o_y_grad = outputs[k_o_y_grad].get<xpu, 2, DType>(stream);
    x_grad_grad = FlattenAs2DTail<xpu, DType>(outputs[k_x_grad_grad], ctx);
    w_grad_grad = FlattenAs2DTail<xpu, DType>(outputs[k_w_grad_grad], ctx);
  }
  linalg_gemm(o_y, o_w_grad, x_grad_grad, false, false, stream);
  linalg_gemm(o_y, o_x_grad, w_grad_grad, true, false, stream);
  if (! param.no_bias) {
    //  TODO(larroy)
  }
}


template<typename xpu>
void FullyConnectedGradGradDTypeDispatch(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const int dtype = inputs[0].type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FullyConnectedGradGradCompute<xpu, DType>(attrs, ctx, inputs, req, outputs);
  });
}


}  // namespace op
}  // namespace mxnet
namespace std {
template<>
struct hash<mxnet::op::FullyConnectedParam> {
  size_t operator()(const mxnet::op::FullyConnectedParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.num_hidden);
    ret = dmlc::HashCombine(ret, val.no_bias);
    ret = dmlc::HashCombine(ret, val.flatten);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_

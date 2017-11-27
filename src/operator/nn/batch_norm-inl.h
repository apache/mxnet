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
 * Copyright (c) 2017 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu, Chris Olivier, Da Zheng
 */
#ifndef MXNET_OPERATOR_NN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace mxnet {
namespace op {

namespace batchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta, kInMovingMean,
  kInMovingVar};  // kGamma: weights, kBeta: biases
enum BatchNormOpOutputs {kOut, kMean, kVar};  // req, out_data
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};  // aux_states

/*! \brief Default channel axis if none specified int he params */
constexpr int DEFAULT_AXIS = 1;
}  // namespace batchnorm

/*! \brief Parameters for BatchNoram operator */
struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  double eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int axis;
  bool cudnn_off;
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be no less than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(axis).set_default(mxnet::op::batchnorm::DEFAULT_AXIS)
      .describe("Specify which shape axis the channel is specified");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
      .describe("Do not select CUDNN operator, if available");
  }
};

static inline bool IsBNWriting(const OpReqType ort) {
  return ort == kWriteTo || ort == kWriteInplace;
}

template <typename xpu, typename DType, typename AccReal>
void DoBNForward(mshadow::Stream<cpu> *stream,
                 const OpContext &ctx, const BatchNormParam& param,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);

template <typename xpu, typename DType, typename AccReal>
void DoBNBackward(mshadow::Stream<cpu> *stream,
                  const OpContext &ctx, const BatchNormParam& param,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);

#if MXNET_USE_CUDA
template <typename xpu, typename DType, typename AccReal>
void DoBNForward(mshadow::Stream<gpu> *stream,
                 const OpContext &ctx, const BatchNormParam& param,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);
template <typename xpu, typename DType, typename AccReal>
void DoBNBackward(mshadow::Stream<gpu> *stream,
                  const OpContext &ctx, const BatchNormParam& param,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA

/*!
 * \brief perform a forward operation of Operator, save the output to TBlob.
 * \param ctx runtime context available to this call
 * \param in_data array of input data, it is const
 * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
 * \param out_data array of output data, pointer is used to indicate that this is holder
 *        the space of TBlob in out_data must be pre-allocated with InferShape
 * \param aux_states Auxiliary states of operator. Normally operator doesn't
 *        need, epecial case like Batch Norm requires.
 * \sa OpReqType, OpContext
 */
template <typename xpu, typename DType, typename AccReal>
void BNForward(const OpContext &ctx, const BatchNormParam& param,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_states) {
  using namespace mshadow;
  using namespace mshadow::expr;

  CHECK_EQ(in_data.size(), 3U);
  CHECK_EQ(aux_states.size(), 2U);
  if (ctx.is_train) {
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(req.size(), 3U);
  } else {
    CHECK_GE(out_data.size(), 1U);
    CHECK_GE(req.size(), 1U);
    CHECK_EQ(req[batchnorm::kOut], kWriteTo);
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  DoBNForward<xpu, DType, AccReal>(s, ctx, param, in_data, req, out_data, aux_states);
}

/*!
 * \brief Perform a Backward Operation, write gradient to the in_grad.
 *
 * \note
 * Convention:
 *   out_grad.size() == OperatorProperty.NumVisibleOutputs()
 *   out_data.size() == OperatorProperty.NumOutputs()
 * out_data can contain additional invisible returns that remembers the
 * state carried from the Forward pass. For example mask in the dropout.
 * The gradients are passed from visible returns in this function.
 *
 * \par
 * Not all the TBlobs in the arguments will be available
 * if you override the DeclareBackwardDependency of corresponding OperatorProperty class.
 * Only the dependencies you declared will be available at corresponding position,
 * the rest of the parameters are simply dummy where you will get a nullptr.
 * You will be safe if you use the default DeclareBackwardDependency.
 * But only declare what you need will give engine more chance for optimization.
 *
 * \param ctx runtime context available to this call
 * \param out_grad the gradient value we get from of the Operator.
 * \param in_data the array of input data.
 * \param out_data the array of output data.
 * \param req request types of the saving operation, can be all types.
 * \param in_grad the array of gradient we need to write to.
 * \param aux_states Auxiliary states of operator. Normally operator doesn't need
 * \sa OperatorProperty, OpReqType, OpContext
 */
template <typename xpu, typename DType, typename AccReal>
void BNBackward(const OpContext &ctx, const BatchNormParam& param,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad,
                const std::vector<TBlob> &aux_states) {
  CHECK_EQ(out_grad.size(), param.output_mean_var ? 3U : 1U);
  CHECK_EQ(in_data.size(), 3U);
  CHECK_EQ(out_data.size(), 3U);
  CHECK_EQ(in_grad.size(), 3U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  DoBNBackward<xpu, DType, AccReal>(s, ctx, param, out_grad, in_data,
                                    out_data, req, in_grad, aux_states);
}

template<typename xpu>
void BatchNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 5U);
  std::vector<TBlob> in_data(inputs.begin(),
                             inputs.begin() + batchnorm::kInMovingMean);
  std::vector<TBlob> aux_states(inputs.begin() + batchnorm::kInMovingMean,
                                inputs.end());
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    BNForward<xpu, DType, AccReal>(ctx, param, in_data, req, outputs, aux_states);
  });
}

template<typename xpu>
void BatchNormGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 11U);
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  int num_out_grads = param.output_mean_var ? 3U : 1U;
  int in_data_start = 3;
  int aux_states_start = in_data_start + batchnorm::kInMovingMean;
  int out_data_start = in_data_start + batchnorm::kInMovingVar + 1;
  std::vector<TBlob> out_grad(inputs.begin(), inputs.begin() + num_out_grads);
  std::vector<TBlob> in_data(inputs.begin() + in_data_start,
                             inputs.begin() + aux_states_start);
  std::vector<TBlob> aux_states(inputs.begin() + aux_states_start,
                                inputs.begin() + out_data_start);
  std::vector<TBlob> out_data(inputs.begin() + out_data_start, inputs.end());
  std::vector<TBlob> in_grad(outputs.begin(), outputs.begin() + 3);

  MSHADOW_REAL_TYPE_SWITCH_EX(out_grad[0].type_flag_, DType, AccReal, {
    BNBackward<xpu, DType, AccReal>(ctx, param, out_grad, in_data, out_data, req,
                                    in_grad, aux_states);
  });
}

#if DMLC_USE_CXX11

namespace batchnorm {

template<typename DType>
class BNTensor3 {
  enum { OUTER, CHANNEL, INNER, COUNT };

 public:
  inline BNTensor3(const TBlob& blob, const int indexOfChannel)
    : dptr_(blob.dptr<DType>())
      , indexOfChannel_(static_cast<size_t>(indexOfChannel < 0
                               ? (static_cast<int>(blob.shape_.ndim()) + indexOfChannel)
                               : indexOfChannel)) {
    shape_[OUTER] = 1;
    for (size_t i = 0; i < indexOfChannel_; ++i) {
      shape_[OUTER] *= blob.shape_[i];
    }
    shape_[CHANNEL] = blob.shape_[indexOfChannel_];
    shape_[INNER] = 1;
    for (size_t i = indexOfChannel_ + 1, n = blob.shape_.ndim(); i < n; ++i) {
      shape_[INNER] *= blob.shape_[i];
    }
  }

  inline BNTensor3(DType *p, const TShape& shape, const int indexOfChannel)
    : dptr_(p)
      , indexOfChannel_(static_cast<size_t>(indexOfChannel < 0
                               ? (static_cast<int>(shape.ndim()) + indexOfChannel)
                               : indexOfChannel)) {
    shape_[OUTER] = 1;
    for (size_t i = 0; i < indexOfChannel_; ++i) {
      shape_[OUTER] *= shape[i];
    }
    shape_[CHANNEL] = shape[indexOfChannel_];
    shape_[INNER] = 1;
    for (size_t i = indexOfChannel_ + 1, n = shape.ndim(); i < n; ++i) {
      shape_[INNER] *= shape[i];
    }
  }

  MSHADOW_FORCE_INLINE bool IsEmpty() const {
    return dptr_ == nullptr;
  }

  MSHADOW_XINLINE size_t Size() const {
    size_t n = 1;
    for (int i = 0; i < COUNT; ++i) {
      n *= shape_[i];
    }
    return n;
  }

  MSHADOW_XINLINE size_t ChannelCount() const {
    return shape_[CHANNEL];
  }

  MSHADOW_XINLINE size_t OuterSize() const {
    return shape_[OUTER];
  }

  MSHADOW_XINLINE size_t InnerSize() const {
    return shape_[INNER];
  }

  /*! \brief start of a given channel's spatial data */
  MSHADOW_XINLINE size_t StartOffset(const size_t channel) const {
    return channel * InnerSize();
  }

  /*! \brief This is the amount to skip to next same-channel data
   * This is the number of bytes to skip from one past the end of the current spatial data
   * to the next start of the same channel's "spatial data"
   * It is assume that the pointer being calculated points just beyond the
   * end of the last blobk of spatial data
   * i.e. RGBRGB <-- 2
   *      RRGGBB <-- 4
   **/
  MSHADOW_XINLINE size_t SkipLengthToNextSameChannelData() const {
    return (ChannelCount() - 1) * InnerSize();
  }

  MSHADOW_XINLINE size_t offset(const size_t outer,
                                const size_t channel,
                                const size_t i) const {
    const size_t spatial_size = InnerSize();
    const size_t skip_length = SkipLengthToNextSameChannelData();
    size_t off = StartOffset(channel);
    off += outer * shape_[CHANNEL] * shape_[INNER];
    const size_t skips = i / spatial_size;
    off += (1 + skip_length) * skips;
    off += i % spatial_size;
    return off;
  }

  MSHADOW_XINLINE DType& get_ref(const size_t batch,
                                 const size_t channel,
                                 const size_t i) {
    const size_t off = offset(batch, channel, i);
    return dptr_[off];
  }

  MSHADOW_XINLINE const DType& get_ref(const size_t batch,
                                       const size_t channel,
                                       const size_t i) const {
    const size_t off = offset(batch, channel, i);
    return dptr_[off];
  }

  DType *dptr_;
  size_t indexOfChannel_;
  size_t shape_[COUNT];
};

inline int GetRealAxis(const TShape& shape, int axis) {
  if (axis < 0) {
    axis += shape.ndim();
  }
  return axis;
}

extern volatile bool disable_mkl;

}  // namespace batchnorm

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif  // MXNET_OPERATOR_NN_BATCH_NORM_INL_H_


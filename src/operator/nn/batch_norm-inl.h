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
 * \author Bing Xu, Chris Olivier
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
enum BatchNormOpInputs {kData, kGamma, kBeta};  // kGamma: weights, kBeta: biases
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

/*! \brief Batch normalization operator */
template <typename xpu, typename DType, typename AccReal>
class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(BatchNormParam param) {
    this->param_ = param;
  }

  static inline bool IsWriting(const OpReqType ort) {
    return ort == kWriteTo || ort == kWriteInplace;
  }

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
  virtual void Forward(const OpContext &ctx,
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
    DoForward(s, ctx, in_data, req, out_data, aux_states);
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
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    DoBackward(s, ctx, out_grad, in_data,
               out_data, req, in_grad, aux_states);
  }

 private:
  void DoForward(mshadow::Stream<cpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);

  void DoBackward(mshadow::Stream<cpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);

#if MXNET_USE_CUDA
  void DoForward(mshadow::Stream<gpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);
  void DoBackward(mshadow::Stream<gpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA

  /*! \brief Batch normalization operator parameters */
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(BatchNormParam param, const int dtype, const TShape& shape);

#if DMLC_USE_CXX11
class BatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);

    const size_t channelAxis = static_cast<size_t>(param_.axis < 0
                            ? static_cast<int>(dshape.ndim()) + param_.axis
                            : param_.axis);
    CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param_.axis;

    const int channelCount = dshape[channelAxis];

    if (dshape.ndim() == 0) {
      return false;
    }

    in_shape->at(1) = TShape(Shape1(channelCount));
    in_shape->at(2) = TShape(Shape1(channelCount));

    out_shape->clear();
    out_shape->push_back(dshape);                // kOut
    out_shape->push_back(Shape1(channelCount));  // kMean
    out_shape->push_back(Shape1(channelCount));  // kVar

    aux_shape->clear();
    aux_shape->push_back(Shape1(channelCount));  // kMovingMean
    aux_shape->push_back(Shape1(channelCount));  // kMovingVar
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    const int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param;
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
         dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    const size_t n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (size_t i = 0; i < n_aux; ++i) {
      aux_type->push_back(dtype_param);
    }
    const size_t n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (size_t i = 1; i < n_out; ++i) {
      out_type->push_back(dtype_param);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[batchnorm::kOut],
            out_data[batchnorm::kMean],
            out_data[batchnorm::kVar],
            in_data[batchnorm::kData],
            in_data[batchnorm::kGamma]
           };
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const BatchNormParam& getParam() const {
    return param_;
  }

 private:
  BatchNormParam param_;
};  // class BatchNormProp

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


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
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu, Chris Olivier, Da Zheng
*/

#include "batch_norm-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_batch_norm-inl.h"
#endif

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {
namespace batchnorm {

/*! \brief Global disable of batchnorm mkl operator for unit testing */
volatile bool disable_mkl = false;

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const BNTensor3<DType> &tensor,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = tensor.OuterSize();
  const size_t matrixSize = tensor.InnerSize();
  const size_t skipLength = tensor.SkipLengthToNextSameChannelData();
  const size_t startOffset = tensor.StartOffset(channel);
  DType *data = tensor.dptr_ + startOffset;

  for (size_t outer = 0; outer < num; ++outer) {
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++);
    }
    data += skipLength;
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType1, typename DType2, typename OnData>
static inline void ForEachFast(const BNTensor3<DType1> &in_data,
                               const BNTensor3<DType2> &out_data,
                               const size_t channel,
                               OnData onData) {
  const size_t num         = in_data.OuterSize();
  const size_t matrixSize  = in_data.InnerSize();
  const size_t skipLength  = in_data.SkipLengthToNextSameChannelData();
  const size_t startOffset = in_data.StartOffset(channel);

  DType1  *data = in_data.dptr_  + startOffset;
  DType2 *odata = out_data.dptr_ + startOffset;

  for (size_t outer = 0; outer < num; ++outer) {
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++, odata++);
    }
    data  += skipLength;
    odata += skipLength;
  }
}

}  // namespace batchnorm

/*! \brief Forward CPU */
template <typename xpu, typename DType, typename AccReal>
void BatchNormForwardImpl(mshadow::Stream<cpu> *,
                          const OpContext &ctx, const BatchNormParam& param_,
                          const std::vector<TBlob> &in_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &out_data,
                          const std::vector<TBlob> &aux_states) {
  // Input
  batchnorm::BNTensor3<DType> inputData(in_data[batchnorm::kData], param_.axis);
  const TBlob &weights         = in_data[batchnorm::kGamma];
  const TBlob &bias            = in_data[batchnorm::kBeta];

  // Aux (Moving)
  const TBlob &runningMean     = aux_states[batchnorm::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

  // Output
  batchnorm::BNTensor3<DType> outputData(out_data[batchnorm::kOut], param_.axis);
  const TBlob &meanVector      = out_data[batchnorm::kMean];
  const TBlob &varianceVector  = out_data[batchnorm::kVar];

  AccReal *mean = meanVector.dptr<AccReal>();
  AccReal  *var = varianceVector.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;
  const size_t channelCount = inputData.ChannelCount();
  const size_t itemCountPerChannel = inputData.Size() / channelCount;

  #pragma omp parallel for
  for (int channel = 0; channel < static_cast<int>(channelCount); ++channel) {
    if (is_train_and_not_global_stats) {
      // compute mean per input
      mean[channel] = 0;
      ForEachFast(inputData, channel, [mean, channel](const DType *in_data) {
        mean[channel] += *in_data; });
      mean[channel] /= itemCountPerChannel;

      // compute variance per input
      const AccReal thisMean = mean[channel];
      var[channel] = 0;
      ForEachFast(inputData, channel,
                  [var, thisMean, channel](const DType *current_in_data) {
                    const AccReal current = *current_in_data;
                    var[channel] += (current - thisMean) * (current - thisMean);
                  });

      const AccReal sum = var[channel];

      AccReal invstd;
      if (sum == 0 && param_.eps == 0.0) {
        // Nobody likes to divide by zero
        invstd = 0;
      } else {
        const AccReal variance = sum / itemCountPerChannel;
        invstd = VARIANCE_TO_INVSTD(variance, param_.eps);
      }
      var[channel] = invstd;
    } else {
      const AccReal *rm = runningMean.dptr<AccReal>();
      const AccReal *rv = runningVariance.dptr<AccReal>();

      mean[channel] = rm[channel];
      var[channel] = VARIANCE_TO_INVSTD(rv[channel], param_.eps);
    }

    // compute output
    AccReal *w = weights.dptr<AccReal>();
    const AccReal *b = bias.dptr<AccReal>();

    const AccReal thisMean = mean[channel];
    const AccReal thisInvstd = var[channel];
    const AccReal thisWeight = w[channel];
    const AccReal thisBias = b[channel];

    // note that var is still invstd
    if (!param_.fix_gamma) {
      if (IsBNWriting(req[batchnorm::kData])) {
        ForEachFast(inputData, outputData, channel,
                    [thisWeight, thisBias, thisMean, thisInvstd](const DType *in_data,
                                                                 DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - thisMean) * thisInvstd) * thisWeight + thisBias);
                    });
      }
    } else {
      if (IsBNWriting(req[batchnorm::kGamma])) {
        w[channel] = AccReal(1);
      }
      if (IsBNWriting(req[batchnorm::kData])) {
        ForEachFast(inputData, outputData, channel,
                    [thisWeight, thisBias, thisMean, thisInvstd](const DType *in_data,
                                                                 DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - thisMean) * thisInvstd) + thisBias);
                    });
      }
    }
  }
}

template <typename xpu, typename DType, typename AccReal>
void BatchNormBackwardImpl(mshadow::Stream<cpu> *,
                           const OpContext &ctx, const BatchNormParam& param_,
                           const std::vector<TBlob> &out_grad,
                           const std::vector<TBlob> &in_data,
                           const std::vector<TBlob> &out_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &in_grad,
                           const std::vector<TBlob> &aux_states) {
  // Input Data
  batchnorm::BNTensor3<DType> inputData(in_data[batchnorm::kData], param_.axis);
  const TBlob &weights   = in_data[batchnorm::kGamma];

  // Input Grad
  batchnorm::BNTensor3<DType> gradIn(in_grad[batchnorm::kData], param_.axis);
  const TBlob &gradWeight = in_grad[batchnorm::kGamma];
  const TBlob &gradBias   = in_grad[batchnorm::kBeta];

  // Aux (Moving)
  const TBlob &runningMean = aux_states[batchnorm::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

  // Output
  batchnorm::BNTensor3<DType> gradOut(out_grad[batchnorm::kOut], param_.axis);
  const TBlob &saveMean = out_data[batchnorm::kMean];
  const TBlob &saveStd  = out_data[batchnorm::kVar];

  const size_t channelCount = inputData.ChannelCount();
  const size_t itemCount    = inputData.Size() / channelCount;

  // Avoid multiple dptr() call within the channel loop
  AccReal *runningMeanDataPtr = runningMean.dptr<AccReal>();
  AccReal *runningVarDataPtr  = runningVariance.dptr<AccReal>();
  const AccReal *saveMeanDataPtr = saveMean.dptr<AccReal>();
  const AccReal *saveInvStdDataPtr = saveStd.dptr<AccReal>();
  AccReal *gradWeightData = gradWeight.dptr<AccReal>();
  AccReal *gradBiasData = gradBias.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;

  #pragma omp parallel for
  for (int channel = 0; channel < static_cast<int>(channelCount); ++channel) {
    const AccReal *weight = weights.dptr<AccReal>();
    const AccReal w = !param_.fix_gamma ? weight[channel] : AccReal(1);
    AccReal mean, invstd;
    if (is_train_and_not_global_stats) {
      mean = saveMeanDataPtr[channel];
      invstd = saveInvStdDataPtr[channel];
      const AccReal variance = INVSTD_TO_VARIANCE(invstd, param_.eps);

      // update running averages
      runningMeanDataPtr[channel] = runningMeanDataPtr[channel] * param_.momentum
                                    + mean * (AccReal(1) - param_.momentum);

      runningVarDataPtr[channel] = runningVarDataPtr[channel] * param_.momentum
                                   + variance * (AccReal(1) - param_.momentum);

    } else {
      mean = runningMeanDataPtr[channel];
      invstd = VARIANCE_TO_INVSTD(runningVarDataPtr[channel], param_.eps);
    }

    // sumGradOut over all gradOutput in feature plane
    AccReal sumGradOut = 0;
    ForEachFast(gradOut, static_cast<size_t>(channel),
                [&sumGradOut](const DType *gradOut_data) {
                  sumGradOut += *gradOut_data;
                });

    // dot product of the Q(X) and gradOuput
    AccReal dotp = 0;
    ForEachFast(inputData, gradOut, static_cast<size_t>(channel),
                [&dotp, mean](const DType *thisInputData, const DType *gradOut_data) {
                  dotp += (*thisInputData - mean) * (*gradOut_data);
                });

    if (!gradIn.IsEmpty() && IsBNWriting(req[batchnorm::kData])) {  // if there's a grad input
      if (is_train_and_not_global_stats) {
        // when in training mode
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

        // projection of gradOutput on to output scaled by std
        const AccReal k = dotp * invstd * invstd / itemCount;
        ForEachFast(inputData, gradIn, static_cast<size_t>(channel),
                    [&mean, &k](const DType *inputDataPtr, DType *gradIn_data) {
                      *gradIn_data = (*inputDataPtr - mean) * k;
                    });

        const AccReal iw = invstd * w;
        const AccReal gradMean = sumGradOut / itemCount;
        ForEachFast(gradOut, gradIn, static_cast<size_t>(channel),
                    [iw, gradMean](const DType *gradOut_data, DType *gradIn_data) {
                      *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * iw;
                    });
      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        const AccReal iw = invstd * w;
        ForEachFast(gradOut, gradIn, static_cast<size_t>(channel),
                    [iw](const DType *gradOut_data, DType *gradIn_data) {
                      *gradIn_data = *gradOut_data * iw;
                    });
      }
    }

    // May want to make this a param eventually
    const AccReal scale = 1.0f;

    if (IsBNWriting(req[batchnorm::kGamma])) {
      if (!param_.fix_gamma) {
        gradWeightData[channel] = scale * dotp * invstd;
      } else {
        gradWeightData[channel] = AccReal(0);
      }
    }

    if (IsBNWriting(req[batchnorm::kBeta])) {
      gradBiasData[channel] = scale * sumGradOut;
    }
  }
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

static bool BatchNormShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, MovingMean, MovingVar]";
  const TShape &dshape = in_shape->at(batchnorm::kData);

  const size_t channelAxis = static_cast<size_t>(param.axis < 0
      ? static_cast<int>(dshape.ndim()) + param.axis
      : param.axis);
  CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param.axis;

  const int channelCount = dshape[channelAxis];

  if (dshape.ndim() == 0) {
    return false;
  }

  in_shape->at(batchnorm::kGamma) = TShape(Shape1(channelCount));
  in_shape->at(batchnorm::kBeta) = TShape(Shape1(channelCount));
  in_shape->at(batchnorm::kInMovingMean) = TShape(Shape1(channelCount));  // kMovingMean
  in_shape->at(batchnorm::kInMovingVar) = TShape(Shape1(channelCount));  // kMovingVar

  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  out_shape->push_back(Shape1(channelCount));  // kMean
  out_shape->push_back(Shape1(channelCount));  // kVar

  return true;
}

static bool BatchNormType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type, std::vector<int> *out_type) {
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
  std::vector<std::string> args{"data", "gamma", "beta", "mean", "var"};
  CHECK_LE(in_type->size(), args.size());
  for (index_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  const size_t n_out = 3;
  out_type->clear();
  out_type->push_back(dtype);
  for (size_t i = 1; i < n_out; ++i) {
    out_type->push_back(dtype_param);
  }
  return true;
}

static inline bool similar_array(const mxnet::NDArray &arr1,
                                 const mxnet::NDArray &arr2,
                                 float tol) {
  float *data1 = reinterpret_cast<float *>(arr1.data().dptr_);
  float *data2 = reinterpret_cast<float *>(arr2.data().dptr_);
  if (arr1.shape().Size() != arr2.shape().Size())
      return false;
  for (size_t i = 0; i < arr1.shape().Size(); i++) {
      if (std::abs(data1[i] - data2[i]) > tol) {
          // printf("similar_array: %.8f, %.8f \n", data1[i], data2[i]);
          return false;
      }
  }
  std::cout << "similar_array: passed all check, tol=" << tol << std::endl;
  return true;
}

#if MXNET_USE_MKLDNN == 1
static inline bool SupportMKLDNNBN(const NDArray &input, const BatchNormParam &param) {
  TShape shape = input.shape();
  bool support = input.storage_type() == kMKLDNNStorage && shape.ndim() == 4
      && param.axis == mxnet::op::batchnorm::DEFAULT_AXIS
      && shape[param.axis] % 8 == 0;
  if (support) {
    // We need to test its data layout. MKLDNN batchnorm doesn't work well on
    // the default layout.
    auto mem = input.GetMKLDNNData();
    auto desc = mem->get_primitive_desc().desc();
    support = desc.data.format != GetDefaultFormat(desc);
  }
  return support;
}
#endif

void BatchNormCompute_CPU(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const std::vector<NDArray> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 5U);
#if MXNET_USE_MKLDNN == 1
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  if (SupportMKLDNNBN(inputs[0], param)) {
    std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);
    std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());

    switch (inputs[0].dtype()) {
      case mshadow::kFloat32:
        MKLDNNBatchNorm_Forward<float>(ctx, param, in_data, req, outputs, aux_states);
        return;
    }
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) {
    in_blobs[i] = inputs[i].data();
  }

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++) {
    out_blobs[i] = outputs[i].data();
  }
  BatchNormCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

void BatchNormGradCompute_CPU(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 11U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  int num_out_grads = param.output_mean_var ? 3U : 1U;
  int in_data_start = 3;
  int aux_states_start = in_data_start + batchnorm::kInMovingMean;
  int out_data_start = in_data_start + batchnorm::kInMovingVar + 1;

  TShape shape = inputs[0].shape();
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNNBN(inputs[0], param)
      && inputs[in_data_start].storage_type() == kMKLDNNStorage) {
    std::vector<NDArray> out_grad(inputs.begin(), inputs.begin() + num_out_grads);
    std::vector<NDArray> in_data(inputs.begin() + in_data_start,
                                 inputs.begin() + aux_states_start);
    std::vector<NDArray> aux_states(inputs.begin() + aux_states_start,
                                    inputs.begin() + out_data_start);
    std::vector<NDArray> out_data(inputs.begin() + out_data_start, inputs.end());
    std::vector<NDArray> in_grad(outputs.begin(), outputs.begin() + 3);

    if (inputs[0].dtype() == mshadow::kFloat32) {
      MKLDNNBatchNorm_Backward<float>(ctx, param, out_grad, in_data,
                                      out_data, req, in_grad, aux_states);
      return;
    }
  }
#endif
  // cast NDArray to TBlob, and call original implementation.
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) {
    in_blobs[i] = inputs[i].data();
  }

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++) {
    out_blobs[i] = outputs[i].data();
  }
  BatchNormGradCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

static inline bool BatchNormStorageType(const nnvm::NodeAttrs &attrs,
                                        const int dev_mask,
                                        DispatchMode *dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 5);
  CHECK_EQ(out_attrs->size(), 3);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && (*in_attrs)[0] == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    for (int& v : *in_attrs) {
      if (v == kUndefinedStorage) v = kDefaultStorage;
    }
    (*out_attrs)[0] = kMKLDNNStorage;
    (*out_attrs)[1] = kDefaultStorage;
    (*out_attrs)[2] = kDefaultStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}

static inline bool backward_BatchNormStorageType(const nnvm::NodeAttrs &attrs,
                                                 const int dev_mask,
                                                 DispatchMode *dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 11);
  CHECK_EQ(out_attrs->size(), 5);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && (*in_attrs)[0] == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    for (int& v : *in_attrs) {
      if (v == kUndefinedStorage) v = kDefaultStorage;
    }
    (*out_attrs)[0] = kMKLDNNStorage;
    (*out_attrs)[1] = kDefaultStorage;
    (*out_attrs)[2] = kDefaultStorage;
    (*out_attrs)[3] = kDefaultStorage;
    (*out_attrs)[4] = kDefaultStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& v : *in_attrs) {
    if (v == - 1) v = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}

NNVM_REGISTER_OP(BatchNorm)
.describe(R"code(Batch normalization.

Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis:

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_var`` as well, which are needed for the backward pass.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

If ``use_global_stats`` is set to be true, then ``moving_mean`` and
``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
the output. It is often used during inference.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
then set ``gamma`` to 1 and its gradient to 0.

)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "var"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{3, 4};
})
.set_attr<nnvm::FInferShape>("FInferShape", BatchNormShape)
.set_attr<nnvm::FInferType>("FInferType", BatchNormType)
.set_attr<FInferStorageType>("FInferStorageType", BatchNormStorageType)
.set_attr<FCompute>("FCompute<cpu>", BatchNormCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormCompute_CPU)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_BatchNorm"})
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BatchNormParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == 3) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == 4) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_BatchNorm)
.set_num_outputs(5)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", backward_BatchNormStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<FCompute>("FCompute<cpu>", BatchNormGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormGradCompute_CPU);

}  // namespace op
}  // namespace mxnet

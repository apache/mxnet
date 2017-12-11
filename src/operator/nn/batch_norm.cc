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
 * \author Bing Xu, Chris Olivier
*/

#include "batch_norm-inl.h"
#include <nnvm/op_attr_types.h>
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017

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
void BatchNormOp<xpu, DType, AccReal>::DoForward(mshadow::Stream<cpu> *,
                                                 const OpContext &ctx,
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
      if (IsWriting(req[batchnorm::kData])) {
        ForEachFast(inputData, outputData, channel,
                    [thisWeight, thisBias, thisMean, thisInvstd](const DType *in_data,
                                                                 DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - thisMean) * thisInvstd) * thisWeight + thisBias);
                    });
      }
    } else {
      if (IsWriting(req[batchnorm::kGamma])) {
        w[channel] = AccReal(1);
      }
      if (IsWriting(req[batchnorm::kData])) {
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
void BatchNormOp<xpu, DType, AccReal>::DoBackward(mshadow::Stream<cpu> *,
                                                  const OpContext &ctx,
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

    if (!gradIn.IsEmpty() && IsWriting(req[batchnorm::kData])) {  // if there's a grad input
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

    if (IsWriting(req[batchnorm::kGamma])) {
      if (!param_.fix_gamma) {
        gradWeightData[channel] = scale * dotp * invstd;
      } else {
        gradWeightData[channel] = AccReal(0);
      }
    }

    if (IsWriting(req[batchnorm::kBeta])) {
      gradBiasData[channel] = scale * sumGradOut;
    }
  }
}

template<>
Operator *CreateOp<cpu>(BatchNormParam param, const int dtype, const TShape& shape) {
  param.axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  Operator *op = nullptr;
#if MXNET_USE_MKL2017 == 1
  if (shape.ndim() == 4
      && param.axis == mxnet::op::batchnorm::DEFAULT_AXIS
      && !mxnet::op::batchnorm::disable_mkl) {
    switch (dtype) {
      case mshadow::kFloat32:
        op = new MKLBatchNormOp<cpu, float>(param);
        break;
      case mshadow::kFloat64:
        op = new MKLBatchNormOp<cpu, double>(param);
        break;
      default:
        // MKL operator doesn't support half_t, so fall through
        break;
    }
  }
#endif
  if (!op) {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                                DType,
                                AccReal, {
                                  op = new BatchNormOp<cpu, DType, AccReal>(param); });
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], (*in_shape)[0]);
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
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
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(BatchNorm)
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

}  // namespace op
}  // namespace mxnet

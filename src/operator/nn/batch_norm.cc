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
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_batch_norm-inl.h"
#endif

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/std::sqrt((__var$) + DType(__eps$)))
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
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, MovingMean, MovingVar]";
  CHECK_EQ(out_shape->size(), 3U);
  const mxnet::TShape &dshape = in_shape->at(batchnorm::kData);

  const size_t channelAxis = static_cast<size_t>(param.axis < 0
      ? static_cast<int>(dshape.ndim()) + param.axis
      : param.axis);
  CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param.axis;

  const int channelCount = dshape[channelAxis];

  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  in_shape->at(batchnorm::kGamma) = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnorm::kBeta) = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnorm::kInMovingMean) = mxnet::TShape(Shape1(channelCount));  // kMovingMean
  in_shape->at(batchnorm::kInMovingVar) = mxnet::TShape(Shape1(channelCount));  // kMovingVar

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
  for (size_t i = 1; i < in_type->size(); ++i) {
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

#if MXNET_USE_MKLDNN == 1
static inline bool SupportMKLDNNBN(const NDArray &input, const BatchNormParam &param) {
  mxnet::TShape shape = input.shape();
  return SupportMKLDNN(input) && shape.ndim() == 4
      && param.axis == mxnet::op::batchnorm::DEFAULT_AXIS
      && shape[param.axis] % 8 == 0
      && !mxnet::op::batchnorm::disable_mkl;
}

void BatchNormComputeExCPU(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 5U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  // MKLDNN batchnorm only works well on the special MKLDNN layout.
  if (SupportMKLDNNBN(inputs[0], param) && inputs[0].IsMKLDNNData()) {
    std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);
    std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());

    if (inputs[0].dtype() == mshadow::kFloat32) {
      MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
      MKLDNNBatchNormForward<float>(ctx, param, in_data, req, outputs, aux_states);
      MKLDNN_OPCHECK_RUN(BatchNormCompute<cpu>, attrs, ctx, inputs, req, outputs);
      return;
    }
  }
  FallBackCompute(BatchNormCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

void BatchNormGradComputeExCPU(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 8U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);

  mxnet::TShape shape = inputs[0].shape();
  // MKLDNN batchnorm only works well on the special MKLDNN layout.
  if (SupportMKLDNNBN(inputs[0], param)
      && (inputs[3].IsMKLDNNData() || inputs[0].IsMKLDNNData())) {
    std::vector<NDArray> out_grad(1);
    std::vector<NDArray> out_data(3);
    std::vector<NDArray> in_data(3);
    std::vector<NDArray> aux_states(2);
    out_grad[0] = inputs[0];
    out_data[batchnorm::kMean] = inputs[1];
    out_data[batchnorm::kVar] = inputs[2];
    in_data[batchnorm::kData] = inputs[3];
    in_data[batchnorm::kGamma] = inputs[4];
    in_data[batchnorm::kBeta] = inputs[5];
    aux_states[batchnorm::kMovingMean] = inputs[6];
    aux_states[batchnorm::kMovingVar] = inputs[7];
    const std::vector<NDArray> &in_grad = outputs;

    if (inputs[0].dtype() == mshadow::kFloat32) {
      MKLDNN_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
      MKLDNNBatchNormBackward<float>(ctx, param, out_grad, in_data,
                                     out_data, req, in_grad, aux_states);
      MKLDNN_OPCHECK_RUN(BatchNormGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
      return;
    }
  }
  FallBackCompute(BatchNormGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

static inline bool BatchNormStorageType(const nnvm::NodeAttrs &attrs,
                                        const int dev_mask,
                                        DispatchMode *dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);

  bool dispatched = false;
#if MXNET_USE_MKLDNN == 1
  if (!dispatched) {
    dispatched = MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode,
                                   in_attrs, out_attrs);
  }
  if (!MKLDNNEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  }
#else
  for (int& v : *in_attrs)
    if (v == - 1) v = kDefaultStorage;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
#endif
  if (!common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) && param.fix_gamma) {
    LOG(FATAL) << "fix_gamma=True is not supported for sparse ndarrays. Tracked at #11647";
  }
  return dispatched;
}

std::vector<nnvm::NodeEntry> BatchNormGrad(const nnvm::NodePtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> out_data;
  out_data.reserve(n->num_outputs());
  for (size_t i = 0; i < n->num_outputs(); ++i)
    out_data.emplace_back(n, i, 0);
  std::vector<nnvm::NodeEntry> heads;
  heads.reserve(8);
  heads.emplace_back(ograds.at(0));
  heads.emplace_back(out_data.at(batchnorm::kMean));
  heads.emplace_back(out_data.at(batchnorm::kVar));
  heads.emplace_back(n->inputs.at(batchnorm::kData));
  heads.emplace_back(n->inputs.at(batchnorm::kGamma));
  heads.emplace_back(n->inputs.at(batchnorm::kBeta));
  heads.emplace_back(n->inputs.at(batchnorm::kInMovingMean));
  heads.emplace_back(n->inputs.at(batchnorm::kInMovingVar));

  nnvm::NodePtr gnode = nnvm::Node::Create();
  gnode->inputs = std::move(heads);
  gnode->control_deps.emplace_back(n);
  gnode->attrs = n->attrs;
  gnode->attrs.op = nnvm::Op::Get("_backward_BatchNorm");
  gnode->attrs.name = n->attrs.name + "_backward";
  // The input of batchnorm
  std::vector<nnvm::NodeEntry> in_grad;
  in_grad.reserve(5);
  for (size_t i = 0; i < 3; ++i)
    in_grad.emplace_back(gnode, i, 0);
  // attach no gradient node to forbid gradient on aux_state
  nnvm::NodePtr ng = nnvm::Node::Create();
  ng->attrs.op = Op::Get("_NoGradient");
  ng->attrs.name = "NoGradient";
  // the aux state of batchnorm
  for (size_t i = 3; i < 5; ++i)
    in_grad.emplace_back(ng);
  return in_grad;
}

NNVM_REGISTER_OP(BatchNorm)
.add_alias("_npx_BatchNorm")
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
the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
two outputs are blocked.

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

.. Note::
  When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,
  the sparse tensors will fallback.

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
.set_attr<mxnet::FInferShape>("FInferShape", BatchNormShape)
.set_attr<nnvm::FInferType>("FInferType", BatchNormType)
.set_attr<FInferStorageType>("FInferStorageType", BatchNormStorageType)
.set_attr<FCompute>("FCompute<cpu>", BatchNormCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormComputeExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", BatchNormGrad)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
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
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BatchNormStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr_parser(ParamParser<BatchNormParam>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", BatchNormGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet

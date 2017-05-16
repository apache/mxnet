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
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {
namespace batchnorm {

template<typename DType>
class DeviceTensor3 {
  DeviceTensor3(const DeviceTensor3&) = delete;

 public:
  inline DeviceTensor3(const TBlob& blob, const size_t indexOfChannel)
    : dptr_(blob.dptr<DType>())
      , indexOfChannel_(indexOfChannel)
      , shape_(3) {
    if (indexOfChannel) {
      shape_[0] = 1;
      for (size_t i = 0; i < indexOfChannel_; ++i) {
        shape_[0] *= blob.shape_[i];
      }
    } else {
      shape_[0] = 0;
    }
    shape_[1] = blob.shape_[indexOfChannel_];
    shape_[2] = 1;
    for (size_t i = indexOfChannel_ + 1, n = blob.shape_.ndim(); i < n; ++i) {
      shape_[2] *= blob.shape_[i];
    }
  }

  inline size_t Size() const {
    size_t n = 1;
    for (int i = 0; i < 3; ++i) {
      n *= shape_[i];
    }
    return n;
  }

  inline size_t ChannelCount() const {
    return shape_[1];
  }

  inline size_t BatchSize() const {
    return shape_[0];
  }

  inline size_t SpatialSize() const {
    return shape_[2];
  }

  DType *dptr_;
  size_t indexOfChannel_;
  TShape shape_;
};

/*! \brief offset, given indices such as bn, channel, depth, row, column */
static inline index_t offset(const TShape& shape,
                             const size_t *indices,
                             const size_t indicesSize) {
  const size_t dim = shape.ndim();
  size_t offset = 0;
  for (size_t i = 0; i < dim; ++i) {
    offset *= shape[i];
    if (indicesSize > i) {
      offset += indices[i];
    }
  }
  return offset;
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType> &tensor,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = tensor.BatchSize();
  const size_t matrixSize = tensor.SpatialSize();

  size_t indices[2] = {0, channel};

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
    indices[0] = batchItem;
    DType *data = tensor.dptr_ + offset(tensor.shape_, &indices[0],
                                        sizeof(indices)/sizeof(indices[0]));
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++);
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType1, typename DType2, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType1> &in_data,
                               const DeviceTensor3<DType2> &out_data,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = in_data.BatchSize();
  const size_t matrixSize = in_data.SpatialSize();

  size_t indices[2] = {0, channel};

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
    indices[0] = batchItem;
    const size_t off = offset(in_data.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
    const DType1 *data = in_data.dptr_ + off;
    DType2 *odata = out_data.dptr_ + off;
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++, odata++);
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType>& tensor,
                               OnData onData) {
  const size_t num        = tensor.BatchSize();
  const size_t channels   = tensor.ChannelCount();
  const size_t matrixSize = tensor.SpatialSize();

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
#pragma openmp for
    for (size_t channel = 0; channel < channels; ++channel) {
      size_t indices[2] = { batchItem, channel };
      const size_t off = offset(tensor.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      const DType *inData = tensor.dptr_ + off;
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(channel, inData++);
      }
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType>& in_data,
                               const DeviceTensor3<DType>& out_data,
                               OnData onData) {
  const size_t num        = in_data.BatchSize();
  const size_t channels   = in_data.ChannelCount();
  const size_t matrixSize = in_data.SpatialSize();

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
#pragma omp parallel for
    for (int channel = 0; channel < channels; ++channel) {
      size_t indices[2] = { batchItem, static_cast<size_t>(channel) };
      const size_t off = offset(in_data.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      const DType *inData = in_data.dptr_ + off;
      DType *outData = out_data.dptr_ + off;
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(channel, inData++, outData++);
      }
    }
  }
}

/*! \brief Compute the mean of each input channel */
template<typename DType, typename AccReal>
static inline void ComputeMean(const DeviceTensor3<DType> &tensor,
                               AccReal *save_mean) {
  const size_t channelCount = tensor.ChannelCount();

  for (size_t i = 0; i < channelCount; ++i) {
    save_mean[i] = 0;
  }

  ForEachFast(tensor,
              [&save_mean](const size_t channel, const DType *in_data){
                save_mean[channel] += *in_data;
              });

  const size_t itemCount = tensor.Size() / channelCount;
  for (size_t i = 0, n = channelCount; i < n; ++i) {
    save_mean[i] /= itemCount;
  }
}

/*! \brief Compute the variance of each input channel, as well as update moving mean/variants */
template<typename DType, typename AccReal>
static inline void ComputeVariance(const DeviceTensor3<DType> &tensor,
                                   const AccReal *mean_data,
                                   const DType eps,
                                   const TShape &oshape,
                                   AccReal *save_std) {
  const size_t channels   = tensor.ChannelCount();
  for (size_t i = 0; i < channels; ++i) {
    save_std[i] = 0;
  }
  ForEachFast(tensor,
              [&save_std, &mean_data](const index_t channel, const DType *current_in_data) {
                const AccReal mean = mean_data[channel];
                const AccReal current = *current_in_data;
                save_std[channel] += (current - mean) * (current - mean);
              });

  const size_t itemCount = tensor.Size() / channels;
#pragma omp parallel for
  for (int channel = 0; channel < channels; ++channel) {
    const AccReal sum = save_std[channel];

    AccReal invstd;
    if (sum == 0 && eps == 0.0) {
      // Nobody likes to divide by zero
      invstd = 0;
    } else {
      const AccReal variance = sum/itemCount;
      invstd = VARIANCE_TO_INVSTD(variance, eps);
    }
    save_std[channel] = invstd;
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
  batchnorm::DeviceTensor3<DType> inputData(in_data[batchnorm::kData], 1);
  const TBlob &weights         = in_data[batchnorm::kGamma];
  const TBlob &bias            = in_data[batchnorm::kBeta];

  // Aux (Moving)
  const TBlob &runningMean     = aux_states[batchnorm::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

  // Output
  batchnorm::DeviceTensor3<DType> outputData(out_data[batchnorm::kOut], 1);
  const TBlob &meanVector      = out_data[batchnorm::kMean];
  const TBlob &varianceVector  = out_data[batchnorm::kVar];

  AccReal *mean = meanVector.dptr<AccReal>();
  AccReal  *var = varianceVector.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;

  if (is_train_and_not_global_stats) {
    // compute mean per input
    ComputeMean(inputData, meanVector.dptr<AccReal>());

    // compute variance per input
    ComputeVariance(inputData,
                    meanVector.dptr<AccReal>(),
                    static_cast<DType>(param_.eps),
                    varianceVector.shape_,
                    var);  // var is actually returned as invstd
  } else {
    const AccReal *rm = runningMean.dptr<AccReal>();
    const AccReal *rv = runningVariance.dptr<AccReal>();

    for (size_t i = 0, n = inputData.shape_[1]; i < n; ++i) {
      mean[i] = rm[i];
      var[i] = VARIANCE_TO_INVSTD(rv[i], param_.eps);
    }
  }

  // compute output
  AccReal        *w = weights.dptr<AccReal>();
  const AccReal  *b = bias.dptr<AccReal>();

    // note that var is still invstd
    if (!param_.fix_gamma) {
      if (IsWriting(req[batchnorm::kData])) {
        ForEachFast(inputData, outputData,
                    [w, b, mean, var](const size_t channel, const DType *in_data, DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - mean[channel]) * var[channel]) * w[channel] + b[channel]);
                    });
      }
    } else {
      if (IsWriting(req[batchnorm::kGamma])) {
        for (size_t i =0, n = weights.Size(); i < n; ++i) {
          w[i] = AccReal(1);
        }
      }
      if (IsWriting(req[batchnorm::kData])) {
        ForEachFast(inputData, outputData,
                    [w, b, mean, var](const size_t channel, const DType *in_data, DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - mean[channel]) * var[channel]) + b[channel]);
                    });
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
  batchnorm::DeviceTensor3<DType> inputData(in_data[batchnorm::kData], 1);
  const TBlob &weights   = in_data[batchnorm::kGamma];

  // Input Grad
  batchnorm::DeviceTensor3<DType> gradIn(in_grad[batchnorm::kData], 1);
  const TBlob &gradWeight = in_grad[batchnorm::kGamma];
  const TBlob &gradBias   = in_grad[batchnorm::kBeta];

  // Aux (Moving)
  const TBlob &runningMean = aux_states[batchnorm::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

  // Output
  batchnorm::DeviceTensor3<DType> gradOut(out_grad[batchnorm::kOut], 1);
  const TBlob &saveMean = out_data[batchnorm::kMean];
  const TBlob &saveStd  = out_data[batchnorm::kVar];

  const size_t channelCount = inputData.shape_[1];
  const size_t itemCount    = inputData.Size() / channelCount;

  // Avoid multiple dptr() call within the channel loop
  AccReal *runningMeanDataPtr = runningMean.dptr<AccReal>();
  AccReal *runningVarDataPtr  = runningVariance.dptr<AccReal>();
  AccReal *saveMeanDataPtr = saveMean.dptr<AccReal>();
  AccReal *saveInvStdDataPtr = saveStd.dptr<AccReal>();
  AccReal *gradWeightData = gradWeight.dptr<AccReal>();
  AccReal *gradBiasData = gradBias.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;

  #pragma omp parallel for
  for (int channel = 0; channel < static_cast<int>(channelCount); ++channel) {
    AccReal *weight = weights.dptr<AccReal>();
    const AccReal w = weight ? weight[channel] : AccReal(1);
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

    if (gradIn.shape_.ndim() && IsWriting(req[batchnorm::kData])) {  // if there's a grad input
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
Operator *CreateOp<cpu>(const BatchNormParam& param, const int dtype, const TShape& shape) {
  Operator *op = nullptr;
#if MXNET_USE_MKL2017 == 1
  if (shape.ndim() == 4) {
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
#define BATCHNORM_LOG_MKL_INFO() do { \
  LOG(INFO) << MKLBatchNormOp<cpu, float>::getName() \
    << " Skipping MKL optimization (unsupported dimension or type)"; \
  } while (0)
#else
#define BATCHNORM_LOG_MKL_INFO() ((void)0)
#endif
  if (!op) {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                                DType,
                                AccReal, {
                                  BATCHNORM_LOG_MKL_INFO();
                                  op = new BatchNormOp<cpu, DType, AccReal>(param); });
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK_GE(in_shape->size(), 1U);
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


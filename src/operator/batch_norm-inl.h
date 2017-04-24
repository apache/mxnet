/*!
 * Copyright (c) 2017 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu, Chris Olivier
 */
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"
#include "mxnet_op.h"

namespace mxnet {
namespace op {

namespace batchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta};  // kGamma: weights, kBeta: biases
enum BatchNormOpOutputs {kOut, kMean, kVar};  // req, out_data
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};  // aux_states
enum BatchNormBackResource {kTempSpace};
}  // namespace batchnorm

/*! \brief Parameters for BatchNoram operator */
struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int channel_position;
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be bigger than CUDNN_BN_MIN_EPSILON "
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
    DMLC_DECLARE_FIELD(channel_position).set_default(1)
      .describe("Position of the channel field in the shape array");
  }
};

template<typename DType>
class DeviceTensor3 {
  DeviceTensor3(const DeviceTensor3&) = delete;
 public:
  inline DeviceTensor3(const TBlob& blob, const size_t indexOfChannel)
    : dptr_(blob.dptr<DType>())
      , shape_(3) {
    shape_[0] = 1;
    for(size_t i = 0; i < indexOfChannel; ++i) {
      shape_[0] *= blob.shape_[i];
    }
    shape_[1] = blob.shape_[indexOfChannel];
    shape_[2] = 1;
    for(size_t i = indexOfChannel + 1, n = blob.shape_.ndim(); i < n; ++i) {
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
  TShape shape_;
};


/*! \brief Batch normalization operator */
template<typename xpu, typename DType, typename AccType>
class BatchNormOp : public Operator
                  , public Callbacker<Operator> {
  typedef ::nnvm::TShape TShape;
  typedef ::mxnet::TBlob TBlob;


 private:
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
  template<typename OnData>
  static inline void forEachFast(DeviceTensor3<DType>& tensor, const size_t channel, OnData onData) {
    const size_t num        = tensor.BatchSize();
    const size_t matrixSize = tensor.SpatialSize();

    size_t indices[2] = {0, channel};

    for (size_t batchItem = 0; batchItem < num; ++batchItem) {
      indices[0] = batchItem;
      DType *data = tensor.dptr_ + offset(tensor.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(data++);
      }
    }
  }

  /*! \brief Fast-foreach when you don't care about the position other than channel */
  template<typename OnData>
  inline void forEachFast(DeviceTensor3<DType>& in_data, DeviceTensor3<DType>& out_data,
                          const size_t channel, OnData onData) {
    const size_t num        = in_data.BatchSize();
    const size_t matrixSize = in_data.SpatialSize();

    size_t indices[2] = {0, channel};

    for (size_t batchItem = 0; batchItem < num; ++batchItem) {
      indices[0] = batchItem;
      const size_t off = offset(in_data.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      const DType *data = in_data.dptr_ + off;
      DType *odata = out_data.dptr_ + off;
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(data++, odata++);
      }
    }
  }

  /*! \brief Fast-foreach when you don't care about the position other than channel */
  template<typename OnData>
  static inline void forEachFast(DeviceTensor3<DType>& tensor, OnData onData) {
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
  template<typename OnData>
  static inline void forEachFast(DeviceTensor3<DType>& in_data,
                                 DeviceTensor3<DType>& out_data,
                                 OnData onData) {
    const size_t num        = in_data.BatchSize();
    const size_t channels   = in_data.ChannelCount();
    const size_t matrixSize = in_data.SpatialSize();

    for (size_t batchItem = 0; batchItem < num; ++batchItem) {
      #pragma omp parallel for
      for (size_t channel = 0; channel < channels; ++channel) {
        size_t indices[2] = { batchItem, channel };
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
  static inline void computeMean(DeviceTensor3<DType>& tensor, DType *save_mean) {
    const size_t channelCount = tensor.ChannelCount();

    for (size_t i = 0; i < channelCount; ++i) {
      save_mean[i] = 0;
    }

    forEachFast(tensor,
                [&save_mean](const size_t channel, const DType *in_data){
                  save_mean[channel] += *in_data;
                });

    const size_t itemCount = tensor.Size() / channelCount;
    for (size_t i = 0, n = channelCount; i < n; ++i) {
      save_mean[i] /= itemCount;
    }
  }

  static inline bool isWriting(const OpReqType ort) {
    return ort == kWriteTo || ort == kWriteInplace;
  }

  /*! \brief inverse standard deviation <-> variance */
  #define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
  #define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

  /*! \brief Compute the variance of each input channel, as well as update moving mean/variants */
  template<typename  Shape>
  inline void computeVariance(DeviceTensor3<DType>& tensor,
                              const DType *mean_data,
                              const DType eps,
                              const DType momentum,
                              const Shape &oshape,
                              DType *save_std) {
    const size_t channels   = tensor.ChannelCount();
    for (size_t i = 0; i < channels; ++i) {
      save_std[i] = 0;
    }
    forEachFast(tensor,
                [&save_std, &mean_data](const index_t channel, const DType *current_in_data) {
                  const DType mean = mean_data[channel];
                  save_std[channel] += (*current_in_data - mean) * (*current_in_data - mean);
                });

    const size_t itemCount = tensor.Size() / channels;
    #pragma omp parallel for
    for (size_t channel = 0; channel < channels; ++channel) {
      const DType sum = save_std[channel];

      DType invstd;
      if (sum == 0 && eps == 0.0) {
        // Nobody likes to divide by zero
        invstd = 0;
      } else {
        const DType variance = sum/itemCount;
        invstd = VARIANCE_TO_INVSTD(variance, eps);
      }
      save_std[channel] = invstd;
    }
  }

 public:
  explicit BatchNormOp(BatchNormParam param) {
    this->param_ = param;
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
#if MXNET_USE_CUDA
  void DoForward(mshadow::Stream<gpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA

  /*! \brief Forward CPU */
  void DoForward(mshadow::Stream<cpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states) {
    // Input
    DeviceTensor3<DType> inputData(in_data[batchnorm::kData], param_.channel_position);
    const TBlob &weights         = in_data[batchnorm::kGamma];
    const TBlob &bias            = in_data[batchnorm::kBeta];

    // Aux (Moving)
    const TBlob &runningMean     = aux_states[batchnorm::kMovingMean];
    const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

    // Output
    DeviceTensor3<DType> outputData(out_data[batchnorm::kOut], param_.channel_position);
    const TBlob &meanVector      = out_data[batchnorm::kMean];
    const TBlob &varianceVector  = out_data[batchnorm::kVar];

    if (ctx.is_train && !param_.use_global_stats) {
      const TShape stride(2);

      // compute mean per input
      computeMean(inputData, meanVector.dptr<DType>());

      // compute variance per input
      computeVariance(inputData,
                      meanVector.dptr<DType>(),
                      param_.eps,
                      param_.momentum,
                      varianceVector.shape_,
                      varianceVector.dptr<DType>());
    } else {
      DType *m = meanVector.dptr<DType>();
      DType *v = varianceVector.dptr<DType>();
      const DType *rm = runningMean.dptr<DType>();
      const DType *rv = runningVariance.dptr<DType>();

      for (size_t i = 0, n = inputData.shape_[param_.channel_position]; i < n; ++i) {
        m[i] = rm[i];
        v[i] = rv[i];
      }
    }

    // compute output
    DType          *w = weights.dptr<DType>();
    const DType    *b = bias.dptr<DType>();
    const DType *mean = meanVector.dptr<DType>();
    DType  *var = varianceVector.dptr<DType>();

    // optionally, keep weights fixed at 1
    if (param_.fix_gamma) {
      for (size_t i =0, n = weights.Size(); i < n; ++i) {
        w[i] = DType(1);
      }
    }

    if (req[batchnorm::kData] == kWriteTo || req[batchnorm::kData] == kWriteInplace) {
      forEachFast(inputData, outputData,
                  [w, b, mean, var](const size_t channel, const DType *in_data, DType *out_data) {
                    *out_data = static_cast<DType>(
                      ((*in_data - mean[channel]) * var[channel]) * w[channel] + b[channel]);});
    }

    // Convert back to "real" variance in order to be consistent
    // with the original operator
    if (ctx.is_train && !param_.use_global_stats) {
      for (size_t i = 0, n = inputData.shape_[param_.channel_position]; i < n; ++i) {
        var[i] = INVSTD_TO_VARIANCE(var[i], param_.eps);
      }
    }
  }

  /*! \brief Forward pass */
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
#if MXNET_USE_CUDA
  void DoBackward(mshadow::Stream<gpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA

  void DoBackward(mshadow::Stream<cpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states) {
    // Input Data
    DeviceTensor3<DType> inputData(in_data[batchnorm::kData], param_.channel_position);
    const TBlob &weights   = in_data[batchnorm::kGamma];

    // Input Grad
    DeviceTensor3<DType> gradIn(in_grad[batchnorm::kData], param_.channel_position);
    const TBlob &gradWeight = in_grad[batchnorm::kGamma];
    const TBlob &gradBias   = in_grad[batchnorm::kBeta];

    // Aux (Moving)
    const TBlob &runningMean = aux_states[batchnorm::kMovingMean];
    const TBlob &runningVariance = aux_states[batchnorm::kMovingVar];

    // Output
    DeviceTensor3<DType> gradOut(out_grad[batchnorm::kOut], param_.channel_position);
    const TBlob &saveMean = out_data[batchnorm::kMean];
    const TBlob &saveStd  = out_data[batchnorm::kVar];

    const size_t channelCount = inputData.shape_[param_.channel_position];
    const size_t itemCount    = inputData.Size() / channelCount;

    // Avoid multiple dptr() call within the channel loop
    DType *runningMeanDataPtr = runningMean.dptr<DType>();
    DType *runningVarDataPtr  = runningVariance.dptr<DType>();
    DType *saveMeanDataPtr = saveMean.dptr<DType>();
    DType *saveVarianceDataPtr = saveStd.dptr<DType>();
    DType *gradWeightData = gradWeight.dptr<DType>();
    DType *gradBiasData = gradBias.dptr<DType>();

    #pragma omp parallel for
    for (size_t channel = 0; channel < channelCount; ++channel) {
      DType *weight = weights.dptr<DType>();
      const DType w = weight ? weight[channel] : DType(1);
      DType mean, invstd;
      if (ctx.is_train) {
        mean = saveMeanDataPtr[channel];
        const DType variance = saveVarianceDataPtr[channel];
        invstd = VARIANCE_TO_INVSTD(variance, param_.eps);

        // update running averages
        runningMeanDataPtr[channel] = runningMeanDataPtr[channel] * param_.momentum
                                      + mean * (DType(1) - param_.momentum);

        runningVarDataPtr[channel] = runningVarDataPtr[channel] * param_.momentum
                                     + variance * (DType(1) - param_.momentum);

      } else {
        mean = runningMeanDataPtr[channel];
        invstd = VARIANCE_TO_INVSTD(runningVarDataPtr[channel], param_.eps);
      }

      // sumGradOut over all gradOutput in feature plane
      DType sumGradOut = 0;
      forEachFast(gradOut, channel,
                  [&sumGradOut](const DType *gradOut_data) {
                    sumGradOut += *gradOut_data;
                  });

      // dot product of the Q(X) and gradOuput
      DType dotp = 0;
      forEachFast(inputData, gradOut, channel,
                  [&dotp, mean](const DType *thisInputData, const DType *gradOut_data) {
                    dotp += (*thisInputData - mean) * (*gradOut_data);
                  });

      if (gradIn.shape_.ndim()) {  // if there's a grad input
        if (ctx.is_train) {
          // when in training mode
          // Q(X) = X - E[x] ; i.e. input centered to zero mean
          // Y = Q(X) / σ    ; i.e. BN output before weight and bias
          // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

          // projection of gradOutput on to output scaled by std
          const DType k = dotp * invstd * invstd / itemCount;
          forEachFast(inputData, gradIn, channel,
                      [&mean, &k](const DType *in_data, DType *gradIn_data) {
                        *gradIn_data = (*in_data - mean) * k;
                      });

          const DType iw = invstd * w;
          const DType gradMean = sumGradOut / itemCount;
          forEachFast(gradOut, gradIn, channel,
                      [iw, gradMean](const DType *gradOut_data, DType *gradIn_data) {
                        *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * iw;
                      });
        } else {
          // when in evaluation mode
          // Q(X) = X - running_mean  ; i.e. input centered to zero mean
          // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
          // dL/dX = w / running_std
          const DType iw = invstd * w;
          forEachFast(gradOut, gradIn, channel,
                      [iw](const DType *gradOut_data, DType *gradIn_data) {
                        *gradIn_data = *gradOut_data  * iw;
                      });
        }
      }

      // May want to make this a param eventually
      const DType scale = 1.0;

      if (isWriting(req[batchnorm::kGamma])) {
        if (!param_.fix_gamma) {
          gradWeightData[channel] = gradWeightData[channel] + scale * dotp * invstd;
        } else {
          gradWeightData[channel] = DType(0);
        }
      }

      if (isWriting(req[batchnorm::kBeta])) {
        gradBiasData[channel] = gradBiasData[channel] + scale * sumGradOut;
      }
    }
  }

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
  /*! \brief Batch normalization operator parameters */
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(BatchNormParam param, int dtype);

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

    if (dshape.ndim() == 0) {
      return false;
    }

    CHECK_LT(param_.channel_position, dshape.ndim())
      << "Invalid channel position " << param_.channel_position
      << " for a " << dshape.ndim() << "-dimensional shape";

    in_shape->at(1) = TShape(Shape1(dshape[param_.channel_position]));
    in_shape->at(2) = TShape(Shape1(dshape[param_.channel_position]));

    out_shape->clear();
    out_shape->push_back(dshape);             // kOut
    out_shape->push_back(Shape1(dshape[param_.channel_position]));  // kMean
    out_shape->push_back(Shape1(dshape[param_.channel_position]));  // kVar

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[param_.channel_position]));  // kMovingMean
    aux_shape->push_back(Shape1(dshape[param_.channel_position]));  // kMovingVar
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
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        CHECK_EQ((*in_type)[i], dtype_param) << "This layer requires uniform type. "
                                             << "Expected " << dtype_param << " v.s. given "
                                             << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        CHECK_EQ((*aux_type)[i], dtype_param) << "This layer requires uniform type. "
                                              << "Expected " << dtype_param << " v.s. given "
                                              << (*aux_type)[i] << " at " << ListArguments()[i];
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

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_


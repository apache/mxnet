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
* \file image_random-inl.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_


#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include "mxnet/base.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace op {
namespace image {

using namespace mshadow;

#if MXNET_USE_CUDA
// NOTE: Kernel launch/map was extremely costly.
// Hence, we use separate CUDA kernels for these operators.
template<typename DType, typename T1, typename T2>
void ToTensorImplCUDA(mshadow::Stream<gpu> *s,
                      const T1 input,
                      const T2 output,
                      const int req,
                      const float normalize_factor);

template<typename DType>
void NormalizeImplCUDA(mshadow::Stream<gpu> *s,
                       const DType *input,
                       DType *output,
                       const int req,
                       const int N,
                       const int C,
                       const int H,
                       const int W,
                       const float mean_d0,
                       const float mean_d1,
                       const float mean_d2,
                       const float std_d0,
                       const float std_d1,
                       const float std_d2);

template<typename DType>
void NormalizeBackwardImplCUDA(mshadow::Stream<gpu> *s,
                               const DType *out_grad,
                               DType *in_grad,
                               const int req,
                               const int N,
                               const int C,
                               const int H,
                               const int W,
                               const float std_d0,
                               const float std_d1,
                               const float std_d2);
#endif  // MXNET_USE_CUDA

// Shape and Type inference for image to tensor operator
inline bool ToTensorShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  mxnet::TShape &shp = (*in_attrs)[0];
  if (!shape_is_known(shp)) return false;

  CHECK((shp.ndim() == 3) || (shp.ndim() == 4))
      << "Input image must have shape (height, width, channels), or "
      << "(N, height, width, channels) but got " << shp;
  if (shp.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({shp[2], shp[0], shp[1]}));
  } else if (shp.ndim() == 4) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({shp[0], shp[3], shp[1], shp[2]}));
  }

  return true;
}

inline bool ToTensorType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

// Operator Implementation
template<typename DType, int req>
inline void ToTensor(float* out_data, const DType* in_data,
                     const int length,
                     const int channels,
                     const float normalize_factor,
                     const int step) {
  // Microsoft Visual C++ compiler does not support omp collapse
  #ifdef _MSC_VER
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif  // _MSC_VER
  for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < length; ++i) {
        KERNEL_ASSIGN(out_data[step + c*length + i], req,
                      (in_data[step + i*channels + c]) / normalize_factor);
      }
  }
}

inline void ToTensorImpl(const std::vector<TBlob> &inputs,
                         const std::vector<TBlob> &outputs,
                         const std::vector<OpReqType> &req,
                         const int length,
                         const int channel,
                         const float normalize_factor,
                         const int step) {
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      float* output = outputs[0].dptr<float>();
      DType* input = inputs[0].dptr<DType>();
      ToTensor<DType, req_type>(output, input, length, channel,
                                normalize_factor, step);
    });
  });
}

template<typename xpu>
void ToTensorOpForward(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  // We do not use temp buffer when performance the operation.
  // Hence, this check is necessary.
  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace updates";

  const float normalize_factor = 255.0f;

  if (std::is_same<xpu, gpu>::value) {
  #if MXNET_USE_CUDA
      mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          if (inputs[0].ndim() == 3) {
            Tensor<gpu, 3, DType> input = inputs[0].get<gpu, 3, DType>(s);
            Tensor<gpu, 3, float> output = outputs[0].get<gpu, 3, float>(s);
            ToTensorImplCUDA<DType, Tensor<gpu, 3, DType>, Tensor<gpu, 3, float>>
            (s, input, output, req_type, normalize_factor);
          } else {
            Tensor<gpu, 4, DType> input = inputs[0].get<gpu, 4, DType>(s);
            Tensor<gpu, 4, float> output = outputs[0].get<gpu, 4, float>(s);
            ToTensorImplCUDA<DType, Tensor<gpu, 4, DType>, Tensor<gpu, 4, float>>
            (s, input, output, req_type, normalize_factor);
          }
        });
      });
  #else
    LOG(FATAL) << "Compile with USE_CUDA=1 to use ToTensor operator on GPU.";
  #endif  // MXNET_USE_CUDA
  } else if (inputs[0].ndim() == 3) {
    // 3D Input - (h, w, c)
    const int length = inputs[0].shape_[0] * inputs[0].shape_[1];
    const int channel = static_cast<int>(inputs[0].shape_[2]);
    const int step = 0;
    ToTensorImpl(inputs, outputs, req, length,
                 channel, normalize_factor, step);
  } else if (inputs[0].ndim() == 4) {
    // 4D input (n, h, w, c)
    const int batch_size = inputs[0].shape_[0];
    const int length = inputs[0].shape_[1] * inputs[0].shape_[2];
    const int channel = static_cast<int>(inputs[0].shape_[3]);
    const int step = channel * length;

    #pragma omp parallel for
    for (auto n = 0; n < batch_size; ++n) {
      ToTensorImpl(inputs, outputs, req, length, channel,
                   normalize_factor, n*step);
    }
  }
}

struct NormalizeParam : public dmlc::Parameter<NormalizeParam> {
  mxnet::Tuple<float> mean;
  mxnet::Tuple<float> std;

  DMLC_DECLARE_PARAMETER(NormalizeParam) {
    DMLC_DECLARE_FIELD(mean)
    .set_default(mxnet::Tuple<float> {0.0f, 0.0f, 0.0f, 0.0f})
    .describe("Sequence of means for each channel. "
              "Default value is 0.");
    DMLC_DECLARE_FIELD(std)
    .set_default(mxnet::Tuple<float> {1.0f, 1.0f, 1.0f, 1.0f})
    .describe("Sequence of standard deviations for each channel. "
              "Default value is 1.");
  }
};

// Shape and Type inference for image Normalize operator

// Shape inference
inline bool NormalizeOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  const auto& dshape = (*in_attrs)[0];
  if (!dshape.ndim()) return false;

  CHECK((dshape.ndim() == 3) || (dshape.ndim() == 4))
      << "Input tensor must have shape (channels, height, width), or "
      << "(N, channels, height, width), but got " << dshape;

  int nchannels = 0;
  if (dshape.ndim() == 3) {
    nchannels = dshape[0];
    CHECK(nchannels == 3 || nchannels == 1)
      << "The first dimension of input tensor must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  } else if (dshape.ndim() == 4) {
    nchannels = dshape[1];
    CHECK(nchannels == 3 || nchannels == 1)
      << "The second dimension of input tensor must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  }

  CHECK((param.mean.ndim() == 1) || (param.mean.ndim() == nchannels))
      << "Invalid mean for input with shape " << dshape
      << ". mean must have either 1 or " << nchannels
      << " elements, but got " << param.mean;
  CHECK(param.std.ndim() == 1 || param.std.ndim() == nchannels)
      << "Invalid std for input with shape " << dshape
      << ". std must have either 1 or " << nchannels
      << " elements, but got " << param.std;

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

// Type Inference
inline bool NormalizeOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

template<typename DType, int req>
inline void Normalize(DType* out_data,
                      const DType* in_data,
                      const int length,
                      const int channels,
                      const int step,
                      const std::vector<float> mean,
                      const std::vector<float> std) {
  // Microsoft Visual C++ compiler does not support omp collapse
  #ifdef _MSC_VER
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif  // _MSC_VER
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < length; ++i) {
      KERNEL_ASSIGN(out_data[step + c*length + i], req,
                    (in_data[step + c*length + i] - mean[c]) / std[c]);
    }
  }
}

inline void NormalizeImpl(const std::vector<TBlob> &inputs,
                          const std::vector<TBlob> &outputs,
                          const std::vector<OpReqType> &req,
                          const int length,
                          const int channels,
                          const int step,
                          const std::vector<float> mean,
                          const std::vector<float> std) {
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      DType* input = inputs[0].dptr<DType>();
      DType* output = outputs[0].dptr<DType>();
      Normalize<DType, req_type>(output, input, length, channels, step,
                                 mean, std);
    });
  });
}

template<typename xpu>
void NormalizeOpForward(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  // Mean and Std can be 1 or 3D only.
  std::vector<float> mean(3);
  std::vector<float> std(3);
  if (param.mean.ndim() == 1) {
    mean[0] = mean[1] = mean[2] = param.mean[0];
  } else {
    mean[0] = param.mean[0];
    mean[1] = param.mean[1];
    mean[2] = param.mean[2];
  }

  if (param.std.ndim() == 1) {
    std[0] = std[1] = std[2] = param.std[0];
  } else {
    std[0] = param.std[0];
    std[1] = param.std[1];
    std[2] = param.std[2];
  }

  if (std::is_same<xpu, gpu>::value) {
    #if MXNET_USE_CUDA
      mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          int N, C, H, W;
          DType *input = nullptr;
          DType *output = nullptr;
          if (inputs[0].ndim() == 3) {
            N = 1;
            C = static_cast<int>(inputs[0].shape_[0]);
            H = static_cast<int>(inputs[0].shape_[1]);
            W = static_cast<int>(inputs[0].shape_[2]);
            input = (inputs[0].get<gpu, 3, DType>(s)).dptr_;
            output = (outputs[0].get<gpu, 3, DType>(s)).dptr_;
          } else {
            N = static_cast<int>(inputs[0].shape_[0]);
            C = static_cast<int>(inputs[0].shape_[1]);
            H = static_cast<int>(inputs[0].shape_[2]);
            W = static_cast<int>(inputs[0].shape_[3]);
            input = (inputs[0].get<gpu, 4, DType>(s)).dptr_;
            output = (outputs[0].get<gpu, 4, DType>(s)).dptr_;
          }
          NormalizeImplCUDA<DType>(s, input, output, req_type,
                                   N, C, H, W,
                                   mean[0], mean[1], mean[2],
                                   std[0], std[1], std[2]);
        });
      });
    #else
      LOG(FATAL) << "Compile with USE_CUDA=1 to use Normalize operator on GPU.";
    #endif  // MXNET_USE_CUDA
  } else if (inputs[0].ndim() == 3) {
    // 3D input (c, h, w)
    const int length = inputs[0].shape_[1] * inputs[0].shape_[2];
    const int channel = static_cast<int>(inputs[0].shape_[0]);
    const int step = 0;
    NormalizeImpl(inputs, outputs, req, length, channel, step, mean, std);
  } else if (inputs[0].ndim() == 4) {
    // 4D input (n, c, h, w)
    const int batch_size = inputs[0].shape_[0];
    const int length = inputs[0].shape_[2] * inputs[0].shape_[3];
    const int channel = static_cast<int>(inputs[0].shape_[1]);
    const int step = channel * length;

    #pragma omp parallel for
    for (auto n = 0; n < batch_size; ++n) {
      NormalizeImpl(inputs, outputs, req, length, channel, n*step, mean, std);
    }
  }
}

// Backward function
template<typename DType, int req>
inline void NormalizeBackward(const DType* out_grad,
                              DType* in_grad,
                              const int length,
                              const int channels,
                              const int step,
                              const std::vector<float> std) {
  // Microsoft Visual C++ compiler does not support omp collapse
  #ifdef _MSC_VER
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif  // _MSC_VER
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < length; ++i) {
      KERNEL_ASSIGN(in_grad[step + c*length + i], req,
                    out_grad[step + c*length + i] * (1.0 / std[c]));
    }
  }
}

inline void NormalizeBackwardImpl(const std::vector<TBlob> &inputs,
                                  const std::vector<TBlob> &outputs,
                                  const std::vector<OpReqType> &req,
                                  const int length,
                                  const int channels,
                                  const int step,
                                  const std::vector<float> std
                                  ) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        DType* out_grad = inputs[0].dptr<DType>();
        DType* in_grad = outputs[0].dptr<DType>();
        NormalizeBackward<DType, req_type>(out_grad, in_grad, length,
                                           channels, step, std);
      });
    });
}

template<typename xpu>
void NormalizeOpBackward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);
  // Std can be 1 or 3D only.
  std::vector<float> std(3);
  if (param.std.ndim() == 1) {
    std[0] = std[1] = std[2] = param.std[0];
  } else {
    std[0] = param.std[0];
    std[1] = param.std[1];
    std[2] = param.std[2];
  }

  // Note: inputs[0] is out_grad
  const TBlob& in_data = inputs[1];

  if (std::is_same<xpu, gpu>::value) {
    #if MXNET_USE_CUDA
      mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          int N, C, H, W;
          DType *in_grad = nullptr;
          DType *out_grad = nullptr;
          if (in_data.ndim() == 3) {
            N = 1;
            C = static_cast<int>(in_data.shape_[0]);
            H = static_cast<int>(in_data.shape_[1]);
            W = static_cast<int>(in_data.shape_[2]);
            out_grad = (inputs[0].get<gpu, 3, DType>(s)).dptr_;
            in_grad = (outputs[0].get<gpu, 3, DType>(s)).dptr_;
          } else {
            N = static_cast<int>(in_data.shape_[0]);
            C = static_cast<int>(in_data.shape_[1]);
            H = static_cast<int>(in_data.shape_[2]);
            W = static_cast<int>(in_data.shape_[3]);
            out_grad = (inputs[0].get<gpu, 4, DType>(s)).dptr_;
            in_grad = (outputs[0].get<gpu, 4, DType>(s)).dptr_;
          }
          NormalizeBackwardImplCUDA<DType>(s, out_grad, in_grad, req_type,
                                           N, C, H, W,
                                           std[0], std[1], std[2]);
        });
      });
    #else
      LOG(FATAL) << "Compile with USE_CUDA=1 to use Normalize backward operator on GPU.";
    #endif  // MXNET_USE_CUDA
  } else if (in_data.ndim() == 3) {
    // 3D input (c, h, w)
    const int length = in_data.shape_[1] * in_data.shape_[2];
    const int channel = static_cast<int>(in_data.shape_[0]);
    const int step = 0;
    NormalizeBackwardImpl(inputs, outputs, req, length, channel, step, std);
  } else if (in_data.ndim() == 4) {
    // 4D input (n, c, h, w)
    const int batch_size = in_data.shape_[0];
    const int length = in_data.shape_[2] * in_data.shape_[3];
    const int channel = static_cast<int>(in_data.shape_[1]);
    const int step = channel * length;

    #pragma omp parallel for
    for (auto n = 0; n < batch_size; ++n) {
      NormalizeBackwardImpl(inputs, outputs, req, length, channel, n*step, std);
    }
  }
}

template<typename DType>
inline DType saturate_cast(const float& src) {
  return static_cast<DType>(src);
}

template<>
inline uint8_t saturate_cast(const float& src) {
  return std::min(std::max(src, 0.f), 255.f);
}

inline bool ImageShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector *in_attrs,
                       mxnet::ShapeVector *out_attrs) {
  mxnet::TShape& dshape = (*in_attrs)[0];
  CHECK_EQ(dshape.ndim(), 3)
      << "Input image must have shape (height, width, channels), but got " << dshape;
  auto nchannels = dshape[dshape.ndim()-1];
  CHECK(nchannels == 3 || nchannels == 1)
      << "The last dimension of input image must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

template<typename DType, int axis>
void FlipImpl(const mxnet::TShape &shape, DType *src, DType *dst) {
  int head = 1, mid = shape[axis], tail = 1;
  for (int i = 0; i < axis; ++i) head *= shape[i];
  for (int i = axis+1; i < shape.ndim(); ++i) tail *= shape[i];

  for (int i = 0; i < head; ++i) {
    for (int j = 0; j < (mid >> 1); ++j) {
      int idx1 = (i*mid + j) * tail;
      int idx2 = idx1 + (mid-(j << 1)-1) * tail;
      for (int k = 0; k < tail; ++k, ++idx1, ++idx2) {
        DType tmp = src[idx1];
        dst[idx1] = src[idx2];
        dst[idx2] = tmp;
      }
    }
  }
}

inline void FlipLeftRight(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    FlipImpl<DType, 1>(inputs[0].shape_, inputs[0].dptr<DType>(),
                       outputs[0].dptr<DType>());
  });
}

inline void FlipTopBottom(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    FlipImpl<DType, 0>(inputs[0].shape_, inputs[0].dptr<DType>(),
                       outputs[0].dptr<DType>());
  });
}

inline void RandomFlipLeftRight(
    const nnvm::NodeAttrs &attrs,
    const OpContext &ctx,
    const std::vector<TBlob> &inputs,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (std::bernoulli_distribution()(prnd->GetRndEngine())) {
      if (outputs[0].dptr_ != inputs[0].dptr_) {
        std::memcpy(outputs[0].dptr_, inputs[0].dptr_, inputs[0].Size() * sizeof(DType));
      }
    } else {
      FlipImpl<DType, 1>(inputs[0].shape_, inputs[0].dptr<DType>(),
                         outputs[0].dptr<DType>());
    }
  });
}

inline void RandomFlipTopBottom(
    const nnvm::NodeAttrs &attrs,
    const OpContext &ctx,
    const std::vector<TBlob> &inputs,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (std::bernoulli_distribution()(prnd->GetRndEngine())) {
      if (outputs[0].dptr_ != inputs[0].dptr_) {
        std::memcpy(outputs[0].dptr_, inputs[0].dptr_, inputs[0].Size() * sizeof(DType));
      }
    } else {
      FlipImpl<DType, 0>(inputs[0].shape_, inputs[0].dptr<DType>(),
                         outputs[0].dptr<DType>());
    }
  });
}

struct RandomEnhanceParam : public dmlc::Parameter<RandomEnhanceParam> {
  float min_factor;
  float max_factor;
  DMLC_DECLARE_PARAMETER(RandomEnhanceParam) {
    DMLC_DECLARE_FIELD(min_factor)
    .set_lower_bound(0.0)
    .describe("Minimum factor.");
    DMLC_DECLARE_FIELD(max_factor)
    .set_lower_bound(0.0)
    .describe("Maximum factor.");
  }
};

inline void AdjustBrightnessImpl(const float& alpha_b,
                                 const OpContext &ctx,
                                 const std::vector<TBlob> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  int length = inputs[0].Size();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* output = outputs[0].dptr<DType>();
    DType* input = inputs[0].dptr<DType>();
    for (int l = 0; l < length; ++l) {
      float val = static_cast<float>(input[l]) * alpha_b;
      output[l] = saturate_cast<DType>(val);
    }
  });
}

inline void RandomBrightness(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomEnhanceParam &param = nnvm::get<RandomEnhanceParam>(attrs.parsed);


  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  float alpha_b = std::uniform_real_distribution<float>(
      param.min_factor, param.max_factor)(prnd->GetRndEngine());

  AdjustBrightnessImpl(alpha_b, ctx, inputs, req, outputs);
}

inline void AdjustContrastImpl(const float& alpha_c,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  static const float coef[] = { 0.299f, 0.587f, 0.114f };

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* output = outputs[0].dptr<DType>();
    DType* input = inputs[0].dptr<DType>();

    float sum = 0.f;
    if (nchannels > 1) {
      for (int l = 0; l < length; ++l) {
        for (int c = 0; c < 3; ++c) sum += input[l*3 + c] * coef[c];
      }
    } else {
      for (int l = 0; l < length; ++l) sum += input[l];
    }
    float gray_mean = sum / static_cast<float>(length);
    float beta = (1 - alpha_c) * gray_mean;

    for (int l = 0; l < length * nchannels; ++l) {
      float val = input[l] * alpha_c + beta;
      output[l] = saturate_cast<DType>(val);
    }
  });
}

inline void RandomContrast(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomEnhanceParam &param = nnvm::get<RandomEnhanceParam>(attrs.parsed);


  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_c = std::uniform_real_distribution<float>(
      param.min_factor, param.max_factor)(prnd->GetRndEngine());

  AdjustContrastImpl(alpha_c, ctx, inputs, req, outputs);
}

inline void AdjustSaturationImpl(const float& alpha_s,
                                 const OpContext &ctx,
                                 const std::vector<TBlob> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &outputs) {
  static const float coef[] = { 0.299f, 0.587f, 0.114f };

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  float alpha_o = 1.f - alpha_s;

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* output = outputs[0].dptr<DType>();
    DType* input = inputs[0].dptr<DType>();

    if (nchannels == 1) {
      for (int l = 0; l < length; ++l) output[l] = input[l];
      return;
    }

    for (int l = 0; l < length; ++l) {
      float gray = 0.f;
      for (int c = 0; c < 3; ++c) {
        gray = input[l*3 + c] * coef[c];
      }
      gray *= alpha_o;
      for (int c = 0; c < 3; ++c) {
        float val = gray + input[l*3 + c] * alpha_s;
        output[l*3 + c] = saturate_cast<DType>(val);
      }
    }
  });
}

inline void RandomSaturation(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomEnhanceParam &param = nnvm::get<RandomEnhanceParam>(attrs.parsed);

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_s = std::uniform_real_distribution<float>(
      param.min_factor, param.max_factor)(prnd->GetRndEngine());

  AdjustSaturationImpl(alpha_s, ctx, inputs, req, outputs);
}

inline void RGB2HLSConvert(const float& src_r,
                    const float& src_g,
                    const float& src_b,
                    float *dst_h,
                    float *dst_l,
                    float *dst_s) {
  float b = src_b / 255.f, g = src_g / 255.f, r = src_r / 255.f;
  float h = 0.f, s = 0.f, l;
  float vmin;
  float vmax;
  float diff;

  vmax = vmin = r;
  vmax = std::fmax(vmax, g);
  vmax = std::fmax(vmax, b);
  vmin = std::fmin(vmin, g);
  vmin = std::fmin(vmin, b);

  diff = vmax - vmin;
  l = (vmax + vmin) * 0.5f;

  if (diff > std::numeric_limits<float>::epsilon()) {
    s = (l < 0.5f) * diff / (vmax + vmin);
    s += (l >= 0.5f) * diff / (2.0f - vmax - vmin);

    diff = 60.f / diff;

    h = (vmax == r) * (g - b) * diff;
    h += (vmax != r && vmax == g) * ((b - r) * diff + 120.f);
    h += (vmax != r && vmax != g) * ((r - g) * diff + 240.f);
    h += (h < 0.f) * 360.f;
  }

  *dst_h = h;
  *dst_l = l;
  *dst_s = s;
}

inline void HLS2RGBConvert(const float& src_h,
                    const float& src_l,
                    const float& src_s,
                    float *dst_r,
                    float *dst_g,
                    float *dst_b) {
  static const int c_HlsSectorData[6][3] = {
    { 1, 3, 0 },
    { 1, 0, 2 },
    { 3, 0, 1 },
    { 0, 2, 1 },
    { 0, 1, 3 },
    { 2, 1, 0 }
  };

  float h = src_h, l = src_l, s = src_s;
  float b = l, g = l, r = l;

  if (s != 0) {
    float p2 = (l <= 0.5f) * l * (1 + s);
    p2 += (l > 0.5f) * (l + s - l * s);
    float p1 = 2 * l - p2;

    h *= 1.f / 60.f;

    if (h < 0) {
      do { h += 6; } while (h < 0);
    } else if (h >= 6) {
      do { h -= 6; } while (h >= 6);
    }

    int sector = static_cast<int>(h);

    h -= sector;

    float tab[4];
    tab[0] = p2;
    tab[1] = p1;
    tab[2] = p1 + (p2 - p1) * (1 - h);
    tab[3] = p1 + (p2 - p1) * h;

    b = tab[c_HlsSectorData[sector][0]];
    g = tab[c_HlsSectorData[sector][1]];
    r = tab[c_HlsSectorData[sector][2]];
  }

  *dst_b = b * 255.f;
  *dst_g = g * 255.f;
  *dst_r = r * 255.f;
}

inline void AdjustHueImpl(float alpha,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  if (inputs[0].shape_[2] == 1) return;

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* input = inputs[0].dptr<DType>();
    DType* output = outputs[0].dptr<DType>();

    for (int i = 0; i < length; ++i) {
      float h, l, s;
      float r = static_cast<float>(*(input++));
      float g = static_cast<float>(*(input++));
      float b = static_cast<float>(*(input++));
      RGB2HLSConvert(r, g, b, &h, &l, &s);
      h += alpha * 360.f;
      HLS2RGBConvert(h, l, s, &r, &g, &b);
      *(output++) = saturate_cast<DType>(r);
      *(output++) = saturate_cast<DType>(g);
      *(output++) = saturate_cast<DType>(b);
    }
  });
}

inline void RandomHue(const nnvm::NodeAttrs &attrs,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomEnhanceParam &param = nnvm::get<RandomEnhanceParam>(attrs.parsed);

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha = std::uniform_real_distribution<float>(
      param.min_factor, param.max_factor)(prnd->GetRndEngine());

  AdjustHueImpl(alpha, ctx, inputs, req, outputs);
}

struct RandomColorJitterParam : public dmlc::Parameter<RandomColorJitterParam> {
  float brightness;
  float contrast;
  float saturation;
  float hue;
  DMLC_DECLARE_PARAMETER(RandomColorJitterParam) {
    DMLC_DECLARE_FIELD(brightness)
    .describe("How much to jitter brightness.");
    DMLC_DECLARE_FIELD(contrast)
    .describe("How much to jitter contrast.");
    DMLC_DECLARE_FIELD(saturation)
    .describe("How much to jitter saturation.");
    DMLC_DECLARE_FIELD(hue)
    .describe("How much to jitter hue.");
  }
};

inline void RandomColorJitter(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomColorJitterParam &param = nnvm::get<RandomColorJitterParam>(attrs.parsed);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);

  int order[4] = {0, 1, 2, 3};
  std::shuffle(order, order + 4, prnd->GetRndEngine());
  bool flag = false;

  for (int i = 0; i < 4; ++i) {
    switch (order[i]) {
      case 0:
        if (param.brightness > 0) {
          float alpha_b = 1.0 + std::uniform_real_distribution<float>(
              -param.brightness, param.brightness)(prnd->GetRndEngine());
          AdjustBrightnessImpl(alpha_b, ctx, flag ? outputs : inputs, req, outputs);
          flag = true;
        }
        break;
      case 1:
        if (param.contrast > 0) {
          float alpha_c = 1.0 + std::uniform_real_distribution<float>(
              -param.contrast, param.contrast)(prnd->GetRndEngine());
          AdjustContrastImpl(alpha_c, ctx, flag ? outputs : inputs, req, outputs);
          flag = true;
        }
        break;
      case 2:
        if (param.saturation > 0) {
          float alpha_s = 1.f + std::uniform_real_distribution<float>(
              -param.saturation, param.saturation)(prnd->GetRndEngine());
          AdjustSaturationImpl(alpha_s, ctx, flag ? outputs : inputs, req, outputs);
          flag = true;
        }
        break;
      case 3:
        if (param.hue > 0) {
          float alpha_h = std::uniform_real_distribution<float>(
              -param.hue, param.hue)(prnd->GetRndEngine());
          AdjustHueImpl(alpha_h, ctx, flag ? outputs : inputs, req, outputs);
          flag = true;
        }
        break;
    }
  }
}

struct AdjustLightingParam : public dmlc::Parameter<AdjustLightingParam> {
  mxnet::Tuple<float> alpha;
  DMLC_DECLARE_PARAMETER(AdjustLightingParam) {
    DMLC_DECLARE_FIELD(alpha)
    .describe("The lighting alphas for the R, G, B channels.");
  }
};

struct RandomLightingParam : public dmlc::Parameter<RandomLightingParam> {
  float alpha_std;
  DMLC_DECLARE_PARAMETER(RandomLightingParam) {
    DMLC_DECLARE_FIELD(alpha_std)
    .set_default(0.05)
    .describe("Level of the lighting noise.");
  }
};

inline void AdjustLightingImpl(const mxnet::Tuple<float>& alpha,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  static const float eig[3][3] = {
      { 55.46 * -0.5675, 4.794 * 0.7192,  1.148 * 0.4009 },
      { 55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140 },
      { 55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203 }
    };

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int channels = inputs[0].shape_[2];
  if (channels == 1) return;

  float pca_r = eig[0][0] * alpha[0] + eig[0][1] * alpha[1] + eig[0][2] * alpha[2];
  float pca_g = eig[1][0] * alpha[0] + eig[1][1] * alpha[1] + eig[1][2] * alpha[2];
  float pca_b = eig[2][0] * alpha[0] + eig[2][1] * alpha[1] + eig[2][2] * alpha[2];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* output = outputs[0].dptr<DType>();
    DType* input = inputs[0].dptr<DType>();

    for (int i = 0; i < length; i++) {
      int base_ind = 3 * i;
      float in_r = static_cast<float>(input[base_ind]);
      float in_g = static_cast<float>(input[base_ind + 1]);
      float in_b = static_cast<float>(input[base_ind + 2]);
      output[base_ind] = saturate_cast<DType>(in_r + pca_r);
      output[base_ind + 1] = saturate_cast<DType>(in_g + pca_g);
      output[base_ind + 2] = saturate_cast<DType>(in_b + pca_b);
    }
  });
}

inline void AdjustLighting(const nnvm::NodeAttrs &attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const AdjustLightingParam &param = nnvm::get<AdjustLightingParam>(attrs.parsed);
  AdjustLightingImpl(param.alpha, ctx, inputs, req, outputs);
}

inline void RandomLighting(const nnvm::NodeAttrs &attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomLightingParam &param = nnvm::get<RandomLightingParam>(attrs.parsed);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  std::normal_distribution<float> dist(0, param.alpha_std);
  float alpha_r = dist(prnd->GetRndEngine());
  float alpha_g = dist(prnd->GetRndEngine());
  float alpha_b = dist(prnd->GetRndEngine());
  AdjustLightingImpl({alpha_r, alpha_g, alpha_b}, ctx, inputs, req, outputs);
}


#define MXNET_REGISTER_IMAGE_AUG_OP(name)                                   \
  NNVM_REGISTER_OP(name)                                                    \
  .set_num_inputs(1)                                                        \
  .set_num_outputs(1)                                                       \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                         \
    [](const NodeAttrs& attrs){                                             \
      return std::vector<std::pair<int, int> >{{0, 0}};                     \
    })                                                                      \
  .set_attr<mxnet::FInferShape>("FInferShape", ImageShape)                   \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)             \
  .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })   \
  .add_argument("data", "NDArray-or-Symbol", "The input.")


#define MXNET_REGISTER_IMAGE_RND_AUG_OP(name)                               \
  MXNET_REGISTER_IMAGE_AUG_OP(name)                                         \
  .set_attr<FResourceRequest>("FResourceRequest",                           \
    [](const NodeAttrs& attrs) {                                            \
      return std::vector<ResourceRequest>{ResourceRequest::kRandom};        \
    })

}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_

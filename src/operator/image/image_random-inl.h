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

inline bool ToTensorShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape &shp = (*in_attrs)[0];
  if (!shp.ndim()) return false;
  CHECK_EQ(shp.ndim(), 3)
      << "Input image must have shape (height, width, channels), but got " << shp;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({shp[2], shp[0], shp[1]}));
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

void ToTensor(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace";

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int channel = inputs[0].shape_[2];

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    float* output = outputs[0].dptr<float>();
    DType* input = inputs[0].dptr<DType>();

    for (int l = 0; l < length; ++l) {
      for (int c = 0; c < channel; ++c) {
        output[c*length + l] = static_cast<float>(input[l*channel + c]) / 255.0f;
      }
    }
  });
}

struct NormalizeParam : public dmlc::Parameter<NormalizeParam> {
  nnvm::Tuple<float> mean;
  nnvm::Tuple<float> std;
  DMLC_DECLARE_PARAMETER(NormalizeParam) {
    DMLC_DECLARE_FIELD(mean)
    .describe("Sequence of mean for each channel.");
    DMLC_DECLARE_FIELD(std)
    .describe("Sequence of standard deviations for each channel.");
  }
};

inline bool NormalizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);
  const auto& dshape = (*in_attrs)[0];
  if (!dshape.ndim()) return false;

  CHECK_EQ(dshape.ndim(), 3)
      << "Input tensor must have shape (channels, height, width), but got "
      << dshape;
  auto nchannels = dshape[0];
  CHECK(nchannels == 3 || nchannels == 1)
      << "The first dimension of input tensor must be the channel dimension with "
      << "either 1 or 3 elements, but got input with shape " << dshape;
  CHECK(param.mean.ndim() == 1 || param.mean.ndim() == nchannels)
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

void Normalize(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  int nchannels = inputs[0].shape_[0];
  int length = inputs[0].shape_[1] * inputs[0].shape_[2];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* input = inputs[0].dptr<DType>();
    DType* output = outputs[0].dptr<DType>();

    for (int i = 0; i < nchannels; ++i) {
      DType mean = param.mean[param.mean.ndim() > 1 ? i : 0];
      DType std = param.std[param.std.ndim() > 1 ? i : 0];
      for (int j = 0; j < length; ++j) {
        output[i*length + j] = (input[i*length + j] - mean) / std;
      }
    }
  });
}

struct ResizeParam : public dmlc::Parameter<ResizeParam> {
  nnvm::Tuple<int> size;
  bool keep_ratio;
  int interp;
  DMLC_DECLARE_PARAMETER(ResizeParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(nnvm::Tuple<int>())
    .describe("Size of new image. Could be (width, height) or (size)");
    DMLC_DECLARE_FIELD(keep_ratio)
    .describe("Whether to resize the short edge or both edges to `size`, "
      "if size is give as an integer.");
    DMLC_DECLARE_FIELD(interp)
    .set_default(1)
    .describe("Interpolation method for resizing. By default uses bilinear"
        "interpolation. See OpenCV's resize function for available choices.");
  }
};

inline std::tuple<int, int> GetHeightAndWidth(int data_h,
                                              int data_w,
                                              const ResizeParam& param) {
  CHECK_LE(param.size.ndim(), 2)
      << "Input size dimension must be 1 or 2, but got "
      << param.size.ndim();
  int resized_h;
  int resized_w;
  if (param.size.ndim() == 1) {
    CHECK_GT(param.size[0], 0)
      << "Input size should greater than 0, but got "
      << param.size[0];
    if (!param.keep_ratio) {
      resized_h = param.size[0];
      resized_w = param.size[0];
    } else {
      if (data_h > data_w) {
        resized_w = param.size[0];
        resized_h = static_cast<int>(data_h * resized_w / data_w);
      } else {
        resized_h = param.size[0];
        resized_w = static_cast<int>(data_w * resized_h / data_h);
      }
    }
  } else {
    CHECK_GT(param.size[0], 0)
      << "Input width should greater than 0, but got "
      << param.size[0];
    CHECK_GT(param.size[1], 0)
      << "Input height should greater than 0, but got "
      << param.size[1];
    resized_h = param.size[1];
    resized_w = param.size[0];
  }

  return std::make_tuple(resized_h, resized_w);
}

bool ResizeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  std::tuple<int, int> t;
  if (ishape.ndim() == 3) {
    t = GetHeightAndWidth(ishape[0], ishape[1], param);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({std::get<0>(t), std::get<1>(t), ishape[2]}));
  } else {
    t = GetHeightAndWidth(ishape[1], ishape[2], param);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0,
      TShape({ishape[0], std::get<0>(t), std::get<1>(t), ishape[3]}));
  }
  return true;
}

void ResizeImpl(const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const int height,
                      const int width,
                      const int interp,
                      const int input_index = 0,
                      const int output_index = 0) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "image resize doesn't support fp16";
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  if (inputs[0].ndim() == 3) {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[2]);
    cv::Mat buf(inputs[0].shape_[0], inputs[0].shape_[1], cv_type, inputs[0].dptr_);
    cv::Mat dst(outputs[0].shape_[0], outputs[0].shape_[1], cv_type, outputs[0].dptr_);
    cv::resize(buf, dst, cv::Size(width, height), 0, 0, interp);
    CHECK(!dst.empty());
    CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
  } else {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[3]);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      cv::Mat buf(inputs[0].shape_[1], inputs[0].shape_[2], cv_type,
        inputs[0].dptr<DType>() + input_index);
      cv::Mat dst(outputs[0].shape_[1], outputs[0].shape_[2], cv_type,
        outputs[0].dptr<DType>() + output_index);
      cv::resize(buf, dst, cv::Size(width, height), 0, 0, interp);
      CHECK(!dst.empty());
      CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr<DType>() + output_index);
    });
  }
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image resize operator.";
#endif  // MXNET_USE_OPENCV
}

void Resize(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  std::tuple<int, int> t;
  if (inputs[0].ndim() == 3) {
    t = GetHeightAndWidth(inputs[0].shape_[0], inputs[0].shape_[1], param);
    ResizeImpl(inputs, outputs, std::get<0>(t), std::get<1>(t), param.interp);
  } else {
    t = GetHeightAndWidth(inputs[0].shape_[1], inputs[0].shape_[2], param);
    const auto batch_size = inputs[0].shape_[0];
    const auto input_step = inputs[0].shape_[1] * inputs[0].shape_[2] * inputs[0].shape_[3];
    const auto output_step = std::get<0>(t) * std::get<1>(t) * inputs[0].shape_[3];
    #pragma omp parallel for
    for (auto i = 0; i < batch_size; ++i) {
      ResizeImpl(inputs, outputs, std::get<0>(t), std::get<1>(t),
        param.interp, i * input_step, i * output_step);
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
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  TShape& dshape = (*in_attrs)[0];
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
void FlipImpl(const TShape &shape, DType *src, DType *dst) {
  int head = 1, mid = shape[axis], tail = 1;
  for (int i = 0; i < axis; ++i) head *= shape[i];
  for (uint32_t i = axis+1; i < shape.ndim(); ++i) tail *= shape[i];

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

void FlipLeftRight(const nnvm::NodeAttrs &attrs,
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

void FlipTopBottom(const nnvm::NodeAttrs &attrs,
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

void RandomFlipLeftRight(
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

void RandomFlipTopBottom(
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

void RandomBrightness(const nnvm::NodeAttrs &attrs,
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

void RGB2HLSConvert(const float& src_r,
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

void HLS2RGBConvert(const float& src_h,
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

void AdjustHueImpl(float alpha,
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

void RandomHue(const nnvm::NodeAttrs &attrs,
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

void RandomColorJitter(const nnvm::NodeAttrs &attrs,
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
  nnvm::Tuple<float> alpha;
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

void AdjustLightingImpl(const nnvm::Tuple<float>& alpha,
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

void AdjustLighting(const nnvm::NodeAttrs &attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const AdjustLightingParam &param = nnvm::get<AdjustLightingParam>(attrs.parsed);
  AdjustLightingImpl(param.alpha, ctx, inputs, req, outputs);
}

void RandomLighting(const nnvm::NodeAttrs &attrs,
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
  .set_attr<nnvm::FInferShape>("FInferShape", ImageShape)                   \
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

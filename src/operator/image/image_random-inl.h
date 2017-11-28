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


#include <mxnet/base.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

inline bool CheckIsImage(const TBlob &image) {
  CHECK_EQ(image.type_flag_, mshadow::kUint8) << "input type is not an image.";
  CHECK_EQ(image.ndim(), 3) << "input dimension is not 3.";
  CHECK(image.shape_[2] == 1 || image.shape_[2] == 3) << "image channel should be 1 or 3.";
}

static void RandomFlip(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
}

inline bool ToTensorType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ((*in_attrs)[0], mshadow::kUint8)
    << "`to_tensor` only supports uint8 input";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

inline bool ToTensorShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape &shp = (*in_attrs)[0];
  CHECK_EQ(shp.ndim(), 3U) << "`to_tensor` only supports 3 dimensions";
  TShape ret(3);
  ret[0] = shp[2];
  ret[1] = shp[0];
  ret[2] = shp[1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}

static void ToTensor(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace";
  CheckIsImage(inputs[0]);

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int channel = inputs[0].shape_[2];

  float* output = outputs[0].dptr<float>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  for (int l = 0; l < length; ++l) {
    for (int c = 0; c < channel; ++c) {
      output[c*length + l] = static_cast<float>(input[l*channel + c]) / 255.0f;
    }
  }
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
      << "Input must have 3 dimensions";

  auto nchannels = dshape[0];
  CHECK(param.mean.ndim() == 1 || param.mean.ndim() == nchannels)
      << "mean must have either 1 or " << nchannels << " elements";
  CHECK(param.std.ndim() == 1 || param.std.ndim() == nchannels)
      << "std must have either 1 or " << nchannels << " elements";

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
}


static void Normalize(const nnvm::NodeAttrs &attrs,
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

struct FlipParam : public dmlc::Parameter<FlipParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("0 or 1. 0 for horizontal flip, 1 for vertical flip.");
  }
};

#define SWAP_IF_INPLACE(dst, dst_idx, src, src_idx) \
  if (dst == src) {                                 \
    std::swap(dst[dst_idx], src[src_idx]);          \
  } else {                                          \
    dst[dst_idx] = src[src_idx];                    \
  }

template<typename DType>
static void FlipImpl(const TShape &shape, DType *src, DType *dst, int axis) {
  const int height = shape[0];
  const int width = shape[1];
  const int nchannel = shape[2];

  const int length = width * nchannel;
  const int height_stride = (src == dst && axis == 1) ? (height >> 1) : height;
  const int width_stride = (src == dst && axis == 0) ? (width >> 1) : width;

  for (int h = 0; h < height_stride; ++h) {
    const int h_dst = (axis == 0) ? h : (height - h);
    for (int w = 0; w < width_stride; ++w) {
      const int w_dst = (axis == 0) ? (width - w) : w;
      const int idx_dst = h_dst * length + w_dst * nchannel;
      const int idx_src = h * length + w * nchannel;
      SWAP_IF_INPLACE(dst, idx_dst, src, idx_src);
      if (nchannel > 1) {
        SWAP_IF_INPLACE(dst, idx_dst + 1, src, idx_src + 1);
        SWAP_IF_INPLACE(dst, idx_dst + 2, src, idx_src + 2);
      }
    }
  }
}

static void Flip(const nnvm::NodeAttrs &attrs,
                  const OpContext &ctx,
                  const std::vector<TBlob> &inputs,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &outputs) {
  const FlipParam &param = nnvm::get<FlipParam>(attrs.parsed);
  CHECK(param.axis == 0 || param.axis == 1) << "flip axis must be 0 or 1.";
  CheckIsImage(inputs[0]);
  const TShape& ishape = inputs[0].shape_;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    FlipImpl(ishape, inputs[0].dptr<DType>(), outputs[0].dptr<DType>(), param.axis);
  });
}

struct RandomBrightnessParam : public dmlc::Parameter<RandomBrightnessParam> {
  float max_brightness;
  DMLC_DECLARE_PARAMETER(RandomBrightnessParam) {
    DMLC_DECLARE_FIELD(max_brightness)
    .set_lower_bound(0.0)
    .describe("Max Brightness.");
  }
};

static void RandomBrightness(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);

  int length = inputs[0].Size();

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  float alpha_b = 1.0 + std::uniform_real_distribution<float>(
      -param.max_brightness, param.max_brightness)(prnd->GetRndEngine());

  for (int l = 0; l < length; ++l) {
    float val = static_cast<float>(input[l]) * alpha_b;
    val = std::min(std::max(val, 0.f), 255.f);
    output[l] = static_cast<uint8_t>(val);
  }
}


struct RandomContrastParam : public dmlc::Parameter<RandomContrastParam> {
  float max_contrast;
  DMLC_DECLARE_PARAMETER(RandomContrastParam) {
    DMLC_DECLARE_FIELD(max_contrast)
    .set_lower_bound(0.0)
    .describe("Max Contrast.");
  }
};


static void RandomContrast(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  static const float coef[] = { 0.299f, 0.587f, 0.114f };
  const RandomContrastParam &param = nnvm::get<RandomContrastParam>(attrs.parsed);

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_c = 1.0 + std::uniform_real_distribution<float>(
    -param.max_contrast, param.max_contrast)(prnd->GetRndEngine());

  float sum = 0.f;
  if (nchannels > 1) {
    for (int l = 0; l < length; ++l) {
      for (int c = 0; c < nchannels; ++c) sum += input[l*nchannels + c] * coef[c];
    }
  } else {
    for (int l = 0; l < length; ++l) sum += input[l];
  }
  float gray_mean = sum / static_cast<float>(length);
  float beta = (1 - alpha_c) * gray_mean;

  for (int l = 0; l < length * nchannels; ++l) {
    float val = input[l] * alpha_c + beta;
    val = std::min(std::max(val, 0.f), 255.f);
    output[l] = static_cast<uint8_t>(val);
  }
}

struct RandomSaturationParam : public dmlc::Parameter<RandomSaturationParam> {
  float max_saturation;
  DMLC_DECLARE_PARAMETER(RandomSaturationParam) {
    DMLC_DECLARE_FIELD(max_saturation)
    .set_default(0.0)
    .describe("Max Saturation.");
  }
};

static void RandomSaturation(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomSaturationParam &param = nnvm::get<RandomSaturationParam>(attrs.parsed);
  static const float coef[] = { 0.299f, 0.587f, 0.114f };

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_s = 1.f + std::uniform_real_distribution<float>(
    -param.max_saturation, param.max_saturation)(prnd->GetRndEngine());
  float alpha_o = 1.f - alpha_s;

  if (nchannels == 1) {
    for (int l = 0; l < length * nchannels; ++l) output[l] = input[l];
    return;
  }

  for (int l = 0; l < length; ++l) {
    float gray = 0.f;
    for (int c = 0; c < nchannels; ++c) {
      gray = input[l*nchannels + c] * coef[c];
    }
    gray *= alpha_o;
    for (int c = 0; c < nchannels; ++c) {
      float val = gray + input[l*nchannels + c] * alpha_s;
      val = std::min(std::max(val, 0.f), 255.f);
      output[l*nchannels + c] = static_cast<uint8_t>(val);
    }
  }
}

struct RandomHueParam : public dmlc::Parameter<RandomHueParam> {
  float max_hue;
  DMLC_DECLARE_PARAMETER(RandomHueParam) {
    DMLC_DECLARE_FIELD(max_hue)
    .set_default(0.0)
    .describe("Max Hue.");
  }
};

template <typename DType> static
void RGB2HLSConvert(const DType src_r,
                    const DType src_g,
                    const DType src_b,
                    DType *dst_h,
                    DType *dst_l,
                    DType *dst_s
                   ) {
  DType b = src_b, g = src_g, r = src_r;
  DType h = 0.f, s = 0.f, l;
  DType vmin;
  DType vmax;
  DType diff;

  vmax = vmin = r;
  vmax = fmax(vmax, g);
  vmax = fmax(vmax, b);
  vmin = fmin(vmin, g);
  vmin = fmin(vmin, b);

  diff = vmax - vmin;
  l = (vmax + vmin) * 0.5f;

  if (diff > std::numeric_limits<DType>::epsilon()) {
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


static  int c_HlsSectorData[6][3] = {
  { 1, 3, 0 },
  { 1, 0, 2 },
  { 3, 0, 1 },
  { 0, 2, 1 },
  { 0, 1, 3 },
  { 2, 1, 0 }
};

template <typename DType>  static  void HLS2RGBConvert(const DType src_h,
    const DType src_l,
    const DType src_s,
    DType *dst_r,
    DType *dst_g,
    DType *dst_b) {


  float h = src_h, l = src_l, s = src_s;
  float b = l, g = l, r = l;

  if (s != 0) {
    float p2 = (l <= 0.5f) * l * (1 + s);
    p2 += (l > 0.5f) * (l + s - l * s);
    float p1 = 2 * l - p2;

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

  *dst_b = b;
  *dst_g = g;
  *dst_r = r;
}

template<typename xpu, typename DType>
static  void RandomHueKernal(const TBlob &input,
                             const TBlob &output,
                             Stream<xpu> *s,
                             int hight,
                             int weight,
                             DType alpha) {
  auto input_3d = input.get<xpu, 3, DType>(s);
  auto output_3d = output.get<xpu, 3, DType>(s);
  for (int h_index = 0; h_index < hight; ++h_index) {
    for (int w_index = 0; w_index < weight; ++w_index) {
      DType h;
      DType l;
      DType s;
      RGB2HLSConvert(input_3d[0][h_index][w_index],
                     input_3d[1][h_index][w_index],
                     input_3d[2][h_index][w_index],
                     &h, &l, &s);
      h += alpha;
      h = std::max(DType(0), std::min(DType(180), h));

      HLS2RGBConvert(
        h, l, s,
        &output_3d[0][h_index][w_index],
        &output_3d[1][h_index][w_index],
        &output_3d[2][h_index][w_index]);
    }
  }
}

template<typename xpu>
static void RandomHue(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  auto input = inputs[0];
  auto output = outputs[0];
  int channel = input.shape_[0];
  int hight = input.shape_[1];
  int weight = input.shape_[2];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu> *prnd = ctx.requested[kRandom].get_random<xpu, real_t>(s);

  const RandomHueParam &param = nnvm::get<RandomHueParam>(attrs.parsed);
  float alpha =  std::uniform_real_distribution<float>(
    -param.max_hue, param.max_hue)(prnd->GetRndEngine());
  auto output_float = output.get<xpu, 3, float>(s);

  MSHADOW_TYPE_SWITCH(input.type_flag_, DType, {
    RandomHueKernal<xpu, DType>(input, output, s, hight, weight, alpha);
  });
}

static void RandomColorJitter(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
}

struct AdjustLightingParam : public dmlc::Parameter<AdjustLightingParam> {
  nnvm::Tuple<float> alpha_rgb;
  nnvm::Tuple<float> eigval;
  nnvm::Tuple<float> eigvec;
  DMLC_DECLARE_PARAMETER(AdjustLightingParam) {
    DMLC_DECLARE_FIELD(alpha_rgb)
    .set_default({0, 0, 0})
    .describe("The lighting alphas for the R, G, B channels.");
    DMLC_DECLARE_FIELD(eigval)
    .describe("Eigen value.")
    .set_default({ 55.46, 4.794, 1.148 });
    DMLC_DECLARE_FIELD(eigvec)
    .describe("Eigen vector.")
    .set_default({ -0.5675,  0.7192,  0.4009,
                   -0.5808, -0.0045, -0.8140,
                   -0.5808, -0.0045, -0.8140 });
  }
};

struct RandomLightingParam : public dmlc::Parameter<RandomLightingParam> {
  float alpha_std;
  nnvm::Tuple<float> eigval;
  nnvm::Tuple<float> eigvec;
  DMLC_DECLARE_PARAMETER(RandomLightingParam) {
    DMLC_DECLARE_FIELD(alpha_std)
    .set_default(0.05)
    .describe("Level of the lighting noise.");
    DMLC_DECLARE_FIELD(eigval)
    .describe("Eigen value.")
    .set_default({ 55.46, 4.794, 1.148 });
    DMLC_DECLARE_FIELD(eigvec)
    .describe("Eigen vector.")
    .set_default({ -0.5675,  0.7192,  0.4009,
                   -0.5808, -0.0045, -0.8140,
                   -0.5808, -0.0045, -0.8140 });
  }
};

void AdjustLightingImpl(uint8_t* dst, const uint8_t* src,
                        float alpha_r, float alpha_g, float alpha_b,
                        const nnvm::Tuple<float> eigval, const nnvm::Tuple<float> eigvec,
                        int H, int W) {
    alpha_r *= eigval[0];
    alpha_g *= eigval[1];
    alpha_b *= eigval[2];
    float pca_r = alpha_r * eigvec[0] + alpha_g * eigvec[1] + alpha_b * eigvec[2];
    float pca_g = alpha_r * eigvec[3] + alpha_g * eigvec[4] + alpha_b * eigvec[5];
    float pca_b = alpha_r * eigvec[6] + alpha_g * eigvec[7] + alpha_b * eigvec[8];
    for (int i = 0; i < H * W; i++) {
        int base_ind = 3 * i;
        float in_r = static_cast<float>(src[base_ind]);
        float in_g = static_cast<float>(src[base_ind + 1]);
        float in_b = static_cast<float>(src[base_ind + 2]);
        dst[base_ind] = std::min(255, std::max(0, static_cast<int>(in_r + pca_r)));
        dst[base_ind + 1] = std::min(255, std::max(0, static_cast<int>(in_g + pca_g)));
        dst[base_ind + 2] = std::min(255, std::max(0, static_cast<int>(in_b + pca_b)));
    }
}

static void AdjustLighting(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    const AdjustLightingParam &param = nnvm::get<AdjustLightingParam>(attrs.parsed);
    CHECK_EQ(param.eigval.ndim(), 3) << "There should be 3 numbers in the eigval.";
    CHECK_EQ(param.eigvec.ndim(), 9) << "There should be 9 numbers in the eigvec.";
    CHECK_EQ(inputs[0].ndim(), 3);
    CHECK_EQ(inputs[0].size(2), 3);
    int H = inputs[0].size(0);
    int W = inputs[0].size(1);
    AdjustLightingImpl(outputs[0].dptr<uint8_t>(), inputs[0].dptr<uint8_t>(),
                       param.alpha_rgb[0], param.alpha_rgb[1], param.alpha_rgb[2],
                       param.eigval, param.eigvec, H, W);
}

static void RandomLighting(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    const RandomLightingParam &param = nnvm::get<RandomLightingParam>(attrs.parsed);
    CHECK_EQ(param.eigval.ndim(), 3) << "There should be 3 numbers in the eigval.";
    CHECK_EQ(param.eigvec.ndim(), 9) << "There should be 9 numbers in the eigvec.";
    CHECK_EQ(inputs[0].ndim(), 3);
    CHECK_EQ(inputs[0].size(2), 3);
    int H = inputs[0].size(0);
    int W = inputs[0].size(1);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
    std::normal_distribution<float> dist(0, param.alpha_std);
    float alpha_r = dist(prnd->GetRndEngine());
    float alpha_g = dist(prnd->GetRndEngine());
    float alpha_b = dist(prnd->GetRndEngine());
    AdjustLightingImpl(outputs[0].dptr<uint8_t>(), inputs[0].dptr<uint8_t>(),
                       alpha_r, alpha_g, alpha_b,
                       param.eigval, param.eigvec, H, W);
}




}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_

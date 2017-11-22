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
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

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

static void RandomHue(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
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
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int base_ind = 3 * (i * W + j);
            float in_r = static_cast<float>(src[base_ind]);
            float in_g = static_cast<float>(src[base_ind + 1]);
            float in_b = static_cast<float>(src[base_ind + 2]);
            dst[base_ind] = std::min(255, std::max(0, static_cast<int>(in_r + pca_r)));
            dst[base_ind + 1] = std::min(255, std::max(0, static_cast<int>(in_g + pca_g)));
            dst[base_ind + 2] = std::min(255, std::max(0, static_cast<int>(in_b + pca_b)));
        }
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

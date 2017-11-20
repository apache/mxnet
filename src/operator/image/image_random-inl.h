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
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "../mxnet_op.h"
#include "image_common.h"
#include "../../operator/operator_common.h"

namespace mxnet {
namespace op {


enum ImageRandomResource { kRandom };

template<typename xpu>
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

template<typename xpu>
static void ToTensor(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  auto input = inputs[0];
  auto output = outputs[0];

  int height = input.shape_[0];
  int weight = input.shape_[1];
  int channel = input.shape_[2];

  typedef float   DstDType;
  typedef uint8_t SrcDType;

  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace";

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    auto input_3d =  input.get<xpu, 3, SrcDType>(s);
    auto output_3d = output.get<xpu, 3, DstDType>(s);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < weight; ++w) {
        for (int c = 0; c < channel; ++c) {
          Assign(output_3d[c][h][w], Req, DstDType(input_3d[h][w][c] / 255.0));
        }
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

struct normalize {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const OpReqType req,
                                  const int nchannel, const int size,
                                  const float *mean, const float *std) {
    int c = 0;
    switch (nchannel) {
      case 1:
        break;
      case 3:
        if (i < size) {
          c = 0;
        } else if (i < (size << 1)) {
          c = 1;
        } else {
          c = 2;
        }
        break;
      default:
        LOG(FATAL) << "not support channel" << nchannel;
    }
    float m = (mean ? mean[c] : 0);
    KERNEL_ASSIGN(out[i], req, static_cast<DType>((in[i] - m) / std[c]));
  }
};

static void NormalizeCheckParam(const nnvm::Tuple<float> &mean,
                                const nnvm::Tuple<float> &std,
                                const int nchannel) {
  CHECK(mean.ndim() == 1 || mean.ndim() == 3)
    << "Mean must be in dimension 1 or 3.";
  CHECK(std.ndim() == 1 || std.ndim() == 3)
    << "Standard deviations must be in dimension 1 or 3.";
  CHECK(nchannel == 1 || nchannel == 3) << "Image channel must be 1 or 3.";
  CHECK_EQ(mean.ndim(), nchannel)
    << "Mean dimension does not agree with image channel.";
  CHECK_EQ(std.ndim(), nchannel)
    << "Standard deviations dimension does not agree with image channel.";
  for (uint32_t c = 0; c < std.ndim(); ++c) {
    CHECK(std[c] > 0) << "Invalid standard deviation " << std[c];
  }
}

template<typename xpu>
static void Normalize(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);
  auto mean = param.mean;
  auto std = param.std;

  int nchannel = inputs[0].shape_[0];
  NormalizeCheckParam(mean, std, nchannel);

  int size = inputs[0].Size() / nchannel;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      mxnet_op::Kernel<normalize, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
        Req, nchannel, size, mean.begin(), std.begin());
    });
  });
}

template<typename xpu>
static void NormalizeBackward(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);
  int nchannel = inputs[0].shape_[0];

  NormalizeCheckParam(param.mean, param.std, nchannel);

  int size = inputs[0].Size() / nchannel;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      mxnet_op::Kernel<normalize, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
        Req, nchannel, size, nullptr, param.std.begin());
      });
  });
}

struct RandomBrightnessParam : public dmlc::Parameter<RandomBrightnessParam> {
  float max_brightness;
  DMLC_DECLARE_PARAMETER(RandomBrightnessParam) {
    DMLC_DECLARE_FIELD(max_brightness)
    .set_default(0.0)
    .describe("Max Brightness.");
  }
};

template<typename xpu>
static void RandomBrightness(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  auto input = inputs[0];
  auto output = outputs[0];
  int channel = input.shape_[0];
  int height = input.shape_[1];
  int weight = input.shape_[2];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu> *prnd = ctx.requested[kRandom].get_random<xpu, real_t>(s);

  const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);
  float alpha_b = 1.0 + std::uniform_real_distribution<float>(
    -param.max_brightness, param.max_brightness)(prnd->GetRndEngine());
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow::op::mul, Req>, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), DType(alpha_b));
    });
  });
}

struct RandomContrastParam : public dmlc::Parameter<RandomContrastParam> {
  float max_contrast;
  DMLC_DECLARE_PARAMETER(RandomContrastParam) {
    DMLC_DECLARE_FIELD(max_contrast)
    .set_default(0.0)
    .describe("Max Contrast.");
  }
};

/*! \brief mul_add operator */
struct mul_add {
  /*! \brief map a, b, c to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b, DType c) {
    return a * b + c;
  }
};

template<typename xpu>
static void RandomContrast(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  auto input = inputs[0];
  auto output = outputs[0];
  int channel = input.shape_[0];
  int height = input.shape_[1];
  int weight = input.shape_[2];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu> *prnd = ctx.requested[kRandom].get_random<xpu, real_t>(s);


  const RandomContrastParam &param = nnvm::get<RandomContrastParam>(attrs.parsed);
  float alpha_c = 1.0 + std::uniform_real_distribution<float>(
    -param.max_contrast, param.max_contrast)(prnd->GetRndEngine());

  const float R2YF = 0.299f;
  const float G2YF = 0.587f;
  const float B2YF = 0.114f;
  static const float coeffs0[] = { R2YF, G2YF, B2YF };

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    auto input_3d = input.get<xpu, 3, DType>(s);
    DType sum = (DType)0.0;
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < weight; ++w) {
          sum += input_3d[c][h][w] * coeffs0[c];
        }
      }
    }
    float gray_mean = sum / static_cast<float>(height * weight);
    float beta = (1 - alpha_c) * gray_mean;

    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mul_add, Req>, xpu>::Launch(
        s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), DType(alpha_c), DType(beta));
    });
  });
}

struct RandomSaturationParam : public dmlc::Parameter<RandomSaturationParam> {
  float max_saturation;
  DMLC_DECLARE_PARAMETER(RandomSaturationParam) {
    DMLC_DECLARE_FIELD(max_saturation)
    .set_default(0.0)
    .describe("Max Saturation.");
  }
};

template<typename xpu>
static void RandomSaturation(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  auto input = inputs[0];
  auto output = outputs[0];
  int channel = input.shape_[0];
  int height = input.shape_[1];
  int weight = input.shape_[2];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu> *prnd = ctx.requested[kRandom].get_random<xpu, real_t>(s);
  const RandomSaturationParam &param = nnvm::get<RandomSaturationParam>(attrs.parsed);
  float alpha_s = 1.0 + std::uniform_real_distribution<float>(
    -param.max_saturation, param.max_saturation)(prnd->GetRndEngine());
  float alpha_o = 1 - alpha_s;
  const float R2YF = 0.299f;
  const float G2YF = 0.587f;
  const float B2YF = 0.114f;
  static const float coeffs0[] = { R2YF, G2YF, B2YF };


  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      auto input_3d =  input.get<xpu, 3, DType>(s);
      auto output_3d = output.get<xpu, 3, DType>(s);
      switch (channel) {
        case 1:
          Assign(output_3d, Req, input_3d)
          break;
        case 3:
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < weight; ++w) {
              float gray =
                input_3d[0][h][w] * R2YF + input_3d[1][h][w] * G2YF + input_3d[2][h][w] * B2YF;
              Assign(output_3d[0][h][w], Req, DType(gray * alpha_s + input_3d[0][h][w] * alpha_o))
            }
          }
          break;
        default:
          LOG(FATAL) << "not support channel" << channel;
      }
    });
  });
}

template<typename xpu>
static void RandomHue(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
}

template<typename xpu>
static void RandomColorJitter(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
}

template<typename xpu>
static void RandomLighting(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
}




}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_

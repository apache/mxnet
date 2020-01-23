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
 * Copyright (c) 2018 by Contributors
 * \file bilinear_resize-inl.h
 * \brief bilinear resize operator
 * \author Hang Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_INL_H_
#define MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
/* contrib
#include "../ndarray/ndarray_function.h"
#include "./operator_common.h"
#include "./mxnet_op.h"
#include "./mshadow_op.h"
*/
#include "../../ndarray/ndarray_function.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace bilinear_resize {
enum BilinearResizeOpMode{simple, odd_scale, like, to_even_down, to_even_up, to_odd_down,
  to_odd_up};
}  // namespace bilinear_resize


namespace mxnet {
namespace op {

struct BilinearSampleParam : public dmlc::Parameter<BilinearSampleParam> {
  int height;
  int width;
  dmlc::optional<float> scale_height;
  dmlc::optional<float> scale_width;
  int mode;
  bool align_corners;
  DMLC_DECLARE_PARAMETER(BilinearSampleParam) {
    DMLC_DECLARE_FIELD(height).set_default(1).set_range(1, 10000)
    .describe("output height (required, but ignored if scale_height is defined or mode is not "
              "\"size\")");
    DMLC_DECLARE_FIELD(width).set_default(1).set_range(1, 10000)
    .describe("output width (required, but ignored if scale_width is defined or mode is not "
              "\"size\")");
    DMLC_DECLARE_FIELD(scale_height).set_default(dmlc::optional<float>())
    .describe("sampling scale of the height (optional, used in modes \"scale\" and \"odd_scale\")");
    DMLC_DECLARE_FIELD(scale_width).set_default(dmlc::optional<float>())
    .describe("sampling scale of the width (optional, used in modes \"scale\" and \"odd_scale\")");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("size", bilinear_resize::simple)
    .add_enum("odd_scale", bilinear_resize::odd_scale)
    .add_enum("like", bilinear_resize::like)
    .add_enum("to_even_down", bilinear_resize::to_even_down)
    .add_enum("to_even_up", bilinear_resize::to_even_up)
    .add_enum("to_odd_down", bilinear_resize::to_odd_down)
    .add_enum("to_odd_up", bilinear_resize::to_odd_up)
    .set_default(bilinear_resize::simple)
    .describe("resizing mode. \"simple\" - output height equals parameter \"height\" if "
              "\"scale_height\" parameter is not defined or input height multiplied by "
              "\"scale_height\" otherwise. Same for width;"
              "\"odd_scale\" - if original height or width is odd, then result height is "
              "calculated like result_h = (original_h - 1) * scale + 1; "
              "for scale > 1 the result shape would be like if we did deconvolution with kernel "
              "= (1, 1) and stride = (height_scale, width_scale); and for scale < 1 shape "
              "would be like we did convolution with kernel = (1, 1) and "
              "stride = (int(1 / height_scale), int( 1/ width_scale);"
              "\"like\" - resize first input to the height and width of second input; "
              "\"to_even_down\" - resize input to nearest lower even height and width "
              "(if original height is odd then result height = original height - 1);"
              "\"to_even_up\" - resize input to nearest bigger even height and width "
              "(if original height is odd then result height = original height + 1);"
              "\"to_odd_down\" - resize input to nearest odd height and width "
              "(if original height is odd then result height = original height - 1);"
              "\"to_odd_up\" - resize input to nearest odd height and width "
              "(if original height is odd then result height = original height + 1);");
  DMLC_DECLARE_FIELD(align_corners).set_default(true)
    .describe("With align_corners = True, the interpolating doesn't proportionally align the"
              "output and input pixels, and thus the output values can depend on the input size.");
  }
};

template <typename DType>
static inline DType area_pixel_compute_scale(
  int64_t input_size,
  int64_t output_size,
  bool align_corners) {
  /* We view each pixel as an area, idx + 0.5 as its center index.
   * Here is an example formula in 1D case.
   * if align_corners: center of two corner pixel areas are preserved,
   *     (0.5, 0.5) -> (0.5, 0.5),
   *     (input_size - 0.5, 0.5) -> (output_size - 0.5)
   *     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
   *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
   * if not align_corners: the whole range is scaled accordingly
   *     scale = input_size / output_size
   *     src_idx + 0.5 = scale * (dst_index + 0.5)
   */
  if (output_size > 1) {
    return align_corners
      ? static_cast<DType>(input_size - 1) / (output_size - 1)
      : static_cast<DType>(input_size) / output_size;
  } else {
    return DType(0);
  }
}

template <typename DType>
static inline DType area_pixel_compute_source_index(
  DType scale,
  int64_t dst_index,
  bool align_corners,
  bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    DType src_idx = scale * (dst_index + 0.5) - 0.5;
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO(chinakook): Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    return (!cubic && src_idx < 0) ? DType(0) : src_idx;
  }
}

static inline bool IsWriting(const OpReqType ort) {
  return ort == kWriteTo || ort == kWriteInplace;
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<cpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output,
                                           bool align_corners);

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<cpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output,
                                              bool modeLike,
                                              bool align_corners);

#if MXNET_USE_CUDA
template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<gpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output,
                                           bool align_corners);

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<gpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output,
                                              bool modeLike,
                                              bool align_corners);
#endif  // MXNET_USE_CUDA

template <typename xpu>
inline void BilinearSampleOpForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<TBlob> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<TBlob> &outputs) {
  const BilinearSampleParam& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  size_t expected = param.mode == bilinear_resize::like ? 2 : 1;
  CHECK_EQ(inputs.size(), expected);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs[0].CheckContiguous(), true);
  if (expected == 2) {
  CHECK_EQ(inputs[1].CheckContiguous(), true);
  }
  CHECK_EQ(outputs[0].CheckContiguous(), true);

  bool align_corners = param.align_corners;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    SpatialUpSamplingBilinearUpdateOutput<xpu, DType, AccReal>(s, inputs, outputs, align_corners);
  });
}


template <typename xpu>
inline void BilinearSampleOpBackward(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<TBlob> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<TBlob> &outputs) {
  const BilinearSampleParam& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  bool modeLike = param.mode == bilinear_resize::like;
  bool align_corners = param.align_corners;
  size_t expected = modeLike ? 2 : 1;
  CHECK_EQ(outputs.size(), expected);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (IsWriting(req[0])) {
    // zero grad before backwarding
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      Fill<false>(s, outputs[0], kWriteTo, 0);
    })
  }
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    SpatialUpSamplingBilinearUpdateGradInput<xpu, DType, AccReal>(s, inputs, outputs
      , modeLike, align_corners);
  });
}


static bool BilinearSampleOpInferShape(const nnvm::NodeAttrs& attrs,
                                       mxnet::ShapeVector *in_shape,
                                       mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  CHECK_EQ(out_shape->size(), 1U) << "Output:[data]";
  const BilinearSampleParam& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  size_t expected = param.mode == bilinear_resize::like ? 2 : 1;
  CHECK_EQ(in_shape->size(), expected);
  mxnet::TShape dshape(in_shape->at(0));
  if (mxnet::op::shape_is_none(dshape)) return false;
  int16_t new_height = -1;
  int16_t new_width = -1;
  switch (param.mode) {
    case bilinear_resize::simple:
    {
      if (param.scale_height.has_value()) {
        new_height = static_cast<int>(param.scale_height.value() * in_shape->at(0)[2]);
      } else {
        new_height = param.height;
      }
      if (param.scale_height.has_value()) {
        new_width = static_cast<int>(param.scale_width.value() * in_shape->at(0)[3]);
      } else {
        new_width = param.width;
      }
      break;
    }
    case bilinear_resize::odd_scale:
    {
      new_height = ((dshape[2] % 2) == 0) ? (int16_t) (dshape[2] * param.scale_height.value()) :
                   (int16_t) ((dshape[2] - 1) * param.scale_height.value()) + 1;
      new_width = ((dshape[3] % 2) == 0) ? (int16_t) (dshape[3] * param.scale_width.value()) :
                  (int16_t) ((dshape[3] - 1) * param.scale_width.value()) + 1;
      break;
    }
    case bilinear_resize::like:
    {
      TShape like_shape(in_shape->at(1));
      if (dshape.ndim() == 0) return false;
      new_height = like_shape[2];
      new_width = like_shape[3];
      break;
    }
    case bilinear_resize::to_even_down:
    {
      new_height = ((dshape[2] % 2) == 0) ? dshape[2] : dshape[2] - 1;
      new_width = ((dshape[3] % 2) == 0) ? dshape[3] : dshape[3] - 1;
      break;
    }
    case bilinear_resize::to_even_up:
    {
      new_height = ((dshape[2] % 2) == 0) ? dshape[2] : dshape[2] + 1;
      new_width = ((dshape[3] % 2) == 0) ? dshape[3] : dshape[3] + 1;
      break;
    }
    case bilinear_resize::to_odd_down:
    {
      new_height = ((dshape[2] % 2) == 1) ? dshape[2] : dshape[2] - 1;
      new_width = ((dshape[3] % 2) == 1) ? dshape[3] : dshape[3] - 1;
      break;
    }
    case bilinear_resize::to_odd_up:
    {
      new_height = ((dshape[2] % 2) == 1) ? dshape[2] : dshape[2] + 1;
      new_width = ((dshape[3] % 2) == 1) ? dshape[3] : dshape[3] + 1;
      break;
    }
    default:
    {
      LOG(FATAL) << "Invalid mode " << param.mode;
    }
  }

  dshape[2] = new_height;
  dshape[3] = new_width;

  out_shape->clear();
  out_shape->push_back(dshape);
  return true;
}


inline uint16_t BilinearSampleOpNumInputs(const NodeAttrs& attrs) {
  auto& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  if (param.mode == bilinear_resize::like) {
    return 2;
  } else {
    return 1;
  }
}

inline uint16_t BilinearSampleOpNumBackwardOutputs(const NodeAttrs& attrs) {
  auto& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  if (param.mode == bilinear_resize::like) {
    return 2;
  } else {
    return 1;
  }
}

inline std::vector<std::string> BilinearSampleOpInputNames(const NodeAttrs& attrs) {
  auto& param = nnvm::get<BilinearSampleParam>(attrs.parsed);
  if (param.mode == bilinear_resize::like) {
    return std::vector<std::string>{"data", "like"};
  } else {
    return std::vector<std::string>{"data"};
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_INL_H_

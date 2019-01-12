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
 *  Copyright (c) 2018 by Contributors
 * \file random_resize_crop-inl.h
 * \brief the image random_resize_crop operator implementation
 */

#ifndef MXNET_OPERATOR_IMAGE_RANDOM_RESIZE_CROP_INL_H_
#define MXNET_OPERATOR_IMAGE_RANDOM_RESIZE_CROP_INL_H_


#include <algorithm>
#include <utility>
#include <vector>
#include <math.h>

#include "mxnet/base.h"
#include "dmlc/optional.h"

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "image_utils.h"

namespace mxnet {
namespace op {
namespace image {

struct RandomResizeCropParam : public dmlc::Parameter<RandomResizeCropParam> {
  nnvm::Tuple<int> size;
  nnvm::Tuple<float> scale;
  nnvm::Tuple<float> ratio;
  int interp;
  DMLC_DECLARE_PARAMETER(RandomResizeCropParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(nnvm::Tuple<int>())
    .describe("Size of the final output.");
    DMLC_DECLARE_FIELD(scale)
    .set_default(nnvm::Tuple<float>())
    .describe("If scale is `(min_area, max_area)`, the cropped image's area will"
        "range from min_area to max_area of the original image's area");
    DMLC_DECLARE_FIELD(ratio)
    .set_default(nnvm::Tuple<float>())
    .describe("Range of aspect ratio of the cropped image before resizing.");
    DMLC_DECLARE_FIELD(interp)
    .describe("Interpolation method for resizing. By default uses bilinear"
        "interpolation. See OpenCV's resize function for available choices.");
  }
};

bool RandomResizeCropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const RandomResizeCropParam& param = nnvm::get<RandomResizeCropParam>(attrs.parsed);
  const auto size = GetHeightAndWidthFromSize(param);

  CHECK((size.height > 0) && (size.width > 0))
      << "Input height and width must be greater than 0";
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({size.height, size.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], size.height, size.width, ishape[kC]}));
  }

  return true;
}

void RandomResizeCrop(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  CHECK((inputs[0].ndim() == 3) || (inputs[0].ndim() == 4))
      << "Input data must be (h, w, c) or (n, h, w, c)";
  const RandomResizeCropParam& param = nnvm::get<RandomResizeCropParam>(attrs.parsed);
  
  auto h = inputs[0].shape_[inputs[0].ndim() == 3 ? H : kH];
  auto w = inputs[0].shape_[inputs[0].ndim() == 3 ? W : kW];
  auto src_area = h * w;

  CHECK(param.scale.ndim() == 1 || param.scale.ndim() == 2)
         << "Input scale must be float in (0, 1] or tuple of (float, float)";
  std::pair<float, float> area;
  if (param.scale.ndim() == 1) {
    area = std::make_pair(param.scale[0], 1.0f);
  } else {
    area = std::make_pair(param.scale[0], param.scale[1]);
  }
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  mshadow::Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  for (auto i = 0; i < 10; ++i) {
    float target_area = static_cast<float>(std::uniform_real_distribution<float>(
      area.first, area.second)(prnd->GetRndEngine()) * src_area);
    float new_ratio = static_cast<float>(std::uniform_real_distribution<float>(
      param.ratio[0], param.ratio[1])(prnd->GetRndEngine()));

    int new_w = static_cast<int>(round(sqrt(target_area * new_ratio)));
    int new_h = static_cast<int>(round(sqrt(target_area / new_ratio)));

    if (std::uniform_real_distribution<float>(0.0, 1.0)(prnd->GetRndEngine()) < 0.5) {
      new_h = new_w;
      new_w = new_h;
    }

    if (new_w <= w && new_h <= h) {
      int x0 = std::uniform_int_distribution<int>(0, w - new_w)(prnd->GetRndEngine());
      int y0 = std::uniform_int_distribution<int>(0, h - new_h)(prnd->GetRndEngine());
      SizeParam size = GetHeightAndWidthFromSize(param);
      bool need_resize = false;

      if ((size.height != new_h) || (size.width != new_w)) {
        need_resize = true;
      }
      if (inputs[0].ndim() == 3) {
        CropImpl(inputs, outputs, x0, y0, new_h, new_w, size, param.interp);
      } else {
        const auto batch_size = inputs[0].shape_[0];
        const auto input_offset = inputs[0].shape_[kH] * inputs[0].shape_[kW] * inputs[0].shape_[kC];
        int output_offset;
        if (need_resize) {
          output_offset = size.height * size.width * outputs[0].shape_[kC];
        } else {
          output_offset = new_h * new_w * outputs[0].shape_[kC];
        }
        #pragma omp parallel for
        for (auto i = 0; i < batch_size; ++i) {
          CropImpl(inputs, outputs, x0, y0, new_h, new_w, size, param.interp, input_offset * i, output_offset * i);
        }
      }
      return ;
    }
  }
  // fall back to center_crop
  return CenterCropImpl(inputs, outputs, GetHeightAndWidthFromSize(param), param.interp);
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_RANDOM_RESIZE_CROP_INL_H_

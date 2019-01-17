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
 * \file totensor_op-inl.h
 * \brief Image to tensor operator
*/
#ifndef MXNET_OPERATOR_IMAGE_TOTENSOR_OP_INL_H_
#define MXNET_OPERATOR_IMAGE_TOTENSOR_OP_INL_H_


#include <vector>
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

// There are no parameters for this operator.
// Hence, no arameter registration.

// Shape and Type inference for image to tensor operator
inline bool ToTensorShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TShape &shp = (*in_attrs)[0];
  if (!shp.ndim()) return false;

  CHECK((shp.ndim() == 3) || (shp.ndim() == 4))
      << "Input image must have shape (height, width, channels), or "
      << "(N, height, width, channels) but got " << shp;
  if (shp.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({shp[2], shp[0], shp[1]}));
  } else if (shp.ndim() == 4) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({shp[0], shp[3], shp[1], shp[2]}));
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

template<int req>
struct totensor_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int l, float* out_data, const DType* in_data,
                                  const int c, const int length, const int channel,
                                  const int step, const float normalize_factor = 255.0f) {
      KERNEL_ASSIGN(out_data[step + c*length + l], req,
                    (in_data[step + l*channel + c]) / normalize_factor);
  }
};

template<typename xpu>
void ToTensorImpl(const OpContext &ctx,
                  const std::vector<TBlob> &inputs,
                  const std::vector<TBlob> &outputs,
                  const std::vector<OpReqType> &req,
                  const int length,
                  const int channel,
                  const int step = 0) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      float* output = outputs[0].dptr<float>();
      DType* input = inputs[0].dptr<DType>();

      for (int c = 0; c < channel; ++c) {
        mxnet_op::Kernel<totensor_forward<req_type>, xpu>::Launch(
          s, length, output, input, c, length, channel, step);
      }
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

  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace updates";

  // 3D Input - (h, w, c)
  if (inputs[0].ndim() == 3) {
    const int length = inputs[0].shape_[0] * inputs[0].shape_[1];
    const int channel = inputs[0].shape_[2];
    ToTensorImpl<xpu>(ctx, inputs, outputs, req, length, channel);
  } else if (inputs[0].ndim() == 4) {
    // 4D input (n, h, w, c)
    const int batch_size = inputs[0].shape_[0];
    const int length = inputs[0].shape_[1] * inputs[0].shape_[2];
    const int channel = inputs[0].shape_[3];
    const int step = channel * length;

    #pragma omp parallel for
    for (auto n = 0; n < batch_size; ++n) {
      ToTensorImpl<xpu>(ctx, inputs, outputs, req, length, channel, n*step);
    }
  }
}

}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_TOTENSOR_OP_INL_H_

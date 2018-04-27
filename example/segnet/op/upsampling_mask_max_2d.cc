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
 * \file upsampling_mask_max_2d.cc
 * \brief
 * \author Pengfei Li
*/

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include "./mxnet_op.h"
#include "./upsampling_mask_max_2d-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
void UpSamplingMaskOp<xpu, DType>::upsample_mask_forward(mshadow::Stream<cpu>* s, 
                              const DType* in_data, DType* out_data, int* mask,
                              const TShape& ishape, const TShape& oshape) {
  const index_t in_offset = ishape[2] * ishape[3];
  const index_t out_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < ishape[0]; ++n) {
    for (index_t c = 0; c < ishape[1]; ++c) {
      for (int ph = 0; ph < ishape[2]; ++ph) {
        for (int pw = 0; pw < ishape[3]; ++pw) {
          const int upsample_index = ph * ishape[3] + pw;
          const int max_idx = mask[upsample_index];
          if (max_idx < out_offset) {
            out_data[max_idx] = in_data[upsample_index];
          }
        }
      }
      mask += in_offset;
      in_data += in_offset;
      out_data += out_offset;
    }
  }
}

template<typename xpu, typename DType>
void UpSamplingMaskOp<xpu, DType>::upsample_mask_backward(mshadow::Stream<cpu>* s, 
                              DType* in_grad, const DType* out_grad, const int* mask,
                              const TShape& ishape, const TShape& oshape) {
  const index_t in_offset = ishape[2] * ishape[3];
  const index_t out_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < ishape[0]; ++n) {
    for (index_t c = 0; c < ishape[1]; ++c) {
      for (int ph = 0; ph < ishape[2]; ++ph) {
        for (int pw = 0; pw < ishape[3]; ++pw) {
          const int upsample_index = ph * ishape[3] + pw;
          const int max_idx = mask[upsample_index];
          if (max_idx < out_offset) {
            in_grad[upsample_index] += out_grad[max_idx];
          }
        }
      }
      mask += in_offset;
      in_grad += in_offset;
      out_grad += out_offset;
    }
  }
}

template<>
Operator *CreateOp<cpu>(UpSamplingMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new UpSamplingMaskOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* UpSamplingMaskProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(UpSamplingMaskParam);

MXNET_REGISTER_OP_PROPERTY(UpSamplingMask, UpSamplingMaskProp)
.describe(R"code(Performs max 2d UpSampling on the input.

The shapes for 2-D UpSampling are

- **data**: *(batch_size, channel, height, width)*
- **mask**: *(batch_size, num_filter, out_height, out_width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

Only mask UpSampling is supported.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the UpSamplingMask operator.")
.add_argument("mask", "NDArray-or-Symbol", "Input mask to the UpSamplingMask operator.")
.add_arguments(UpSamplingMaskParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
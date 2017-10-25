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
 * \file pooling_mask_max_2d.cc
 * \brief
 * \author Pengfei Li
*/

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include "./mxnet_op.h"
#include "./pooling_mask_max_2d-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
void PoolingMaskOp<xpu, DType>::pool_mask_forward(mshadow::Stream<cpu>* s, 
                              const DType* in_data, DType* out_data, int* mask,
                              const TShape& ishape, const TShape& oshape,
                              const TShape& kernel, const TShape& stride, const TShape& pad) {
  using mshadow::red::limits::MinValue;
  const int height = ishape[2], width = ishape[3];
  const int pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_data_offset = ishape[2] * ishape[3];
  const index_t out_data_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          DType max_val = MinValue<DType>();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int in_index = h * width + w;
              if (in_data[in_index] > max_val) {
                max_val = in_data[in_index];
                mask[pool_index] = in_index;
              }
            }
          }
          out_data[pool_index] = max_val;
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
      mask += out_data_offset;
    }
  }
}

template<typename xpu, typename DType>
void PoolingMaskOp<xpu, DType>::pool_mask_backward(mshadow::Stream<cpu>* s, 
                              DType* in_grad, const DType* out_grad, const int* mask,
                              const TShape& ishape, const TShape& oshape,
                              const TShape& kernel, const TShape& stride, const TShape& pad) {
  const index_t in_offset = ishape[2] * ishape[3];
  const index_t out_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < oshape[2]; ++ph) {
        for (int pw = 0; pw < oshape[3]; ++pw) {
          const int pool_index = ph * oshape[3] + pw;
          const int max_idx = mask[pool_index];
          in_grad[max_idx] += out_grad[pool_index];
        }
      }
      in_grad += in_offset;
      out_grad += out_offset;
      mask += out_offset;
    }
  }
}

template<>
Operator *CreateOp<cpu>(PoolingMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PoolingMaskOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingMaskProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(PoolingMaskParam);

MXNET_REGISTER_OP_PROPERTY(PoolingMask, PoolingMaskProp)
.describe(R"code(Performs max 2d pooling on the input.

The shapes for 2-D pooling are

- **data**: *(batch_size, channel, height, width)*
- **mask**: *(batch_size, num_filter, out_height, out_width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The definition of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor((x+2*p-k)/s)+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil((x+2*p-k)/s)+1

Only max pooling is supported.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingMaskParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
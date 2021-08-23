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
 * Copyright (c) 2016 by Contributors
 * \file spatial_transformer.cc
 * \brief
 * \author Wei Wu
*/

#include "./spatial_transformer-inl.h"

namespace mshadow {
template<typename DType>
static MSHADOW_CINLINE bool between(const DType value,
                                    const DType lowerBound,
                                    const DType upperBound) {
  return value >= lowerBound && value <= upperBound;
}

template<typename DType>
inline void BilinearSamplingForward(const Tensor<cpu, 4, DType> &output,
                                    const Tensor<cpu, 4, DType> &input,
                                    const Tensor<cpu, 3, DType> grid_src) {
  DType *out = output.dptr_;
  const DType *data = input.dptr_;
  const DType *grid = grid_src.dptr_;
  const index_t o_n = output.size(0), o_c = output.size(1),
    o_h = output.size(2), o_w = output.size(3);
  const index_t i_c = input.size(1), i_h = input.size(2), i_w = input.size(3);
  for (index_t n = 0; n < static_cast<index_t>(o_n); ++n) {
    for (index_t c = 0; c < static_cast<index_t>(o_c); ++c) {
      for (index_t h = 0; h < static_cast<index_t>(o_h); ++h) {
        for (index_t w = 0; w < static_cast<index_t>(o_w); ++w) {
          const index_t out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
          const index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;
          const DType y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          const DType x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
          const auto top_left_y = static_cast<index_t>(std::floor(y_real));
          const auto top_left_x = static_cast<index_t>(std::floor(x_real));
          const DType top_left_y_w = 1.0 - (y_real - top_left_y);
          const DType top_left_x_w = 1.0 - (x_real - top_left_x);
          const index_t data_index = n * i_c * i_h * i_w + c * i_h * i_w +
                                 top_left_y * i_w + top_left_x;
          DType top_left_v = 0;
          DType top_right_v = 0;
          DType bottom_left_v = 0;
          DType bottom_right_v = 0;
          index_t lower_bound = 0;
          if (between(top_left_x, lower_bound, i_w-1) &&
              between(top_left_y, lower_bound, i_h-1))
            top_left_v = *(data + data_index);
          if (between(top_left_x + 1, lower_bound, i_w-1) &&
              between(top_left_y, lower_bound, i_h-1))
            top_right_v = *(data + data_index + 1);
          if (between(top_left_x, lower_bound, i_w-1) &&
              between(top_left_y + 1, lower_bound, i_h-1))
            bottom_left_v = *(data + data_index + i_w);
          if (between(top_left_x+1, lower_bound, i_w-1) &&
              between(top_left_y + 1, lower_bound, i_h-1))
            bottom_right_v = *(data + data_index + i_w + 1);
          *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                             top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                             bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                             bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
        }
      }
    }
  }
}

template<typename DType>
inline void BilinearSamplingBackward(const Tensor<cpu, 4, DType> &input_grad,
                                     const Tensor<cpu, 3, DType> &grid_src_data,
                                     const Tensor<cpu, 4, DType> &output_grad,
                                     const Tensor<cpu, 4, DType> &input_data) {
  DType *g_input = input_grad.dptr_;
  DType *grid_src = grid_src_data.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  const index_t o_n = output_grad.size(0), o_c = output_grad.size(1),
    o_h = output_grad.size(2), o_w = output_grad.size(3);
  const index_t i_c = input_data.size(1), i_h = input_data.size(2), i_w = input_data.size(3);
  for (index_t n = 0; n < static_cast<index_t>(o_n); ++n) {
     for (index_t h = 0; h < static_cast<index_t>(o_h); ++h) {
        for (index_t w = 0; w < static_cast<index_t>(o_w); ++w) {
          DType top_left_y_gw = 0.0;
          DType top_left_x_gw = 0.0;
          const index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
          const DType y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          const DType x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;
          const auto top_left_y = static_cast<index_t>(std::floor(y_real));
          const auto top_left_x = static_cast<index_t>(std::floor(x_real));
          const DType top_left_y_w = 1.0 - (y_real - top_left_y);
          const DType top_left_x_w = 1.0 - (x_real - top_left_x);
          for (index_t c = 0; c < static_cast<index_t>(o_c); ++c) {
            index_t grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
            const index_t data_index = n * i_c * i_h * i_w + c * i_h * i_w +
                                   top_left_y * i_w + top_left_x;
            // calc 4 vertex value in input data
            DType top_left_v = 0;
            DType top_right_v = 0;
            DType bottom_left_v = 0;
            DType bottom_right_v = 0;
            index_t lower_bound = 0;
            if (between(top_left_x, lower_bound, i_w-1) &&
                between(top_left_y, lower_bound, i_h-1)) {
              *(g_input + data_index) += *(grad + grad_index) * top_left_y_w * top_left_x_w;
              top_left_v = *(data + data_index);
            }
            if (between(top_left_x+1, lower_bound, i_w-1) &&
                between(top_left_y, lower_bound, i_h-1)) {
              *(g_input + data_index + 1) += *(grad + grad_index) * top_left_y_w
                                             * (1.0 - top_left_x_w);
              top_right_v = *(data + data_index + 1);
            }
            if (between(top_left_x, lower_bound, i_w-1) &&
                between(top_left_y+1, lower_bound, i_h-1)) {
              *(g_input + data_index+ i_w) += *(grad + grad_index) * (1.0 - top_left_y_w)
                                              * top_left_x_w;
              bottom_left_v = *(data + data_index + i_w);
            }
            if (between(top_left_x+1, lower_bound, i_w-1) &&
                between(top_left_y+1, lower_bound, i_h-1)) {
              *(g_input + data_index+ i_w + 1) += *(grad + grad_index) * (1.0 - top_left_y_w)
                                                  * (1.0 - top_left_x_w);
              bottom_right_v = *(data + data_index + i_w + 1);
            }
            // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
            top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                             (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                             * top_left_x_w);
            top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v +
                             (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                             * top_left_y_w);
          }
          // calc grid_src grad
          *(grid_src + grid_src_index + o_h * o_w) = top_left_y_gw * (i_h - 1) / 2;
          *(grid_src + grid_src_index) = top_left_x_gw * (i_w - 1) / 2;
        }
      }
    }
  }

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(SpatialTransformerParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SpatialTransformerOp<cpu, DType>(param);
  })
  return op;
}

Operator *SpatialTransformerProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SpatialTransformerParam);

MXNET_REGISTER_OP_PROPERTY(SpatialTransformer, SpatialTransformerProp)
.add_argument("data", "NDArray-or-Symbol",
              "Input data to the SpatialTransformerOp.")
.add_argument("loc", "NDArray-or-Symbol",
              "localisation net, the output dim should be 6 when transform_type "
              "is affine. You shold initialize the weight and bias with identity tranform.")
.add_arguments(SpatialTransformerParam::__FIELDS__())
.describe("Applies a spatial transformer to input feature map.");

}  // namespace op
}  // namespace mxnet

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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_psroi_pooling.cc
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
#include "./deformable_psroi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
  template<typename DType>
  inline void DeformablePSROIPoolForward(const Tensor<cpu, 4, DType> &out,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 2, DType> &bbox,
    const Tensor<cpu, 4, DType> &trans,
    const Tensor<cpu, 4, DType> &top_count,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // NOT_IMPLEMENTED;
    return;
  }

  template<typename DType>
  inline void DeformablePSROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
    const Tensor<cpu, 4, DType> &trans_grad,
    const Tensor<cpu, 4, DType> &out_grad,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 2, DType> &bbox,
    const Tensor<cpu, 4, DType> &trans,
    const Tensor<cpu, 4, DType> &top_count,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // NOT_IMPLEMENTED;
    return;
  }
}  // namespace mshadow

namespace mxnet {
namespace op {

  template<>
  Operator *CreateOp<cpu>(DeformablePSROIPoolingParam param, int dtype) {
    Operator* op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeformablePSROIPoolingOp<cpu, DType>(param);
    });
    return op;
  }

  Operator *DeformablePSROIPoolingProp::CreateOperatorEx(
    Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
  }

  DMLC_REGISTER_PARAMETER(DeformablePSROIPoolingParam);

  MXNET_REGISTER_OP_PROPERTY(_contrib_DeformablePSROIPooling, DeformablePSROIPoolingProp)
    .describe("Performs deformable position-sensitive region-of-interest pooling on inputs.\n"
      "The DeformablePSROIPooling operation is described in https://arxiv.org/abs/1703.06211 ."
      "batch_size will change to the number of region bounding boxes after DeformablePSROIPooling")
    .add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
    .add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
      "[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
      "of designated region of interest. batch_index indicates the index of corresponding image "
      "in the input data")
    .add_argument("trans", "Symbol", "transition parameter")
    .add_arguments(DeformablePSROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

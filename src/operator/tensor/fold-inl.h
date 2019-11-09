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
 * Copyright (c) 2019 by Contributors
 * \file fold-inl.h
 * \brief CPU implementation of unfold operator
 * \author Istvan Fehervari
*/
#ifndef MXNET_OPERATOR_TENSOR_FOLD_INL_H_
#define MXNET_OPERATOR_TENSOR_FOLD_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include "../operator_common.h"
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct UnfoldParam : public dmlc::Parameter<UnfoldParam> {
  int dim;
  int kernel_size;
  uint32_t stride;
  DMLC_DECLARE_PARAMETER(UnfoldParam) {
    DMLC_DECLARE_FIELD(dim)
      .set_default(-1)
      .describe("Dimension to unfold");
    DMLC_DECLARE_FIELD(kernel_size)
      .set_lower_bound(1)
      .describe("Size of each unfolded block");
    DMLC_DECLARE_FIELD(stride)
      .set_lower_bound(1)
      .set_default(1)
      .describe("Stride of the sliding window");
  }
};  // struct UnfoldParam

inline mxnet::TShape UnfoldShapeImpl(const mxnet::TShape& ishape, const int k,
                            const int32_t dim, const int32_t stride) {
  int32_t axis = CheckAxis(dim, ishape.ndim());

  auto elems_target_dim = ishape[axis];

  CHECK_LE(k, elems_target_dim) << "kernel size " << k << " must be less than or equal"
                             " to the number of elements in the target dimension";

  int num_windows = (elems_target_dim - k) / stride + 1;

  auto o_ndim = ishape.ndim() + 1;
  mxnet::TShape oshape(o_ndim, -1);

  for (auto i = 0; i < ishape.ndim(); i++) {
    oshape[i] = ishape[i];
  }

  oshape[o_ndim - 1] = k;
  oshape[axis] = num_windows;
  return oshape;
}

inline bool UnfoldOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

inline bool UnfoldOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  const UnfoldParam& param = nnvm::get<UnfoldParam>(attrs.parsed);

  mxnet::TShape oshape = UnfoldShapeImpl(ishape,
                                  param.kernel_size,
                                  param.dim,
                                  param.stride);
  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Failed to infer shape for unfold.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

template<int req>
struct unfold {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t idx, DType* out, const DType* in,
                                  mshadow::Shape<4> oshape,
                                  mshadow::Shape<3> ishape,
                                  index_t stride) {
    using namespace mxnet_op;

    auto cloc = unravel(idx, oshape);
    auto num_slice = cloc[1];
    auto num_element = cloc[3];
    auto ipos = (num_slice * stride) + num_element;
    auto j = ravel(Shape3(cloc[0], ipos, cloc[2]), ishape);

    KERNEL_ASSIGN(out[idx], req, in[j]);
  }
};

template<int req>
struct unfold_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t idx, DType* out, const DType* in,
                                  mshadow::Shape<3> oshape,
                                  mshadow::Shape<4> ishape,
                                  index_t stride, index_t kernel_size) {
    using namespace mxnet_op;

    auto cloc = unravel(idx, oshape);
    DType summed = 0.0;

    for (index_t e=0; e < kernel_size; e++) {
        auto p = (cloc[1] - e) / stride;
        if (p < ishape[1]) {
          index_t j = ravel(Shape4(cloc[0], p, cloc[2], e), ishape);
          summed += in[j];
      }
    }

    KERNEL_ASSIGN(out[idx], req, summed);
  }
};

template<typename xpu>
void UnfoldOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const UnfoldParam& param = nnvm::get<UnfoldParam>(attrs.parsed);

  uint32_t idim = ishape.ndim();
  uint32_t tdim = CheckAxis(param.dim, ishape.ndim());

  index_t leading = 1,
          trailing = 1,
          ibody = ishape[tdim],
          obody = oshape[tdim];

  for (uint32_t i = 0; i < tdim; ++i) {
      leading *= ishape[i];
  }

  for (uint32_t i = tdim + 1; i < idim; ++i) {
      trailing *= ishape[i];
  }

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<unfold<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                              in_data.dptr<DType>(), Shape4(leading, obody, trailing,
                              param.kernel_size), Shape3(leading, ibody, trailing), param.stride);
      });
  });
}

template<typename xpu>
void UnfoldOpBackward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const UnfoldParam& param = nnvm::get<UnfoldParam>(attrs.parsed);

  uint32_t odim = oshape.ndim();
  uint32_t tdim = CheckAxis(param.dim, oshape.ndim());

  index_t leading = 1,
          trailing = 1,
          ibody = ishape[tdim],
          obody = oshape[tdim];

  for (uint32_t i = 0; i < tdim; ++i) {
      leading *= oshape[i];
  }

  for (uint32_t i = tdim + 1; i < odim; ++i) {
      trailing *= oshape[i];
  }

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<unfold_backward<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(),
                              in_data.dptr<DType>(), Shape3(leading, obody, trailing),
                              Shape4(leading, ibody, trailing, param.kernel_size),
                              param.stride, param.kernel_size);
      });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_FOLD_INL_H_

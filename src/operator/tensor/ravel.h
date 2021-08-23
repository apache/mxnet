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
 * \file ravel.h
 * \brief Operators for ravel/unravel of indices.
 */
#ifndef MXNET_OPERATOR_TENSOR_RAVEL_H_
#define MXNET_OPERATOR_TENSOR_RAVEL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct RavelParam : public dmlc::Parameter<RavelParam> {
  mxnet::TShape shape;
  DMLC_DECLARE_PARAMETER(RavelParam) {
    DMLC_DECLARE_FIELD(shape)
      .set_default(mxnet::TShape())
      .describe("Shape of the array into which the multi-indices apply.");
  }
};

inline bool RavelOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const mxnet::TShape& shape = nnvm::get<RavelParam>(attrs.parsed).shape;
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_GT(shape.ndim(), 0) << "Empty shape parameter for ravel operator.";
  if ((*in_attrs)[0].ndim() > 0) {
    CHECK_EQ((*in_attrs)[0].ndim(), 2)
      << "Input to ravel operator must be two-dimensional.";
    CHECK_EQ((*in_attrs)[0][0], shape.ndim())
      << "First dimension of input of ravel operator does not match shape parameter dimension.";
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape1((*in_attrs)[0][1]));
    return true;
  }
  if ((*out_attrs)[0].ndim() > 0) {
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, Shape2(shape.ndim(), (*out_attrs)[0][0]));
    return true;
  }
  return false;
}

inline bool UnravelOpShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const mxnet::TShape& shape = nnvm::get<RavelParam>(attrs.parsed).shape;
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_GT(shape.ndim(), 0) << "Empty shape parameter for unravel operator.";
  const mxnet::TShape &in_shape = (*in_attrs)[0];
  if (in_shape.ndim() > 0) {
    mxnet::TShape out_shape(in_shape.ndim() + 1, -1);
    out_shape[0] = shape.ndim();
    for (int i = 0; i < in_shape.ndim(); ++i) {
      out_shape[i+1] = in_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
    return true;
  }
  if ((*out_attrs)[0].ndim() > 0) {
    const mxnet::TShape &out_shape = (*out_attrs)[0];
    CHECK_EQ((*out_attrs)[0].ndim(), 2)
      << "Output of unravel operator must be two-dimensional.";
    CHECK_EQ((*out_attrs)[0][0], shape.ndim())
      << "First dimension of output of ravel operator does not match shape parameter dimension.";
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, mxnet::TShape(
          out_shape.data() + 1, out_shape.data() + out_shape.ndim()));
    return true;
  }
  return false;
}

struct ravel_index {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, index_t N, index_t ndim, index_t *shape,
                                  DType *unravelled, DType *ravelled) {
    index_t ret = 0;
    #pragma unroll
    for (index_t j = 0; j < ndim; ++j) {
      ret = ret * shape[j] + unravelled[i+j*N];
    }
    ravelled[i] = ret;
  }
};

struct unravel_index {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, index_t N, index_t ndim, index_t *shape,
                                  DType *unravelled, DType *ravelled) {
    index_t idx(ravelled[i]);
    #pragma unroll
    for (int j = ndim-1; j > 0; --j) {
      index_t tmp = idx / shape[j];
      unravelled[i+j*N] = idx - tmp*shape[j];
      idx = tmp;
    }
    unravelled[i] = idx;
  }
};

template<typename xpu>
void RavelForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& shape = nnvm::get<RavelParam>(attrs.parsed).shape;
  std::vector<index_t> buffer(shape.data(), shape.data()+shape.ndim());
  Tensor<xpu, 1, index_t> work
    = ctx.requested[0].get_space_typed<xpu, 1, index_t>(Shape1(shape.ndim()), s);
  Copy(work, Tensor<cpu, 1, index_t>(&buffer[0], Shape1(buffer.size()), 0), s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    Tensor<xpu, 1, OType> in = inputs[0].FlatTo1D<xpu, OType>(s);
    Tensor<xpu, 1, OType> out = outputs[0].FlatTo1D<xpu, OType>(s);
    mxnet_op::Kernel<ravel_index, xpu>::Launch(s, out.size(0), out.size(0), in.size(0)/out.size(0),
                                               work.dptr_, in.dptr_, out.dptr_);
  });
}

template<typename xpu>
void UnravelForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape& shape = nnvm::get<RavelParam>(attrs.parsed).shape;
  std::vector<index_t> buffer(shape.data(), shape.data()+shape.ndim());
  Tensor<xpu, 1, index_t> work
    = ctx.requested[0].get_space_typed<xpu, 1, index_t>(Shape1(shape.ndim()), s);
  Copy(work, Tensor<cpu, 1, index_t>(&buffer[0], Shape1(buffer.size()), 0), s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    Tensor<xpu, 1, OType> in = inputs[0].FlatTo1D<xpu, OType>(s);
    Tensor<xpu, 1, OType> out = outputs[0].FlatTo1D<xpu, OType>(s);
    mxnet_op::Kernel<unravel_index, xpu>::Launch(
        s, in.shape_.Size(), in.shape_.Size(), shape.ndim(),
        work.dptr_, out.dptr_, in.dptr_);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_RAVEL_H_

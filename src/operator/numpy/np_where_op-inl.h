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
 * Copyright (c) 2017 by Contributors
 * \file np_where_op.cc
 * \brief Function definition of numpy operator where
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../common/utils.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

#define NUMPY_WHERE_MAX_DIM 5

using namespace mshadow;

template<int ndim>
struct numpy_where_kernel {
  template<typename CType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, OpReqType req, const Shape<ndim> &cstride,
                                  const Shape<ndim> &xstride, const Shape<ndim> &ystride,
                                  const Shape<ndim> &oshape, CType *datac, DType *datax,
                                  DType *datay, DType *out) {
    Shape<ndim> coord = mxnet_op::unravel(base, oshape);
    auto cidx = static_cast<index_t>(mxnet_op::dot(coord, cstride));
    auto xidx = static_cast<index_t>(mxnet_op::dot(coord, xstride));
    auto yidx = static_cast<index_t>(mxnet_op::dot(coord, ystride));
    KERNEL_ASSIGN(out[base], req, datac[cidx] != CType(0) ? datax[xidx] : datay[yidx]);
  }
};

template<int ndim, bool is_left>
struct numpy_where_backward_kernel {
  template<typename CType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, OpReqType req, const Shape<ndim> &cstride,
                                  const Shape<ndim> &oshape, CType *datac, DType *datao, DType *grad) {
    Shape<ndim> coord = mxnet_op::unravel(base, oshape);
    auto cidx = static_cast<index_t>(mxnet_op::dot(coord, cstride));
    if (is_left) {
      KERNEL_ASSIGN(grad[base], req, datac[cidx] != CType(0) ? datao[base] : DType(0));
    } else {
      KERNEL_ASSIGN(grad[base], req, datac[cidx] == CType(0) ? datao[base] : DType(0));
    }
  }
};

inline bool NumpyWhereOpShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& operand1 = (*in_attrs)[0];
  mxnet::TShape& operand2 = (*in_attrs)[1];
  mxnet::TShape& operand3 = (*in_attrs)[2];

  if (operand1 == operand2 && operand2 == operand3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, operand1);
    return shape_is_known(out_attrs->at(0));
  }
  mxnet::TShape out(std::max({operand1.ndim(), operand2.ndim(), operand3.ndim()}), -1);
  const int b1 = out.ndim() - operand1.ndim();
  const int b2 = out.ndim() - operand2.ndim();
  const int b3 = out.ndim() - operand3.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int s1 = 1, s2 = 1, s3 = 1;
    if (i >= b1) s1 = operand1[i-b1];
    if (i >= b2) s2 = operand2[i-b2];
    if (i >= b3) s3 = operand3[i-b3];
    if (!(s1 == s2 && s2 == s3)) {
      CHECK((s1 == 1 && s2 == 1) || (s1 == 1 && s3 == 1) || (s2 == 1 && s3 == 1) ||
            (s1 == 1 && s2 == s3) || (s2 == 1 && s1 == s3) || (s3 == 1 && s1 == s2))
        << "Operands could not be broadcast together.";
      out[i] = std::max({s1, s2, s3});
    } else {
      out[i] = s1;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return shape_is_known(out);
}

inline bool NumpyWhereOpType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->at(1), in_attrs->at(2));
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  return (out_attrs->at(0) != -1);
}

template<typename xpu>
inline void NumpyWhereOpForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  CHECK_LE(outputs[0].shape_.ndim(), NUMPY_WHERE_MAX_DIM);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<Shape<NUMPY_WHERE_MAX_DIM>> in_strides;
  in_strides.resize(3);
  for (int i = 0; i < 3; ++i) {
    TShape expanded_ishape(NUMPY_WHERE_MAX_DIM, 1);
    const TShape& ishape = inputs[i].shape_;
    const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
    for (int j = 0; j < ishape.ndim(); ++j) {
      expanded_ishape[j + ndim_delta] = ishape[j];
    }
    in_strides[i] = mxnet_op::calc_stride(expanded_ishape.get<NUMPY_WHERE_MAX_DIM>());
  }
  TShape expanded_oshape(NUMPY_WHERE_MAX_DIM, 1);
  const int ndim_delta = expanded_oshape.ndim() - outputs[0].shape_.ndim();
  for (int j = 0; j < outputs[0].shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (outputs[0].shape_)[j];
  }
  Shape<NUMPY_WHERE_MAX_DIM> oshape = expanded_oshape.get<NUMPY_WHERE_MAX_DIM>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, CType, {
      mxnet_op::Kernel<numpy_where_kernel<NUMPY_WHERE_MAX_DIM>, xpu>::Launch(
        s, outputs[0].Size(), req[0],
        in_strides[0], in_strides[1], in_strides[2], oshape,
        inputs[0].dptr<CType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(), outputs[0].dptr<DType>());
    });
  });
}

template<typename xpu>
inline void NumpyWhereOpBackward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK(common::is_float(inputs[0].type_flag_)) << "Backward only supports float types!";
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // get expanded oshape
  TShape expanded_oshape(NUMPY_WHERE_MAX_DIM, 1);
  int ndim_delta = expanded_oshape.ndim() - inputs[0].shape_.ndim();
  for (int j = 0; j < inputs[0].shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (inputs[0].shape_)[j];
  }
  Shape<NUMPY_WHERE_MAX_DIM> oshape = expanded_oshape.get<NUMPY_WHERE_MAX_DIM>();
  // get cond stride
  TShape expanded_cshape(NUMPY_WHERE_MAX_DIM, 1);
  ndim_delta = expanded_cshape.ndim() - inputs[1].shape_.ndim();
  for (int j = 0; j < inputs[1].shape_.ndim(); ++j) {
    expanded_cshape[j + ndim_delta] = (inputs[1].shape_)[j];
  }
  Shape<NUMPY_WHERE_MAX_DIM> cstride = mxnet_op::calc_stride(expanded_cshape.get<NUMPY_WHERE_MAX_DIM>());
  // get expanded lshape
  TShape expanded_lshape(NUMPY_WHERE_MAX_DIM, 1);
  ndim_delta = expanded_lshape.ndim() - outputs[0].shape_.ndim();
  for (int j = 0; j < outputs[0].shape_.ndim(); ++j) {
    expanded_lshape[j + ndim_delta] = (outputs[0].shape_)[j];
  }
  // get expanded rshape
  TShape expanded_rshape(NUMPY_WHERE_MAX_DIM, 1);
  ndim_delta = expanded_rshape.ndim() - outputs[1].shape_.ndim();
  for (int j = 0; j < outputs[1].shape_.ndim(); ++j) {
    expanded_rshape[j + ndim_delta] = (outputs[1].shape_)[j];
  }

  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[1].type_flag_, CType, {
      Tensor<xpu, 1, char> largespace;
      Tensor<xpu, NUMPY_WHERE_MAX_DIM, DType> workspace;
      size_t ws_size;
      if (!(inputs[0].shape_ != outputs[0].shape_) || !(inputs[0].shape_ != outputs[1].shape_)) {
        size_t ws_size1 = broadcast::ReduceWorkspaceSize<NUMPY_WHERE_MAX_DIM, DType>(
            s, expanded_lshape, req[0], expanded_oshape);
        size_t ws_size2 = broadcast::ReduceWorkspaceSize<NUMPY_WHERE_MAX_DIM, DType>(
            s, expanded_rshape, req[1], expanded_oshape);
        ws_size = std::max(ws_size1, ws_size2);
      }
      // process left output
      if (inputs[0].shape_ == outputs[0].shape_) {
        mxnet_op::Kernel<numpy_where_backward_kernel<NUMPY_WHERE_MAX_DIM, true>, xpu>::Launch(
          s, inputs[0].Size(), req[0], cstride, oshape,
          inputs[1].dptr<CType>(), inputs[0].dptr<DType>(), outputs[0].dptr<DType>());
      } else {
        largespace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(inputs[0].shape_.Size() * sizeof(DType) + ws_size), s);
        workspace = Tensor<xpu, NUMPY_WHERE_MAX_DIM, DType>(
            reinterpret_cast<DType*>(largespace.dptr_ + ws_size), expanded_oshape.get<NUMPY_WHERE_MAX_DIM>(), s);
        mxnet_op::Kernel<numpy_where_backward_kernel<NUMPY_WHERE_MAX_DIM, true>, xpu>::Launch(
          s, inputs[0].Size(), req[0], cstride, oshape,
          inputs[1].dptr<CType>(), inputs[0].dptr<DType>(), workspace.dptr_);
        if (NeedSafeAcc<true>(outputs[0].type_flag_, outputs[0].type_flag_)) {
          ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
              ctx, {TBlob(workspace)}, {req[0]}, {outputs[0].reshape(expanded_lshape)}, expanded_lshape);
        } else {
          ReduceAxesComputeImpl<xpu, mshadow_op::sum, false>(
              ctx, {TBlob(workspace)}, {req[0]}, {outputs[0].reshape(expanded_lshape)}, expanded_lshape);
        }
      }
      // process right output
      if (inputs[0].shape_ == outputs[1].shape_) {
        mxnet_op::Kernel<numpy_where_backward_kernel<NUMPY_WHERE_MAX_DIM, false>, xpu>::Launch(
          s, inputs[0].Size(), req[1], cstride, oshape,
          inputs[1].dptr<CType>(), inputs[0].dptr<DType>(), outputs[1].dptr<DType>());
      } else {
        largespace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(inputs[0].shape_.Size() * sizeof(DType) + ws_size), s);
        workspace = Tensor<xpu, NUMPY_WHERE_MAX_DIM, DType>(
            reinterpret_cast<DType*>(largespace.dptr_ + ws_size), expanded_oshape.get<NUMPY_WHERE_MAX_DIM>(), s);
        mxnet_op::Kernel<numpy_where_backward_kernel<NUMPY_WHERE_MAX_DIM, false>, xpu>::Launch(
          s, inputs[0].Size(), req[1], cstride, oshape,
          inputs[1].dptr<CType>(), inputs[0].dptr<DType>(), workspace.dptr_);
        if (NeedSafeAcc<true>(outputs[1].type_flag_, outputs[1].type_flag_)) {
          ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
              ctx, {TBlob(workspace)}, {req[1]}, {outputs[1].reshape(expanded_rshape)}, expanded_rshape);
        } else {
          ReduceAxesComputeImpl<xpu, mshadow_op::sum, false>(
              ctx, {TBlob(workspace)}, {req[1]}, {outputs[1].reshape(expanded_rshape)}, expanded_rshape);
        }
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_

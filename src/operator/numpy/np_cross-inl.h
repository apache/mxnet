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
 *  Copyright (c) 2020 by Contributors
 * \file np_cross-inl.h
 * \brief Function definition of cross product of two (arrays of) vectors
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct NumpyCrossParam : public dmlc::Parameter<NumpyCrossParam> {
  int axisa, axisb, axisc;
  DMLC_DECLARE_PARAMETER(NumpyCrossParam) {
    DMLC_DECLARE_FIELD(axisa)
    .set_default(-1)
    .describe("Axis of `a` that defines the vector(s).  By default, the last axis.");
    DMLC_DECLARE_FIELD(axisb)
    .set_default(-1)
    .describe("Axis of `b` that defines the vector(s).  By default, the last axis.");
    DMLC_DECLARE_FIELD(axisc)
    .set_default(-1)
    .describe("Axis of `c` containing the cross product vector(s)."
              "Ignored if both input vectors have dimension 2, as the return is scalar."
              "By default, the last axis.");
  }
};

// Get moveaxis index.
inline mxnet::Tuple<int> GetMoveaxisIndex(const int& source,
                                          const int& destination,
                                          const mxnet::TShape& shape) {
  const int ndim = shape.ndim();
  const int src_axis = CheckAxis(source, ndim);
  const int dest_axis = CheckAxis(destination, ndim);
  std::vector<int> moveaxis_index_vec;
  for (int i = 0; i < ndim; ++i) {
    if (i != src_axis) { moveaxis_index_vec.push_back(i); }
  }
  moveaxis_index_vec.insert(moveaxis_index_vec.begin() + dest_axis, src_axis);
  return mxnet::Tuple<int>(moveaxis_index_vec);
}

// Get moveaxis shape.
inline mxnet::TShape GetMoveaxisShape(const Tuple<int>& moveaxis_index,
                                      const mxnet::TShape& org_shape) {
  const int ndim = org_shape.ndim();
  if (ndim == 0) { return mxnet::TShape(0, 0); }
  CHECK_EQ(moveaxis_index.ndim(), org_shape.ndim()) << "moveaxis index dismatch original shape.";
  std::vector<int> moveaxis_shape_vec(ndim, -1);
  for (int i = 0; i < ndim; ++i) {
    moveaxis_shape_vec[i] = org_shape[moveaxis_index[i]];
  }
  return mxnet::TShape(moveaxis_shape_vec.begin(), moveaxis_shape_vec.end());
}

// Get or check broadcast shape for cross product.
inline void GetOrCheckLRShape(const nnvm::NodeAttrs& attrs,
                              const mxnet::TShape& a_moveaxis_shape,
                              const mxnet::TShape& b_moveaxis_shape,
                              mxnet::TShape *c_shape_ptr = nullptr) {
  const int a_ndim = a_moveaxis_shape.ndim();
  const int b_ndim = b_moveaxis_shape.ndim();
  mxnet::TShape a_cutoff_shape(a_ndim - 1, -1);
  mxnet::TShape b_cutoff_shape(b_ndim - 1, -1);
  for (int i = 0; i < a_ndim - 1; ++i) {
    a_cutoff_shape[i] = a_moveaxis_shape[i];
  }
  for (int i = 0; i < b_ndim - 1; ++i) {
    b_cutoff_shape[i] = b_moveaxis_shape[i];
  }
  mxnet::ShapeVector in_shape_vec({ a_cutoff_shape, b_cutoff_shape});
  mxnet::ShapeVector out_shape_vec({ mxnet::TShape() });
  mxnet::op::BinaryBroadcastShape(attrs, &in_shape_vec, &out_shape_vec);
  if (c_shape_ptr && (a_moveaxis_shape[a_ndim - 1] == 3 || b_moveaxis_shape[b_ndim - 1] == 3)) {
    mxnet::TShape c_shape(out_shape_vec[0].ndim() + 1, -1);
    for (int i = 0; i < c_shape.ndim() - 1; ++i) {
      c_shape[i] = out_shape_vec[0][i];
    }
    c_shape[c_shape.ndim() - 1] = 3;
    *c_shape_ptr = c_shape;
  } else {
    *c_shape_ptr = out_shape_vec[0];
  }
}

// Get data[..., 0] shape.
inline mxnet::TShape GetCutoffShape(const mxnet::TShape& shape) {
  if (shape.ndim() == 0 || !ndim_is_known(shape)) { return mxnet::TShape(0, 0); }
  mxnet::TShape cutoff_shape(shape.ndim() - 1, -1);
  for (int i = 0; i < shape.ndim() - 1; ++i) { cutoff_shape[i] = shape[i]; }
  return cutoff_shape;
}

struct CrossInAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_ptr, DType *out_ptr,
                                  const int stride, const int index, const int msize) {
    if (index < stride && i * stride + index < msize) {
      out_ptr[i] = in_ptr[i * stride + index];
    }
  }
};

template<int req>
struct CrossOutAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_ptr, DType *out_ptr,
                                  const int positive, const int stride,
                                  const int index, const int msize) {
    if (index < stride && i * stride + index < msize) {
      KERNEL_ASSIGN(out_ptr[i * stride + index], req, positive == 1 ? in_ptr[i] : -in_ptr[i]);
    }
  }
};

template<int req>
struct ResAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<typename DType>
inline size_t AligndWorkspaceSize(const size_t& offset,
                                  const size_t& add_size) {
  size_t m32 = 32;
  size_t wks = offset + add_size;
  return ((wks * sizeof(DType) + m32 - 1) / m32 * m32 + sizeof(DType) - 1) / sizeof(DType);
}

// Calculate workspace size for numpy cross forward op.
template<typename xpu, typename DType>
inline size_t NumpyCrossWorkspaceSize(const mxnet::TShape& a_moveaxis_shape,
                                      const mxnet::TShape& b_moveaxis_shape,
                                      const mxnet::TShape& c_moveaxis_shape,
                                      const mxnet::TShape& c_shape,
                                      const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<OpReqType>& req) {
  if (kNullOp == req[0]) { return 0U; }
  // Zero-size input, no need to launch kernel
  if (0U == a_moveaxis_shape.Size() || 0U == b_moveaxis_shape.Size()) { return 0U; }

  size_t workspace_size = 0;
  const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
  const int a_ndim = a_moveaxis_shape.ndim();
  const int b_ndim = b_moveaxis_shape.ndim();
  const int c_ndim = c_moveaxis_shape.ndim();
  const int a_axis = CheckAxis(param.axisa, a_ndim);
  const int b_axis = CheckAxis(param.axisb, b_ndim);

  if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
    if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
      // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
      workspace_size += a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      workspace_size += b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      workspace_size += c_shape.Size();
    } else {
      // Case 2, 3, 4: a.shape[-1] == 3 or b.shape[-1] == 3, param.axisc is not ignored.
      workspace_size += a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      workspace_size += b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      workspace_size += c_moveaxis_shape.Size();
      workspace_size += 3 * c_moveaxis_shape.ProdShape(0, c_ndim - 1);
    }
    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      workspace_size += a_moveaxis_shape.Size();
      workspace_size += b_moveaxis_shape.Size();
    }
  } else {
    if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
      // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  a_moveaxis_shape.ProdShape(0, a_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  b_moveaxis_shape.ProdShape(0, b_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  c_shape.Size());
    } else {
      // Case 2, 3, 4: a.shape[-1] == 3 or b.shape[-1] == 3, param.axisc is not ignored.
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  a_moveaxis_shape.ProdShape(0, a_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  b_moveaxis_shape.ProdShape(0, b_ndim - 1));
      for (int i = 0; i < 3; ++i) {
        workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                    c_moveaxis_shape.ProdShape(0, c_ndim - 1));
      }
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  c_moveaxis_shape.Size());
    }
    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  a_moveaxis_shape.Size());
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  b_moveaxis_shape.Size());
    }
  }
  return workspace_size;
}

template<typename xpu, typename DType, int a_dim, int b_dim>
struct NumpyCrossForwardImpl {
  static void op(const TBlob& a,
                 const TBlob& b,
                 const TBlob& c,
                 const std::vector<Tuple<int> >& moveaxis_index_vec,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req,
                 const Tensor<xpu, 1, DType>& workspace) {
    CHECK(a_dim == 3 || b_dim == 3)
      << "no specialized NumpyCrossOp defined for template parameters.";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
    const Tuple<int>& a_moveaxis_index = moveaxis_index_vec[0];
    const Tuple<int>& b_moveaxis_index = moveaxis_index_vec[1];
    const mxnet::TShape& a_moveaxis_shape = moveaxis_shape_vec[0];
    const mxnet::TShape& b_moveaxis_shape = moveaxis_shape_vec[1];
    const mxnet::TShape& c_moveaxis_shape = moveaxis_shape_vec[2];
    const int a_ndim = a_moveaxis_shape.ndim();
    const int b_ndim = b_moveaxis_shape.ndim();
    const int c_ndim = b_moveaxis_shape.ndim();
    const int a_axis = CheckAxis(param.axisa, a_ndim);
    const int b_axis = CheckAxis(param.axisb, b_ndim);
    const int c_axis = CheckAxis(param.axisc, c_ndim);
    CHECK_EQ(c_moveaxis_shape[c_ndim - 1], 3)
      << "no specialized NumpyCrossOp defined for template parameters.";

    TBlob aw_data, bw_data, c_data, cw_data, a_data, b_data;
    std::vector<TBlob> cw_data_vec;
    if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
      // Allocate workspace in cpu, no need to align address.
      DType *aw_ptr = workspace.dptr_;
      DType *bw_ptr = aw_ptr + a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      DType *cw_ptr = bw_ptr + b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      DType *c_ptr = cw_ptr + 3 * c_moveaxis_shape.ProdShape(0, c_ndim - 1);
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        DType *a_ptr = c_ptr + c_moveaxis_shape.Size();
        DType *b_ptr = a_ptr + a_moveaxis_shape.Size();
        a_data = TBlob(a_ptr, a_moveaxis_shape, a.dev_mask(), a.dev_id());
        b_data = TBlob(b_ptr, b_moveaxis_shape, b.dev_mask(), b.dev_id());
        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                           mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                           mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
      aw_data = TBlob(aw_ptr, GetCutoffShape(a_moveaxis_shape), a.dev_mask(), a.dev_id());
      bw_data = TBlob(bw_ptr, GetCutoffShape(b_moveaxis_shape), b.dev_mask(), b.dev_id());
      cw_data = TBlob(cw_ptr, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      c_data = TBlob(c_ptr, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      for (int i = 0; i < 3; ++i) {
        cw_data_vec.push_back(TBlob(cw_ptr + i * c_moveaxis_shape.ProdShape(0, c_ndim - 1),
                                    GetCutoffShape(c_moveaxis_shape), c.dev_mask(), c.dev_id()));
      }
    } else {
      // Allocate workspace in cpu, need to align address.
      size_t offset = 0;
      aw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(a_moveaxis_shape),
                      a.dev_mask(), a.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, aw_data.shape_.Size());

      bw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(b_moveaxis_shape),
                      b.dev_mask(), b.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, bw_data.shape_.Size());

      cw_data = TBlob(workspace.dptr_ + offset, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      for (int i = 0; i < 3; ++i) {
        cw_data_vec.push_back(TBlob(workspace.dptr_ + offset,
                                    GetCutoffShape(c_moveaxis_shape), c.dev_mask(), c.dev_id()));
        offset = AligndWorkspaceSize<DType>(offset, cw_data_vec[i].shape_.Size());
      }
      c_data = TBlob(workspace.dptr_ + offset, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, c_data.shape_.Size());

      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        a_data = TBlob(workspace.dptr_ + offset, a_moveaxis_shape,
                       a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_data.shape_.Size());

        b_data = TBlob(workspace.dptr_ + offset, b_moveaxis_shape,
                       b.dev_mask(), b.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, b_data.shape_.Size());

        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                          mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                          mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    }
    std::vector<int> positive_vec;
    std::vector<int> a_index_vec, b_index_vec, c_index_vec;
    std::vector<OpReqType> req_vec;
    if (a_dim == 2 && b_dim == 3) {
      a_index_vec = {1, 0, 0, 1};
      b_index_vec = {2, 2, 1, 0};
      c_index_vec = {0, 1, 2, 2};
      positive_vec = {1, 0, 1, 0};
      req_vec = { kWriteTo, kWriteTo, kWriteTo, kAddTo };
    } else if (a_dim == 3 && b_dim == 2) {
      a_index_vec = {2, 2, 0, 1};
      b_index_vec = {1, 0, 1, 0};
      c_index_vec = {0, 1, 2, 2};
      positive_vec = {0, 1, 1, 0};
      req_vec = { kWriteTo, kWriteTo, kWriteTo, kAddTo };
    } else {
      a_index_vec = {1, 2, 2, 0, 0, 1};
      b_index_vec = {2, 1, 0, 2, 1, 0};
      c_index_vec = {0, 0, 1, 1, 2, 2};
      positive_vec = {1, 0, 1, 0, 1, 0};
      req_vec = { kWriteTo, kAddTo, kWriteTo, kAddTo, kWriteTo, kAddTo};
    }
    for (size_t i = 0; i < a_index_vec.size(); ++i) {
      int idx = c_index_vec[i];
      mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(),
                                                   a_data.dptr<DType>(),
                                                   aw_data.dptr<DType>(),
                                                   a_data.size(a_ndim - 1),
                                                   a_index_vec[i],
                                                   a_data.Size());
      mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(),
                                                   b_data.dptr<DType>(),
                                                   bw_data.dptr<DType>(),
                                                   b_data.size(b_ndim - 1),
                                                   b_index_vec[i],
                                                   b_data.Size());
      BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                       { aw_data, bw_data },
                                                       { kWriteTo },
                                                       { cw_data_vec[idx] });
      MXNET_ASSIGN_REQ_SWITCH(req_vec[i], req_type, {
        mxnet_op::Kernel<CrossOutAssign<req_type>, xpu>::Launch(s, cw_data_vec[idx].Size(),
                                                                cw_data_vec[idx].dptr<DType>(),
                                                                c_data.dptr<DType>(),
                                                                positive_vec[i],
                                                                c_data.size(c_ndim - 1),
                                                                idx,
                                                                c_data.Size());
      });
    }
    cw_data = cw_data.reshape(c.shape_);
    const DType *res_ptr = c_data.dptr<DType>();
    if (c_axis != c_ndim -1) {
      const Tuple<int> c_axis_index = GetMoveaxisIndex(-1, param.axisc, c_moveaxis_shape);
      TransposeImpl<xpu>(ctx.run_ctx, c_data, cw_data,
                         mxnet::TShape(c_axis_index.begin(), c_axis_index.end()));
      res_ptr = cw_data.dptr<DType>();
    }
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
        s, c.Size(), res_ptr, c.dptr<DType>());
    });
  }
};

template<typename xpu, typename DType>
struct NumpyCrossForwardImpl<xpu, DType, 2, 2> {
  static void op(const TBlob& a,
                 const TBlob& b,
                 const TBlob& c,
                 const std::vector<Tuple<int> >& moveaxis_index_vec,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req,
                 const Tensor<xpu, 1, DType>& workspace) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
    const Tuple<int>& a_moveaxis_index = moveaxis_index_vec[0];
    const Tuple<int>& b_moveaxis_index = moveaxis_index_vec[1];
    const mxnet::TShape& a_moveaxis_shape = moveaxis_shape_vec[0];
    const mxnet::TShape& b_moveaxis_shape = moveaxis_shape_vec[1];
    const mxnet::TShape& c_shape = c.shape_;
    const int a_ndim = a_moveaxis_shape.ndim();
    const int b_ndim = b_moveaxis_shape.ndim();
    const int a_axis = CheckAxis(param.axisa, a_ndim);
    const int b_axis = CheckAxis(param.axisb, b_ndim);

    TBlob aw_data, bw_data, cw_data, a_data, b_data;
    if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
      // Allocate workspace in cpu, no need to align address.
      DType *aw_ptr = workspace.dptr_;
      DType *bw_ptr = aw_ptr + a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      DType *cw_ptr = bw_ptr + b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      aw_data = TBlob(aw_ptr, GetCutoffShape(a_moveaxis_shape), a.dev_mask(), a.dev_id());
      bw_data = TBlob(bw_ptr, GetCutoffShape(b_moveaxis_shape), b.dev_mask(), b.dev_id());
      cw_data = TBlob(cw_ptr, c_shape, c.dev_mask(), c.dev_id());
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        DType *a_ptr = cw_ptr + c_shape.Size();
        DType *b_ptr = a_ptr + a_moveaxis_shape.Size();
        a_data = TBlob(a_ptr, a_moveaxis_shape, a.dev_mask(), a.dev_id());
        b_data = TBlob(b_ptr, b_moveaxis_shape, b.dev_mask(), b.dev_id());
        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                          mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                          mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    } else {
      // Allocate workspace in cpu, need to align address.
      size_t offset = 0;
      aw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(a_moveaxis_shape),
                      a.dev_mask(), a.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, aw_data.shape_.Size());

      bw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(b_moveaxis_shape),
                      b.dev_mask(), b.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, bw_data.shape_.Size());

      cw_data = TBlob(workspace.dptr_ + offset, c_shape,
                      c.dev_mask(), c.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, cw_data.shape_.Size());
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        a_data = TBlob(workspace.dptr_ + offset, a_moveaxis_shape,
                       a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_data.shape_.Size());

        b_data = TBlob(workspace.dptr_ + offset, b_moveaxis_shape,
                       b.dev_mask(), b.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, b_data.shape_.Size());

        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                          mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                          mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    }
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(),
                                                 a_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(),
                                                 a_data.size(a_ndim - 1),
                                                 0, a_data.Size());
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(),
                                                 b_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(),
                                                 b_data.size(b_ndim - 1),
                                                 1, b_data.Size());
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { aw_data, bw_data },
                                                     { req[0] }, { c });
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(),
                                                 a_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(),
                                                 a_data.size(a_ndim - 1),
                                                 1, a_data.Size());
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(),
                                                 b_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(),
                                                 b_data.size(b_ndim - 1),
                                                 0, b_data.Size());
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { aw_data, bw_data },
                                                     { kWriteTo }, { cw_data });
    BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx, { c, cw_data },
                                                       { kWriteTo }, { c });
  }
};

template<typename xpu>
void NumpyCrossForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& c = outputs[0];

  if (kNullOp == req[0]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == a.Size() || 0U == b.Size()) { return; }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  const mxnet::TShape& c_shape = c.shape_;
  const int a_ndim = a_shape.ndim();
  const int b_ndim = b_shape.ndim();
  const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
  Tuple<int> a_moveaxis_index = GetMoveaxisIndex(param.axisa, -1, a_shape);
  Tuple<int> b_moveaxis_index = GetMoveaxisIndex(param.axisb, -1, b_shape);
  Tuple<int> c_moveaxis_index = GetMoveaxisIndex(param.axisc, -1, c_shape);
  mxnet::TShape a_moveaxis_shape = GetMoveaxisShape(a_moveaxis_index, a_shape);
  mxnet::TShape b_moveaxis_shape = GetMoveaxisShape(b_moveaxis_index, b_shape);
  mxnet::TShape c_moveaxis_shape = GetMoveaxisShape(c_moveaxis_index, c_shape);
  const std::vector<mxnet::TShape > shape_vec({ a_moveaxis_shape, b_moveaxis_shape,
                                                c_moveaxis_shape });
  const std::vector<Tuple<int> > index_vec({ a_moveaxis_index, b_moveaxis_index,
                                             c_moveaxis_index });

  MSHADOW_SGL_DBL_TYPE_SWITCH(c.type_flag_, DType, {
    // Calculate workspace.
    size_t workspace_size = NumpyCrossWorkspaceSize<xpu, DType>(a_moveaxis_shape,
                                                                b_moveaxis_shape,
                                                                c_moveaxis_shape,
                                                                c_shape,
                                                                attrs, ctx, req);
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
      Shape1(workspace_size), s);

    if (a_moveaxis_shape[a_ndim - 1] == 2) {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
        NumpyCrossForwardImpl<xpu, DType, 2, 2>::op(a, b, c, index_vec, shape_vec, attrs,
                                                    ctx, req, workspace);
      } else {
        // Case 2: a.shape[-1] == 2 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 2, 3>::op(a, b, c, index_vec, shape_vec, attrs,
                                                    ctx, req, workspace);
      }
    } else {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 3: a.shape[-1] == 3 and b.shape[-1] == 2, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 3, 2>::op(a, b, c, index_vec, shape_vec, attrs,
                                                    ctx, req, workspace);
      } else {
        // Case 4: a.shape[-1] == 3 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 3, 3>::op(a, b, c, index_vec, shape_vec, attrs,
                                                    ctx, req, workspace);
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_

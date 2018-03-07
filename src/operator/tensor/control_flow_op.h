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
 * \file control_flow.h
 * \brief Function definitions of operators for controlling flow
 */
#ifndef MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_
#define MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

/*! \brief Choose elements from x or y depending on condition.
 * The condition, x, and y have the same shape.
 * The returned array is formed by elements from x or y
 * depending on the elements of condition.
 */
template<int req>
struct where {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const CType* cond,
                                  const DType* x, const DType* y) {
    KERNEL_ASSIGN(out[i], req, (0 != cond[i]? x[i] : y[i]));
  }
};

/*! \brief Choose elements from x or y depending on condition.
 * The condition, x, and y have the same shape.
 * The returned array is formed by elements from x or y
 * depending on the elements of condition.
 * The condition is a csr matrix, while x and y are both dense.
 */
template<int req>
struct where_csr {
  // DType is the output data type
  // CType is condition data type
  // i is for i-th row in the output
  template<typename DType, typename CType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const IType* cond_idx,
                                  const IType* cond_indptr, const CType* cond_data,
                                  const nnvm::dim_t num_cols, const DType* x) {
    using nnvm::dim_t;
    const dim_t offset = i * num_cols;
    for (dim_t j = cond_indptr[i]; j < cond_indptr[i + 1]; j++) {
      const CType data = cond_data[j];
      if (data != 0) {
        const IType col_idx = cond_idx[j];
        const dim_t out_idx = offset + col_idx;
        KERNEL_ASSIGN(out[out_idx], req, x[out_idx]);
      }
    }
  }
};


/*! \brief Choose elements from x or y depending on condition
 * The condition is a vector whose size is the same as the
 * x's first dim size.
 * The returned array is formed by rows from x or y depending on
 * the condition's elements.
 */
template<int req>
struct where_batch {
  // DType is the output data type
  // CType is the condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const CType* cond,
                                  const DType* x, const DType* y, int M) {
    KERNEL_ASSIGN(out[i], req, (0 != cond[i/M]? x[i] : y[i]));
  }
};

/*!
 * \brief Template for calculating grad[x] and grad[y].
 * template argument req is OpReqType; negate indicates
 * whether the output is grad_x (negate=true)
 * or grad_y (negate=false).
 */
template<int req, bool negate>
struct where_backward {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* grad_out,
                                  const DType* grad_in,
                                  const CType* cond) {
    KERNEL_ASSIGN(grad_out[i], req,
      ((0 == cond[i])^negate)? grad_in[i] : static_cast<DType>(0));
  }
};

/*!
 * \brief Template for calculating grad[x] and grad[y].
 * template argument req is OpReqType; negate indicates
 * whether the output is grad_x (negate=true)
 * or grad_y (negate=false).
 * cond is a csr matrix, while others are dense ones.
 */
template<int req, bool negate>
struct where_backward_csr {
  // DType is the output data type
  // CType is condition data type
  // IType is condition aux data type
  template<typename DType, typename CType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* grad_out,
                                  const DType* grad_in,
                                  const CType* cond_data,
                                  const IType* cond_idx,
                                  const IType* cond_indptr,
                                  const nnvm::dim_t num_cols) {
    const IType offset = i * num_cols;
    const DType zero = static_cast<DType>(0);
    for (IType j = cond_indptr[i]; j < cond_indptr[i + 1]; j++) {
      const IType col = cond_idx[j];
      const IType grad_offset = offset + col;
      KERNEL_ASSIGN(grad_out[grad_offset], req,
        ((0 == cond_data[j])^negate)? grad_in[grad_offset] : zero);
    }
  }
};


/*!
 * \brief Template for calculating grad[x] and grad[y].
 * template argument req is OpReqType; negate indicates
 * whether the output is grad_x (negate=true)
 * or grad_y (negate=false).
 * The condition is a vector whose size is the same as the
 * x's first dim size.
 */
template<int req, bool negate>
struct where_batch_backward {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* grad_out,
                                  const DType* grad_in,
                                  const CType* cond, int M) {
    KERNEL_ASSIGN(grad_out[i], req,
      ((0 == cond[i/M])^negate)? grad_in[i] : static_cast<DType>(0));
  }
};

inline bool WhereOpShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape>* in_attrs,
                         std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  TShape tshape((*in_attrs)[1]);
  if (!shape_assign(&tshape, (*in_attrs)[2])) return false;
  if (!shape_assign(&tshape, (*out_attrs)[0])) return false;
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, tshape);
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, tshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);

  if ((*in_attrs)[0].ndim() == tshape.ndim()) {
    if (!shape_assign(&tshape, (*in_attrs)[0])) return false;
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tshape);
    return true;
  } else if ((*in_attrs)[0].ndim() == 1) {
    return (*in_attrs)[0].Size() == static_cast<size_t>(tshape[0]);
  }
  return false;
}

inline bool WhereOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  int dtype = -1;
  if (!type_assign(&dtype, (*in_attrs)[1])) return false;
  if (!type_assign(&dtype, (*in_attrs)[2])) return false;
  if (!type_assign(&dtype, (*out_attrs)[0])) return false;
  if (-1 == dtype) return false;

  TYPE_ASSIGN_CHECK(*in_attrs, 1, dtype);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, dtype);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);

  return true;
}

inline bool WhereOpForwardStorageType(const nnvm::NodeAttrs& attrs,
                                      const int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int cond_stype = in_attrs->at(0);
  const int x_stype = in_attrs->at(1);
  const int y_stype = in_attrs->at(2);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (!dispatched && cond_stype == kCSRStorage && x_stype == kDefaultStorage &&
      y_stype == kDefaultStorage) {
    // csr, dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

inline bool WhereOpBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  const auto in_grad_stype = in_attrs->at(0);
  const auto cond_stype = in_attrs->at(1);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, dns -> dns, dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && cond_stype == kCSRStorage && in_grad_stype == kDefaultStorage) {
    // dns, csr -> dns, dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}



template<typename xpu>
void WhereOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& cond = inputs[0];
  const TBlob& x = inputs[1];
  const TBlob& y = inputs[2];
  const TBlob& out = outputs[0];
  if (out.Size() == 0) return;
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.type_flag_, CType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        if (cond.shape_ == x.shape_) {
          Kernel<where<req_type>, xpu>::Launch(s, out.Size(), out.dptr<DType>(),
                                               cond.dptr<CType>(), x.dptr<DType>(),
                                               y.dptr<DType>());
        } else {
          Kernel<where_batch<req_type>, xpu>::Launch(s, out.Size(), out.dptr<DType>(),
                                                     cond.dptr<CType>(), x.dptr<DType>(),
                                                     y.dptr<DType>(), x.Size()/cond.Size());
        }
      });
    });
  });
}

template<typename xpu>
void WhereOpForwardCsrImpl(mshadow::Stream<xpu> *s,
                           const NDArray& cond,
                           const TBlob& x,
                           const TBlob& y,
                           const OpReqType req,
                           const TBlob& out) {
  using namespace mxnet_op;
  using namespace csr;
  if (out.Size() == 0 || req == kNullOp) return;
  CHECK(cond.shape() == x.shape_)
    << "WhereOpForwardCsrImpl only supports inputs of same 2-D shapes";
  CHECK(req == kWriteInplace || req == kWriteTo)
    << "WhereOpForwardCsrImpl doesn't support req = " << req;
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.dtype(), CType, {
      MSHADOW_TYPE_SWITCH(cond.aux_type(kIdx), IType, {
        MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
          mshadow::Copy(out.FlatTo1D<xpu, DType>(s), y.FlatTo1D<xpu, DType>(s), s);
          // no condition is satisfied
          if (!cond.storage_initialized()) return;
          IType* cond_idx = cond.aux_data(kIdx).dptr<IType>();
          IType* cond_indptr = cond.aux_data(kIndPtr).dptr<IType>();
          CType* cond_data = cond.data().dptr<CType>();
          Kernel<where_csr<req_type>, xpu>::Launch(s, cond.shape()[0], out.dptr<DType>(),
                 cond_idx, cond_indptr, cond_data, cond.shape()[1], x.dptr<DType>());
        });
      });
    });
  });
}

template<typename xpu>
void WhereOpForwardEx(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const int cond_stype = inputs[0].storage_type();
  const int x_stype = inputs[1].storage_type();
  const int y_stype = inputs[2].storage_type();
  const auto& out_stype = outputs[0].storage_type();
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(inputs[0].shape().ndim(), 1) << "WhereOpForwardEx with 1-D cond is not implemented";
  if (cond_stype == kCSRStorage && x_stype == kDefaultStorage &&
      y_stype == kDefaultStorage && out_stype == kDefaultStorage) {
    WhereOpForwardCsrImpl(s, inputs[0], inputs[1].data(), inputs[2].data(), req[0],
                          outputs[0].data());
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to condition, x, and y. The gradient
 * with respect to condition is always 0. The gradient
 * with respect to x and y depends on the corresponding
 * elements in the condition.
 * The inputs are gradient with respect to the output
 * of the operator, condition, x, and y.
 * The outputs are gradients with respect to
 * condition, x, and y.
 */
template<typename xpu>
void WhereOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& grad_in = inputs[0];
  const TBlob& cond = inputs[1];
  const TBlob& grad_x = outputs[0];
  const TBlob& grad_y = outputs[1];
  if (grad_in.Size() == 0) return;
  MSHADOW_TYPE_SWITCH(grad_in.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.type_flag_, CType, {
      bool same_shape = (cond.shape_ == grad_in.shape_);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type_x, {
        if (same_shape) {
          Kernel<where_backward<req_type_x, true>, xpu>::Launch(s, grad_in.Size(),
            grad_x.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>());
        } else {
          Kernel<where_batch_backward<req_type_x, true>, xpu>::Launch(s, grad_in.Size(),
            grad_x.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>(),
            grad_in.Size()/cond.Size());
        }
      });
      MXNET_ASSIGN_REQ_SWITCH(req[1], req_type_y, {
        if (same_shape) {
          Kernel<where_backward<req_type_y, false>, xpu>::Launch(s, grad_in.Size(),
            grad_y.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>());
        } else {
          Kernel<where_batch_backward<req_type_y, false>, xpu>::Launch(s, grad_in.Size(),
            grad_y.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>(),
            grad_in.Size()/cond.Size());
        }
      });
    });
  });
}

template<typename xpu>
void WhereOpBackwardCsrImpl(mshadow::Stream<xpu> *s,
                            const TBlob& grad_in,
                            const NDArray& cond,
                            const std::vector<OpReqType>& req,
                            const TBlob& grad_x,
                            const TBlob& grad_y) {
  using namespace mxnet_op;
  using namespace csr;
  if (grad_in.Size() == 0) return;
  CHECK(cond.shape() == grad_x.shape_)
    << "WhereOpForwardCsrImpl only supports inputs of same 2-D shapes";
  CHECK_NE(req[0], kAddTo) << "WhereOpForwardCsrImpl doesn't support kAddTo";
  CHECK_NE(req[1], kAddTo) << "WhereOpForwardCsrImpl doesn't support kAddTo";
  MSHADOW_TYPE_SWITCH(grad_in.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.dtype(), CType, {
      MSHADOW_IDX_TYPE_SWITCH(cond.aux_type(kIdx), IType, {
        if (req[0] != kNullOp) {
          Fill<false>(s, grad_x, req[0], 0);
          // some conditions are satisfied
          if (cond.storage_initialized()) {
            const IType* cond_indptr = cond.aux_data(kIndPtr).dptr<IType>();
            const IType* cond_idx = cond.aux_data(kIdx).dptr<IType>();
            const CType* cond_data = cond.data().dptr<CType>();
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type_x, {
              Kernel<where_backward_csr<req_type_x, true>, xpu>::Launch(s, cond.shape()[0],
                grad_x.dptr<DType>(), grad_in.dptr<DType>(), cond_data, cond_idx,
                cond_indptr, cond.shape()[1]);
            });
          }
        }
        if (req[1] != kNullOp) {
          mshadow::Copy(grad_y.FlatTo1D<xpu, DType>(s), grad_in.FlatTo1D<xpu, DType>(s), s);
          CHECK_EQ(req[1], kWriteTo);
          if (cond.storage_initialized()) {
            const IType* cond_indptr = cond.aux_data(kIndPtr).dptr<IType>();
            const IType* cond_idx = cond.aux_data(kIdx).dptr<IType>();
            const CType* cond_data = cond.data().dptr<CType>();
            MXNET_ASSIGN_REQ_SWITCH(req[1], req_type_y, {
              Kernel<where_backward_csr<req_type_y, false>, xpu>::Launch(s, cond.shape()[0],
                grad_y.dptr<DType>(), grad_in.dptr<DType>(), cond_data, cond_idx,
                cond_indptr, cond.shape()[1]);
            });
          }
        }
      });
    });
  });
}

template<typename xpu>
void WhereOpBackwardEx(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (inputs[1].shape().ndim() == 1) {
    LOG(FATAL) << "WhereOpBackwardEx with 1-D cond is not implemented";
  }
  const auto grad_in_stype = inputs[0].storage_type();
  const auto cond_stype = inputs[1].storage_type();
  const auto grad_x_stype = outputs[0].storage_type();
  const auto grad_y_stype = outputs[1].storage_type();
  if (grad_in_stype == kDefaultStorage && cond_stype == kCSRStorage &&
      grad_x_stype == kDefaultStorage && grad_y_stype == kDefaultStorage) {
    WhereOpBackwardCsrImpl(s, inputs[0].data(), inputs[1], req, outputs[0].data(),
                           outputs[1].data());
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_

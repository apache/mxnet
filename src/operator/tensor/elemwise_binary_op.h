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
 * \file elemwise_binary_op.h
 * \brief Function definition of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "elemwise_unary_op.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

inline bool ElemwiseBinaryBackwardUseInStorageType(const nnvm::NodeAttrs& attrs,
                                                   const Context& ctx,
                                                   int *dispatch_type,
                                                   std::vector<int> *in_attrs,
                                                   std::vector<int> *out_attrs) {
  using namespace common;
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 2U);
  const auto lhs_grad_stype = out_attrs->at(0);
  auto& rhs_grad_stype = out_attrs->at(1);
  bool dispatched = false;
  if (ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = dispatch_on_storage(out_attrs, kDefaultStorage,
                                     dispatch_type, kDispatchFCompute);
  } else if (ContainsOnlyStorage(*in_attrs, kRowSparseStorage)) {
    // rsp, rsp, rsp -> [dns, rsp], [dns, rsp]
    dispatched = dispatch_on_storage(out_attrs, kRowSparseStorage,
                                     dispatch_type, kDispatchFComputeEx);
    // when some grad_stype is already kDefaultStorage, FComputeEx can handle that, too
    if ((lhs_grad_stype == kDefaultStorage || lhs_grad_stype == kRowSparseStorage) &&
        (rhs_grad_stype == kDefaultStorage || rhs_grad_stype == kRowSparseStorage)) {
      TYPE_ASSIGN_CHECK(dispatch_type, 0, kDispatchFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched) {
    dispatch_on_storage(out_attrs, kDefaultStorage,
                        dispatch_type, kDispatchFComputeFallback);
    LogStorageFallback(attrs, ctx, in_attrs, out_attrs);
  }
  return true;
}

inline bool ElemwiseBinaryComputeStorageType(const nnvm::NodeAttrs& attrs,
                                             const Context& ctx,
                                             int *dispatch_type,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto& lhs_stype = in_attrs->at(0);
  auto& rhs_stype = in_attrs->at(1);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (lhs_stype == kDefaultStorage && rhs_stype == kDefaultStorage) {
    // dns, dns -> dns
    dispatched = dispatch_on_storage(&out_stype, kDefaultStorage,
                                     dispatch_type, kDispatchFCompute);
  } else if (lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage
             && out_stype == kDefaultStorage) {
    // rsp, rsp -> dns
    dispatched = dispatch_on_storage(&out_stype, kDefaultStorage,
                                     dispatch_type, kDispatchFComputeEx);
  } else if (lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage) {
    // rsp, rsp -> rsp
    dispatched = dispatch_on_storage(&out_stype, kRowSparseStorage,
                                     dispatch_type, kDispatchFComputeEx);
  } else if ((lhs_stype == kRowSparseStorage && rhs_stype == kDefaultStorage) ||
             (lhs_stype == kDefaultStorage && rhs_stype == kRowSparseStorage)) {
    // rsp, dns -> dns / dns, rsp -> dns
    dispatched = dispatch_on_storage(&out_stype, kDefaultStorage,
                                     dispatch_type, kDispatchFComputeEx);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kCSRStorage) {
    // csr, csr -> csr
    dispatched = dispatch_on_storage(&out_stype, kCSRStorage,
                                     dispatch_type, kDispatchFComputeEx);
  }
  if (!dispatched) {
    dispatch_on_storage(&out_stype, kDefaultStorage,
                        dispatch_type, kDispatchFComputeFallback);
    LogStorageFallback(attrs, ctx, in_attrs, out_attrs);
  }
  return true;
}

inline bool ElemwiseMulStorageType(const nnvm::NodeAttrs& attrs,
                                       const Context& ctx,
                                       int *dispatch_type,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), 1U) << " in operator " << attrs.name;
  NDArrayStorageType stype = kDefaultStorage;
  for (size_t i = 0; i < 2U; ++i) {
    const NDArrayStorageType in_stype = static_cast<NDArrayStorageType>((*in_attrs)[i]);
    if (in_stype != kDefaultStorage) {
      if (stype == kDefaultStorage) {
        stype = in_stype;
      }
    }
  }
  STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, 0, stype);
  if (stype == kDefaultStorage) {
    TYPE_ASSIGN_CHECK(dispatch_type, 0, kDispatchFComputeFallback);
  }
  else {
    TYPE_ASSIGN_CHECK(dispatch_type, 0, kDispatchFComputeEx);
  }
  return true;
}

/*! Gather binary operator functions into ElemwiseBinaryOp class */
class ElemwiseBinaryOp : public OpBase {
 public:
  template<typename OP, int Req>
  struct BackwardUseNoneOp {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad, const DType *ograd) {
      KERNEL_ASSIGN(igrad[i], Req, OP::Map(ograd[i]));
    }
  };

  template<typename OP, int Req>
  struct BackwardUseInOp {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad,
                                    const DType *ograd, const DType *lhs, const DType *rhs) {
      KERNEL_ASSIGN(igrad[i], Req, ograd[i] * OP::Map(lhs[i], rhs[i]));
    }
  };

  /*! \brief For sparse, assume missing rvalue is 0 */
  template<typename OP, int Req>
  struct MissingRValueOp {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], DType(0)));
    }
  };

  /*! \brief For sparse, assume missing lvalue is 0 */
  template<typename OP, int Req>
  struct MissingLValueOp {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *rhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(DType(0), rhs[i]));
    }
  };

 private:
  /*! \brief Fill contiguous dense output rows with value computed from 0 lhs and 0 rhs input */
  template<typename xpu, typename DType, typename OP>
  static inline size_t FillDense(mshadow::Stream<xpu> *s,
                                 const size_t idx_l,
                                 const size_t idx_r,
                                 const OpReqType req,
                                 mshadow::Tensor<xpu, 2, DType> *out,
                                 const size_t iter_out) {
    using namespace mshadow::expr;
    const int index_out_min = std::min(idx_l, idx_r);
    if (static_cast<size_t>(index_out_min) > iter_out) {
      const size_t size = (*out)[iter_out].shape_.Size();
      const DType zero_input_val = OP::Map(DType(0), DType(0));
      #pragma omp parallel for
      for (int i = iter_out; i < index_out_min; ++i) {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          mxnet_op::Kernel<SetToScalar<Req>, xpu>::Launch(s, size, (*out)[i].dptr_,
                                                          zero_input_val);
        });
      }
    }
    return index_out_min;
  }

  template<typename DType>
  static inline bool IsSameArray(const NDArray& a1, const NDArray& a2) {
    return a1.var() == a2.var();
  }

  /*! \brief Binary op handling for lhr/rhs: RspDns, RspRsp, DnsRsp, or RspRsp->Dns result */
  template<typename DType, typename IType, typename OP>
  static void RspRspOp(mshadow::Stream<cpu> *s,
                       const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const NDArray &lhs,
                       const NDArray &rhs,
                       const OpReqType req,
                       const NDArray &output,
                       const bool lhs_may_be_dense,
                       const bool rhs_may_be_dense,
                       const bool allow_inplace);

  /*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
  template<typename DType, typename IType, typename CType, typename OP>
  static inline void CsrCsrOp(mshadow::Stream<cpu> *s,
                              const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const NDArray &lhs,
                              const NDArray &rhs,
                              const OpReqType req,
                              const NDArray &output);
  /*! \brief Minimum of three */
  static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
  }

  /*! \brief Maximum of three */
  static MSHADOW_XINLINE size_t maxthree(const size_t a, const size_t b, const size_t c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
  }

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, typename DType,
    bool lhs_may_be_dense, bool rhs_may_be_dense, typename BackupCompute>
  static void ComputeExDenseLRValue_(const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs,
                                     BackupCompute backup_compute) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] != kNullOp) {
      const NDArray *sparse = &inputs[0];
      if (sparse->storage_type() == kDefaultStorage) {
        sparse = &inputs[1];
        if (sparse->storage_type() == kDefaultStorage) {
          // Do we need to worry about sparse result here?
          CHECK_EQ(outputs[0].storage_type(), kDefaultStorage);
          MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
          return;
        }
      }
      bool allowed = false;
      if (lhs_may_be_dense && rhs_may_be_dense) {
        allowed = common::ContainsNonDefaultStorage(inputs);
      } else if (lhs_may_be_dense) {
        allowed = inputs[1].storage_type() != kDefaultStorage;
      } else if (rhs_may_be_dense) {
        allowed = inputs[0].storage_type() != kDefaultStorage;
      } else {
        allowed = !common::ContainsNonDefaultStorage(inputs);
      }
      if (allowed) {
        allowed = !common::ContainsStorage(inputs, kCSRStorage);
      }
      // If any input or output is dense, fallback to FCompute
      if (allowed) {
        mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
        MSHADOW_IDX_TYPE_SWITCH(sparse->aux_type(rowsparse::kIdx), IType, {
          RspRspOp<DType, IType, OP>(
            s, attrs, ctx, inputs[0], inputs[1],
            req[0], outputs[0],
            lhs_may_be_dense, rhs_may_be_dense, false);
        });
      } else {
        // May be lhs=dense, rhs=sparse
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             backup_compute,
                             "ComputeExDenseLRValue_");
      }
    }
  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BackwardUseNone_(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                      / DataType<DType>::kLanes);
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[0].dptr<DType>());
    } else if (req[0] != kNullOp) {
      DType *lgrad_dptr = outputs[0].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BackwardUseNoneOp<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr);
      });
    }
    if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[1].dptr<DType>());
    } else if (req[1] != kNullOp) {
      DType *rgrad_dptr = outputs[1].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
        Kernel<BackwardUseNoneOp<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr);
      });
    }
  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BackwardUseIn_(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
    DCHECK_EQ(outputs.size(), 2U);
    DCHECK_EQ(inputs.size(), 3U);
    mxnet_op::Stream<xpu> *s = ctx.get_stream<xpu>();
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    const DType *lhs_dptr = inputs[1].dptr<DType>();
    const DType *rhs_dptr = inputs[2].dptr<DType>();
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      const int size = static_cast<int>(
        (outputs[0].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * lgrad_dptr = outputs[0].dptr<DType>();
      mxnet_op::Kernel<BackwardUseInOp<LOP, Req>, xpu>::Launch(
        s, size, lgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
      const int size = static_cast<int>(
        (outputs[1].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * rgrad_dptr = outputs[1].dptr<DType>();
      mxnet_op::Kernel<BackwardUseInOp<ROP, Req>, xpu>::Launch(
        s, size, rgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
  }

  template<
    typename xpu,
    typename LOP,
    typename ROP,
    typename DType,
    bool in0_ok_dense = false,
    bool in1_ok_dense = false,
    bool in2_ok_dense = false,
    typename BackupCompute>
  static inline void BackwardUseInEx_(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<NDArray> &outputs,
                                      BackupCompute backup_compute) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    // lhs grad
    if (req[0] != kNullOp) {
      // RspRspOp can handle dense outputs so long as OP(0, 0) == 0
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        RspRspOp<DType, IType, LOP>(
          s, attrs, ctx, inputs[1], inputs[2], req[0], outputs[0],
          false, false, false);
      });
      // lhs in-place
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        RspRspOp<DType, IType, mshadow::op::mul>(
          s, attrs, ctx, outputs[0], inputs[0], req[0], outputs[0],
          false, false, true);
      });
    }
    // rhs grad
    if (req[1] != kNullOp) {
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        RspRspOp<DType, IType, ROP>(
          s, attrs, ctx, inputs[1], inputs[2], req[1], outputs[1],
          false, false, false);
      });
      // rhs in-place
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        RspRspOp<DType, IType, mshadow::op::mul>(
          s, attrs, ctx, inputs[0], outputs[1], req[1], outputs[1],
          false, false, true);
      });
    }
  }

 public:
  template<typename xpu, typename OP>
  static void Compute(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void ComputeWithHalf2(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] == kNullOp) return;
    const auto lhs_stype = inputs[0].storage_type();
    const auto rhs_stype = inputs[1].storage_type();
    const auto out_stype = outputs[0].storage_type();
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    // rsp, rsp -> rsp/dns
    if (lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage &&
        (out_stype == kRowSparseStorage || out_stype == kDefaultStorage)) {
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
          RspRspOp<DType, IType, OP>(
            s, attrs, ctx, inputs[0], inputs[1],
            req[0], outputs[0], false, false, false);
        });
      });
    } else if (lhs_stype == kCSRStorage && rhs_stype == kCSRStorage &&
               out_stype == kCSRStorage) {
      // csr, csr -> csr
      MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(csr::kIdx), IType, {
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(csr::kIndPtr), CType, {
          MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
            CsrCsrOp<DType, IType, CType, OP>(
              s, attrs, ctx, inputs[0], inputs[1], req[0], outputs[0]);
          });
        });
      });
    } else {
      LOG(FATAL) << "Not implemented: " << OperatorInfoEx(attrs, ctx, inputs, req, outputs);
    }
  }

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, bool lhs_may_be_dense, bool rhs_may_be_dense>
  static void ComputeExDenseLRValue(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      ComputeExDenseLRValue_<xpu, OP, DType, lhs_may_be_dense, rhs_may_be_dense>(
        attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BackwardUseNone(const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const std::vector<TBlob> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BackwardUseNoneWithHalf2(const nnvm::NodeAttrs &attrs,
                                              const OpContext &ctx,
                                              const std::vector<TBlob> &inputs,
                                              const std::vector<OpReqType> &req,
                                              const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BackwardUseNoneEx(const nnvm::NodeAttrs &attrs,
                                       const OpContext &ctx,
                                       const std::vector<NDArray> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 1U);   // output grad
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    const auto in_stype = inputs[0].storage_type();
    const auto lhs_stype = outputs[0].storage_type();
    const auto rhs_stype = outputs[1].storage_type();
    // lhs grad
    if (req[0] != kNullOp) {
      if (in_stype == kRowSparseStorage && lhs_stype == kRowSparseStorage) {
        // rsp -> rsp, _. op requires 0-input returns 0-output
        DCHECK_LT(fabs(static_cast<float>(LOP::Map(0))), 1e-5f);
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          UnaryOp::KernelComputeEx<xpu, BackwardUseNoneOp<LOP, Req>>(attrs, ctx, inputs,
                                                                     req, {outputs[0]});
        });
      } else {
        LOG(FATAL) << "Not implemented: " << OperatorInfoEx(attrs, ctx, inputs, req, outputs);
      }
    }
    // rhs grad
    if (req[1] != kNullOp) {
      if (in_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage) {
        // rsp -> _, rsp. op requires 0-input returns 0-output
        DCHECK_LT(fabs(static_cast<float>(ROP::Map(0))), 1e-5f);
        MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
          UnaryOp::KernelComputeEx<xpu, BackwardUseNoneOp<ROP, Req>>(attrs, ctx, inputs,
                                                                     req, {outputs[1]});
        });
      } else {
        LOG(FATAL) << "Not implemented: " << OperatorInfoEx(attrs, ctx, inputs, req, outputs);
      }
    }
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BackwardUseIn(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BackwardUseInWithHalf2(const nnvm::NodeAttrs &attrs,
                                            const OpContext &ctx,
                                            const std::vector<TBlob> &inputs,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<
    typename xpu, typename LOP, typename ROP,
    bool in0_ok_dense = false, bool in1_ok_dense = false, bool in2_ok_dense = false>
  static inline void BackwardUseInEx(const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
    using namespace common;
    CHECK_EQ(inputs.size(), 3U);
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    const auto lhs_grad_stype = outputs[0].storage_type();
    const auto rhs_grad_stype = outputs[1].storage_type();
    if (ContainsOnlyStorage(inputs, kRowSparseStorage) &&
        (lhs_grad_stype == kDefaultStorage || lhs_grad_stype == kRowSparseStorage) &&
        (rhs_grad_stype == kDefaultStorage || rhs_grad_stype == kRowSparseStorage)) {
      // rsp, rsp, rsp -> [dns, rsp], [dns, rsp]
      MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
        BackwardUseInEx_<xpu, LOP, ROP, DType, in0_ok_dense, in1_ok_dense, in2_ok_dense>(
          attrs, ctx, inputs, req, outputs, BackwardUseIn<xpu, LOP, ROP>);
      });
    }
  }
};  // class ElemwiseBinaryOp

#define MXNET_OPERATOR_REGISTER_BINARY(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"lhs", "rhs"};                \
    })                                                              \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "NDArray-or-Symbol", "first input")          \
  .add_argument("rhs", "NDArray-or-Symbol", "second input")

/*! \brief Binary launch */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(__name$, __kernel$)            \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                             \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseBinaryComputeStorageType) \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>)

/*! \brief Binary launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(__name$, __kernel$)                     \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                                         \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeDnsOutput<1, true, false>) \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)                 \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

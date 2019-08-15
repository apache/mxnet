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
 *  Copyright (c) 2016 by Contributors
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
#include "../../engine/openmp.h"
#include "elemwise_unary_op.h"
#include "../../common/utils.h"
#include "./init_op.h"

namespace mxnet {
namespace op {

/*! Gather binary operator functions into ElemwiseBinaryOp class */
class ElemwiseBinaryOp : public OpBase {
 public:
  /*! \brief For sparse, assume missing rvalue is 0 */
  template<typename OP, int Req>
  struct MissingRValueOp {
    typedef OP Operation;
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], DType(0)));
    }
  };

  /*! \brief For sparse, assume missing lvalue is 0 */
  template<typename OP, int Req>
  struct MissingLValueOp {
    typedef OP Operation;
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *rhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(DType(0), rhs[i]));
    }
  };

 private:
  /*!
   * \brief CSR operation requires temp space
   */
  enum ResourceRequestType {
    kTempSpace
  };

  /*!
   * \brief Fill contiguous dense output rows with value computed from 0 lhs and 0 rhs input
   *        CPU-Only version
   */
  template<typename DType, typename OP, typename xpu>
  static inline size_t FillDense(mshadow::Stream<xpu> *s,
                                 const size_t idx_l,
                                 const size_t idx_r,
                                 const OpReqType req,
                                 mshadow::Tensor<xpu, 2, DType> *out,
                                 const size_t iter_out) {
    const int index_out_min = static_cast<int>(std::min(idx_l, idx_r));
    if (static_cast<size_t>(index_out_min) > iter_out) {
      const DType zero_input_val = OP::Map(DType(0), DType(0));
      #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = static_cast<int>(iter_out); i < index_out_min; ++i) {
        Fill<false>(s, (*out)[i], req, zero_input_val);
      }
    }
    return static_cast<size_t>(index_out_min);  // MSVC wants OMP loops to always use 'int'
  }

  static inline bool IsSameArray(const NDArray& a1, const NDArray& a2) {
    return a1.var() == a2.var();
  }

  /*! \brief Minimum of three */
  static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
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
        Kernel<mxnet_op::op_with_req<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr);
      });
    }
    if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[1].dptr<DType>());
    } else if (req[1] != kNullOp) {
      DType *rgrad_dptr = outputs[1].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
        Kernel<mxnet_op::op_with_req<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr);
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
      mxnet_op::Kernel<mxnet_op::op_with_req<mxnet_op::backward_grad_tuned<LOP>, Req>, xpu>::Launch(
        s, size, lgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
      const int size = static_cast<int>(
        (outputs[1].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * rgrad_dptr = outputs[1].dptr<DType>();
      mxnet_op::Kernel<mxnet_op::op_with_req<mxnet_op::backward_grad_tuned<ROP>, Req>, xpu>::Launch(
        s, size, rgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
  }

  template<
    typename xpu,
    typename LOP,
    typename ROP,
    bool in0_ok_dense = false,
    bool in1_ok_dense = false,
    bool in2_ok_dense = false,
    typename BackupCompute>
  static inline void RspRspOpBackward(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<NDArray> &outputs,
                                      BackupCompute backup_compute) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    // lhs grad
    if (req[0] != kNullOp) {
      // RspRspOp can handle dense outputs so long as OP(0, 0) == 0
      RspRspOp<LOP>(
          s, attrs, ctx, inputs[1], inputs[2], req[0], outputs[0],
          false, false, false, false);
      // lhs in-place
      RspRspOp<op::mshadow_op::mul>(
          s, attrs, ctx, outputs[0], inputs[0], req[0], outputs[0],
          false, false, true, false);
    }
    // rhs grad
    if (req[1] != kNullOp) {
      RspRspOp<ROP>(
        s, attrs, ctx, inputs[1], inputs[2], req[1], outputs[1],
        false, false, false, false);
      // rhs in-place
      RspRspOp<op::mshadow_op::mul>(
        s, attrs, ctx, inputs[0], outputs[1], req[1], outputs[1],
        false, false, true, false);
    }
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void DnsCsrCsrOpBackward(const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<NDArray> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &outputs) {
    const bool supported_ops = std::is_same<mshadow_op::right, LOP>::value &&
                                std::is_same<mshadow_op::left, ROP>::value;
    CHECK(supported_ops)
      << "Only backward for mul is supported (LOP should be right, ROP should be left)";
    const NDArray& out_grad = inputs[0];
    const NDArray& lhs_in = inputs[1];
    const NDArray& rhs_in = inputs[2];
    const NDArray& lhs_grad = outputs[0];
    const NDArray& rhs_grad = outputs[1];
    const bool reverse = (outputs[0].storage_type() == kCSRStorage);
    if (reverse) {
      DnsCsrCsrOp<xpu, mshadow_op::mul>(attrs, ctx, out_grad, rhs_in, req[0], lhs_grad, false);
      Compute<xpu, mshadow_op::mul>(attrs, ctx, {out_grad.data(), lhs_in.data()}, {req[1]},
                                    {rhs_grad.data()});
    } else {
      DnsCsrCsrOp<xpu, mshadow_op::mul>(attrs, ctx, out_grad, lhs_in, req[1], rhs_grad, false);
      Compute<xpu, mshadow_op::mul>(attrs, ctx, {out_grad.data(), rhs_in.data()}, {req[0]},
                                    {lhs_grad.data()});
    }
  }

 public:
  /*! \brief Binary op handling for lhr/rhs: RspDns, RspRsp, DnsRsp, or RspRsp->Dns result */
  template<typename OP>
  static void RspRspOp(mshadow::Stream<cpu> *s,
                       const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const NDArray &lhs,
                       const NDArray &rhs,
                       OpReqType req,
                       const NDArray &output,
                       bool lhs_may_be_dense,
                       bool rhs_may_be_dense,
                       bool allow_inplace,
                       bool scatter);

  /*! \brief Binary op handling for lhr/rhs: RspDns, RspRsp, DnsRsp, or RspRsp->Dns result */
  template<typename OP>
  static void RspRspOp(mshadow::Stream<gpu> *s,
                       const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const NDArray &lhs,
                       const NDArray &rhs,
                       OpReqType req,
                       const NDArray &output,
                       bool lhs_may_be_dense,
                       bool rhs_may_be_dense,
                       bool allow_inplace,
                       bool scatter);

  /*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
  template<typename OP>
  static void CsrCsrOp(mshadow::Stream<cpu> *s,
                       const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const NDArray &lhs,
                       const NDArray &rhs,
                       OpReqType req,
                       const NDArray &output);

  /*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
  template<typename OP>
  static void CsrCsrOp(mshadow::Stream<gpu> *s,
                       const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const NDArray &lhs,
                       const NDArray &rhs,
                       OpReqType req,
                       const NDArray &output);

  /*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
  template<typename OP>
  static void DnsCsrDnsOp(mshadow::Stream<cpu> *s,
                          const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &lhs,
                          const NDArray &rhs,
                          OpReqType req,
                          const NDArray &output,
                          const bool reverse);

  /*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
  template<typename OP>
  static void DnsCsrDnsOp(mshadow::Stream<gpu> *s,
                          const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &lhs,
                          const NDArray &rhs,
                          OpReqType req,
                          const NDArray &output,
                          const bool reverse);

  /*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
  template<typename xpu, typename OP>
  static void DnsCsrCsrOp(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &lhs,
                          const NDArray &rhs,
                          OpReqType req,
                          const NDArray &output,
                          const bool reverse);

  /*! \brief DNS -op- RSP binary operator for non-canonical NDArray */
  template<typename xpu, typename OP>
  static void DnsRspDnsOp(mshadow::Stream<xpu> *s,
                          const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &lhs,
                          const NDArray &rhs,
                          OpReqType req,
                          const NDArray &output,
                          const bool reverse);

 public:
  /*!
   * \brief Rsp-op-Rsp operation which produces a dense result
   * \param attrs Attributes
   * \param dev_mask Device mask
   * \param dispatch_mode Dispatch Mode
   * \param in_attrs Input storage attributes
   * \param out_attrs Output storage attributes
   * \return true if handled
   */
  static bool SparseSparseWithDenseResult(const nnvm::NodeAttrs& attrs,
                                          int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs);

  /*!
   * \brief Allow one of the binary inputs to be dense and still produce a sparse output.
   *        Typically used for sparse * dense = sparse.
   *        Note: for csr, it dispatches to fallback other than csr, csr -> csr
   * \param attrs Attributes
   * \param dev_mask Device mask
   * \param dispatch_mode Dispatch Mode
   * \param in_attrs Input storage attributes
   * \param out_attrs Output storage attributes
   * \return true if handled
   */
  static bool PreferSparseStorageType(const nnvm::NodeAttrs& attrs,
                                      int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int> *in_attrs,
                                      std::vector<int> *out_attrs) {
    using namespace common;
    CHECK_EQ(in_attrs->size(), 2U) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), 1U) << " in operator " << attrs.name;
    const auto& lhs_stype = in_attrs->at(0);
    const auto& rhs_stype = in_attrs->at(1);
    auto& out_stype = out_attrs->at(0);
    bool dispatched = false;
    const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
    const auto dispatch_ex = invalid_ctx ? DispatchMode::kFComputeFallback :
                             DispatchMode::kFComputeEx;
    if (!dispatched && ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
      // dns, dns -> dns
      dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFCompute);
    }
    if (!dispatched && ContainsOnlyStorage(*in_attrs, kRowSparseStorage)) {
        // rsp, rsp -> rsp
        dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                         dispatch_mode, dispatch_ex);
    }
    if (!dispatched && ContainsOnlyStorage(*in_attrs, kCSRStorage)) {
        // csr, csr -> csr
        dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                         dispatch_mode, dispatch_ex);
    }
    if (!dispatched &&
        ((lhs_stype == kRowSparseStorage && rhs_stype == kDefaultStorage) ||
         (lhs_stype == kDefaultStorage && rhs_stype == kRowSparseStorage))) {
        // rsp, dns -> rsp
        // dns, rsp -> rsp
        dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                         dispatch_mode, dispatch_ex);
    }
    if (!dispatched &&
        ((lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) ||
         (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage))) {
        // csr, dns -> csr
        // dns, csr -> csr
        dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                         dispatch_mode, DispatchMode::kFComputeEx);
    }
    if (!dispatched) {
      dispatched = dispatch_fallback(out_attrs, dispatch_mode);
    }
    return dispatched;
  }


  /*!
   * \brief Allow one of the inputs to be dense and produce a dense output,
   *        for rsp inputs only support when both inputs are rsp type.
   * \param attrs Attributes
   * \param dev_mask Device mask
   * \param dispatch_mode Dispatch Mode
   * \param in_attrs Input storage attributes
   * \param out_attrs Output storage attributes
   * \return true if handled
   */
  template<bool cpu_only, bool rsp, bool csr>
  static bool PreferDenseStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
    using namespace common;
    CHECK_EQ(in_attrs->size(), 2);
    CHECK_EQ(out_attrs->size(), 1);
    const auto lhs_stype = (*in_attrs)[0];
    const auto rhs_stype = (*in_attrs)[1];
    bool dispatched = false;
    const bool invalid_ctx = cpu_only && dev_mask != mshadow::cpu::kDevMask;
    const auto dispatch_ex = invalid_ctx ? DispatchMode::kFComputeFallback :
                                           DispatchMode::kFComputeEx;
    if (!dispatched && ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
      // dns, dns ... -> dns
      dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFCompute);
    }
    if (!dispatched && rsp && ContainsOnlyStorage(*in_attrs, kRowSparseStorage)) {
      // rsp, rsp, ... -> rsp
      dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                       dispatch_mode, DispatchMode::kFComputeEx);
    }
    if (!dispatched && csr && ContainsOnlyStorage(*in_attrs, kCSRStorage)) {
      // csr, csr, ... -> csr
      dispatched = storage_type_assign(out_attrs, kCSRStorage,
                                       dispatch_mode, dispatch_ex);
    }
    if (!dispatched && ((lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage) ||
                        (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage))) {
      // dense, csr -> dense / csr, dense -> dense
      dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFComputeEx);
    }
    if (!dispatched && ((lhs_stype == kDefaultStorage && rhs_stype == kRowSparseStorage) ||
                        (lhs_stype == kRowSparseStorage && rhs_stype == kDefaultStorage))) {
      // dense, rsp -> dense / rsp, dense -> dense
      dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFComputeEx);
    }
    if (!dispatched) {
      dispatch_fallback(out_attrs, dispatch_mode);
    }
    return true;
  }


  /*!
   * \brief Backward pass computing input gradient using forward inputs
   * \param attrs Attributes
   * \param dev_mask Device mask
   * \param dispatch_mode Dispatch Mode
   * \param in_attrs Input storage attributes
   * \param out_attrs Output storage attributes
   * \return true if handled
   */
  static bool BackwardUseInStorageType(const nnvm::NodeAttrs& attrs,
                                       int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs);

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
          if (size != 0) {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, size,
            outputs[0].dptr<DType>(),
            inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
          }
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
          if (size != 0) {
            Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(s, size,
            outputs[0].dptr<DType>(),
            inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
          }
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
    using namespace common;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] == kNullOp) return;
    const auto lhs_stype = inputs[0].storage_type();
    const auto rhs_stype = inputs[1].storage_type();
    const auto out_stype = outputs[0].storage_type();
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    if ((ContainsOnlyStorage(inputs, kRowSparseStorage)) &&
        (out_stype == kRowSparseStorage || out_stype == kDefaultStorage)) {
      // rsp, rsp -> rsp
      // rsp, rsp -> dns
      RspRspOp<OP>(
            s, attrs, ctx, inputs[0], inputs[1], req[0], outputs[0], false, false, false, false);
    } else if (ContainsOnlyStorage(inputs, kCSRStorage) && out_stype == kCSRStorage) {
      // csr, csr -> csr
      CsrCsrOp<OP>(s, attrs, ctx, inputs[0], inputs[1], req[0], outputs[0]);
    } else if (((lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) ||
                (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage)) &&
                out_stype == kDefaultStorage) {
      const NDArray& dns = (lhs_stype == kDefaultStorage)? inputs[0] : inputs[1];
      const NDArray& csr = (lhs_stype == kCSRStorage)? inputs[0] : inputs[1];
      const bool reverse = (lhs_stype == kCSRStorage);

      DnsCsrDnsOp<OP>(s, attrs, ctx, dns, csr, req[0], outputs[0], reverse);
    } else if (((lhs_stype == kRowSparseStorage && rhs_stype == kDefaultStorage) ||
                (lhs_stype == kDefaultStorage && rhs_stype == kRowSparseStorage)) &&
                out_stype == kDefaultStorage) {
      const NDArray& dns = (lhs_stype == kDefaultStorage)? inputs[0] : inputs[1];
      const bool reverse = (lhs_stype == kRowSparseStorage);
      const NDArray& rsp = (reverse)? inputs[0] : inputs[1];

      DnsRspDnsOp<xpu, OP>(s, attrs, ctx, dns, rsp, req[0], outputs[0], reverse);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }

  /*! \brief ComputeEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, bool lhs_may_be_dense, bool rhs_may_be_dense>
  static void ComputeDnsLRValueEx(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] == kNullOp) return;
    const auto lhs_stype = inputs[0].storage_type();
    const auto rhs_stype = inputs[1].storage_type();
    const auto out_stype = outputs[0].storage_type();
    if ((out_stype == kRowSparseStorage || out_stype == kDefaultStorage) &&
        ((lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage) ||
         (lhs_stype == kRowSparseStorage && rhs_stype == kDefaultStorage) ||
         (lhs_stype == kDefaultStorage && rhs_stype == kRowSparseStorage)) &&
         lhs_may_be_dense && rhs_may_be_dense) {
      // rsp, rsp -> rsp
      // rsp, rsp -> dns
      // rsp, dns -> rsp
      // dns, rsp -> rsp
      // More than once dense not allowed (this will be checked in RspRspOp):
      //   rsp, dns -> dns  <-- NOT ALLOWED
      //   dns, rsp -> dns  <-- NOT ALLOWED
      mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
      RspRspOp<OP>(
            s, attrs, ctx, inputs[0], inputs[1],
            req[0], outputs[0], lhs_may_be_dense, rhs_may_be_dense, false, false);
    } else if (lhs_stype == kCSRStorage && rhs_stype == kCSRStorage) {
      ComputeEx<xpu, OP>(attrs, ctx, inputs, req, outputs);
    } else if (((lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) ||
                (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage)) &&
                out_stype == kCSRStorage) {
      const NDArray& dns = (lhs_stype == kDefaultStorage)? inputs[0] : inputs[1];
      const NDArray& csr = (lhs_stype == kCSRStorage)? inputs[0] : inputs[1];
      const bool reverse = (lhs_stype == kCSRStorage);

      DnsCsrCsrOp<xpu, OP>(attrs, ctx, dns, csr, req[0], outputs[0], reverse);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
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
      if (in_stype == lhs_stype && (in_stype == kRowSparseStorage || in_stype == kCSRStorage)) {
        CHECK_EQ(outputs[0].storage_type(), in_stype);
        // rsp -> rsp, _. op requires 0-input returns 0-output
        DCHECK_LT(std::fabs(static_cast<float>(LOP::Map(0))), 1e-5f);
        UnaryOp::ComputeEx<xpu, LOP>(attrs, ctx, inputs, req, {outputs[0]});
      } else {
        LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
      }
    }
    // rhs grad
    if (req[1] != kNullOp) {
      if (in_stype == rhs_stype && (in_stype == kRowSparseStorage || in_stype == kCSRStorage)) {
        CHECK_EQ(outputs[0].storage_type(), in_stype);
        // rsp -> _, rsp. op requires 0-input returns 0-output
        DCHECK_LT(std::fabs(static_cast<float>(ROP::Map(0))), 1e-5f);
        UnaryOp::ComputeEx<xpu, ROP>(attrs, ctx, inputs, req, {outputs[1]});
      } else {
        LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
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
    const auto out_grad_stype = inputs[0].storage_type();
    const auto lhs_grad_stype = outputs[0].storage_type();
    const auto rhs_grad_stype = outputs[1].storage_type();
    if (ContainsOnlyStorage(inputs, kRowSparseStorage) &&
        (lhs_grad_stype == kDefaultStorage || lhs_grad_stype == kRowSparseStorage) &&
        (rhs_grad_stype == kDefaultStorage || rhs_grad_stype == kRowSparseStorage)) {
      // rsp, rsp, rsp -> [dns, rsp], [dns, rsp]
      RspRspOpBackward<xpu, LOP, ROP, in0_ok_dense, in1_ok_dense, in2_ok_dense>(
        attrs, ctx, inputs, req, outputs, BackwardUseIn<xpu, LOP, ROP>);
    }
    if (((lhs_grad_stype == kDefaultStorage && rhs_grad_stype == kCSRStorage) ||
         (lhs_grad_stype == kCSRStorage && rhs_grad_stype == kDefaultStorage)) &&
        out_grad_stype == kDefaultStorage) {
      // dns, csr, dns -> [csr, dns] / csr, dns, dns -> [dns, csr]
      DnsCsrCsrOpBackward<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
    }
  }
};  // class ElemwiseBinaryOp

/*! \brief Binary launch */
#define MXNET_OPERATOR_REGISTER_BINARY(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"lhs", "rhs"};                \
    })                                                              \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "NDArray-or-Symbol", "first input")          \
  .add_argument("rhs", "NDArray-or-Symbol", "second input")

/*! \brief Binary launch, with FComputeEx for csr and rsp available */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(__name$, __kernel$)              \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                               \
  .set_attr<FInferStorageType>("FInferStorageType",                                     \
    ElemwiseStorageType<2, 1, true, true, true>)                                        \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>) \
  .set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */ \
    [](const NodeAttrs& attrs) { \
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})

/*! \brief Binary launch, with FComputeEx for csr and rsp available.
           when inputs contain both sparse and dense, sparse output is preferred. */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_PS(__name$, __kernel$)           \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                               \
  .set_attr<FInferStorageType>("FInferStorageType",                                     \
                               ElemwiseBinaryOp::PreferSparseStorageType)               \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>) \
  .set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */ \
    [](const NodeAttrs& attrs) { \
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})


/*! \brief Binary launch, dense result
 *         FInferStorageType attr is not set using this macro.
 *         By default DefaultStorageType is used.
 */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(__name$, __kernel$)                  \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                                      \
  .set_attr<FInferStorageType>("FInferStorageType",                                            \
                               ElemwiseBinaryOp::SparseSparseWithDenseResult)                  \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>)

/*! \brief Binary launch, with FComputeEx for prefer dense */
#define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_PD(__name$, __kernel$)              \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                               \
  .set_attr<FInferStorageType>("FInferStorageType",                                     \
    ElemwiseBinaryOp::PreferDenseStorageType<true, true, true>)                         \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>) \
  .set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */ \
    [](const NodeAttrs& attrs) { \
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

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
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./init_op.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

template<typename OP, int Req>
struct BinaryOp {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* lhs,
    const DType* rhs) {
    KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], rhs[i]));
  }
};

template<typename xpu, typename OP, typename DType>
void BinaryCompute_(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
    /DataType<DType>::kLanes);
  DType* out_dptr = outputs[0].dptr<DType>();
  DType* lhs_dptr = inputs[0].dptr<DType>();
  DType* rhs_dptr = inputs[1].dptr<DType>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    Kernel<BinaryOp<OP, Req>, xpu>::Launch(s, size, out_dptr, lhs_dptr, rhs_dptr);
  });
}

template<typename xpu, typename OP>
void BinaryCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BinaryCompute_<xpu, OP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

template<typename xpu, typename OP>
void BinaryComputeWithHalf2(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    BinaryCompute_<xpu, OP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

template<typename xpu, typename op>
void BinaryLaunch(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<op, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
  });
}

template<typename OP, int Req >
struct BinaryOpBackwardUseNone {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* igrad, const DType* ograd) {
    KERNEL_ASSIGN(igrad[i], Req, OP::Map(ograd[i]));
  }
};

template<typename xpu, typename LOP, typename ROP, typename DType>
void BinaryBackwardUseNone_(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
    /DataType<DType>::kLanes);
  DType* lgrad_dptr = outputs[0].dptr<DType>();
  DType* rgrad_dptr = outputs[1].dptr<DType>();
  DType* ograd_dptr = inputs[0].dptr<DType>();
  if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
    CHECK_EQ(ograd_dptr, lgrad_dptr);
  } else if (req[0] != kNullOp) {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req,
      {Kernel<BinaryOpBackwardUseNone<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr,
        ograd_dptr);});
  }
  if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
    CHECK_EQ(ograd_dptr, rgrad_dptr);
  } else if (req[1] != kNullOp) {
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req,
      {Kernel<BinaryOpBackwardUseNone<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr,
        ograd_dptr);});
  }
}

// TODO(haibin) This is a single-thread inefficient implementation
// This implementation only works on CPU
template<typename xpu, typename OP>
void BinaryComputeRspRspImpl(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) return;
  CHECK(req[0] == kWriteTo) << "only kWriteTo is supported for rowsparse elemwise_add";
  using namespace rowsparse;
  using namespace mshadow;
  auto &lhs = inputs[0];
  auto &rhs = inputs[1];
  auto &output = outputs[0];

  bool init_l = lhs.storage_initialized();
  bool init_r = rhs.storage_initialized();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // both inputs are zeros
  if (!init_l && !init_r) {
    NDArray out = output;
    FillZerosRspImpl(s, &out);
    return;
  }
  // Memory Estimation: This is (roughly) the number of result rows. We still
  // need to subtract the number of common rows
  unsigned int num_rows_l = lhs.aux_shape(kIdx)[0];
  unsigned int num_rows_r = rhs.aux_shape(kIdx)[0];
  unsigned int num_rows_total = num_rows_l + num_rows_r;
  auto row_len = output.shape().ProdShape(1, output.shape().ndim());
  output.CheckAndAlloc({Shape1(num_rows_total)});
  CHECK_GT(row_len, 0);
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    MSHADOW_TYPE_SWITCH(lhs.aux_type(kIdx), IType, {
      // Indices
      auto indices_l = lhs.aux_data(kIdx).dptr<IType>();
      auto indices_r = rhs.aux_data(kIdx).dptr<IType>();
      auto indices_out = output.aux_data(kIdx).dptr<IType>();
      // Data
      auto data_l = lhs.data().get_with_shape<cpu, 2, DType>(Shape2(num_rows_l, row_len), s);
      auto data_r = rhs.data().get_with_shape<cpu, 2, DType>(Shape2(num_rows_r, row_len), s);
      auto out = output.data().get_with_shape<cpu, 2, DType>(Shape2(num_rows_total, row_len), s);

      // TODO(haibin) A more appropriate way: Copy to output, then apply ops
      size_t iter_l = 0;
      size_t iter_r = 0;
      size_t iter_out = 0;
      int32_t num_common_rows = 0;
      while (iter_l < num_rows_l && iter_r < num_rows_r) {
        auto idx_l = indices_l[iter_l];
        auto idx_r = indices_r[iter_r];
        if (idx_l == idx_r) {
          // Same row
          indices_out[iter_out] = idx_l;
          Copy(out[iter_out], data_l[iter_l++], s);
          out[iter_out] += data_r[iter_r++];
          num_common_rows++;
        } else if (idx_l < idx_r) {
          // Left only
          indices_out[iter_out] = idx_l;
          Copy(out[iter_out], data_l[iter_l++], s);
        } else {
          // Right only
          indices_out[iter_out] = idx_r;
          Copy(out[iter_out], data_r[iter_r++], s);
        }
        iter_out++;
      }
      // Copying over the rest of the rows
      while (iter_l < num_rows_l) {
        indices_out[iter_out] = indices_l[iter_l];
        Copy(out[iter_out++], data_l[iter_l++], s);
      }
      while (iter_r < num_rows_r) {
        indices_out[iter_out] = indices_r[iter_r];
        Copy(out[iter_out++], data_r[iter_r++], s);
      }
      auto new_sshape = TShape(output.aux_shape(rowsparse::kIdx));
      CHECK_GT(new_sshape[0], num_common_rows);
      new_sshape[0] -= num_common_rows;
      output.set_aux_shape(rowsparse::kIdx, new_sshape);
    });
  });
}

template<typename xpu, typename OP>
void BinaryComputeEx(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  if (typeid(OP) == typeid(mshadow::op::plus)) {
    // If any input is dense, fallback to FCompute
    // TODO(haibin) implement dns + rsp in a separate kernel
    if (common::ContainsDefaultStorage(inputs)) {
      FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                           BinaryCompute<xpu, OP>, "BinaryCompute");
      return;
    }
    CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
    CHECK_EQ(inputs[1].storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
    BinaryComputeRspRspImpl<xpu, OP>(attrs, ctx, inputs, req, outputs);
    return;
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNone(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

// Only implemented for _backward_add for now
template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNoneRsp(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage);
  CHECK_EQ(outputs[0].storage_type(), kRowSparseStorage);
  CHECK_EQ(outputs[1].storage_type(), kRowSparseStorage);
  CHECK(typeid(LOP) == typeid(mshadow_op::identity));
  CHECK(typeid(ROP) == typeid(mshadow_op::identity));
  TShape shape = inputs[0].aux_shape(rowsparse::kIdx);
  outputs[0].CheckAndAlloc({shape});
  outputs[1].CheckAndAlloc({shape});
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), IType, {
      auto lgrad_idx = outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
      auto rgrad_idx = outputs[1].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
      auto ograd_idx = inputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
      auto lgrad = outputs[0].data().FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> rgrad = outputs[1].data().FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> ograd = inputs[0].data().FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
      ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
      ASSIGN_DISPATCH(lgrad_idx, req[0], F<LOP>(ograd_idx));
      ASSIGN_DISPATCH(rgrad_idx, req[1], F<ROP>(ograd_idx));
    });
  });
}
// Only implemented for _backward_add for now
template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  auto stype = inputs[0].storage_type();
  CHECK_EQ(stype, kRowSparseStorage) << "Not implemented yet";
  BinaryBackwardUseNoneRsp<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  // TODO(haibin) fallback for kDefaultStorage
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNoneWithHalf2(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

template<typename OP, int Req>
struct BinaryOpBackwardUseIn {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* igrad,
    const DType* ograd, const DType* lhs, const DType* rhs) {
    KERNEL_ASSIGN(igrad[i], Req, ograd[i]*OP::Map(lhs[i], rhs[i]));
  }
};

template<typename xpu, typename LOP, typename ROP, typename DType>
void BinaryBackwardUseIn_(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
    /DataType<DType>::kLanes);
  DType* lgrad_dptr = outputs[0].dptr<DType>();
  DType* rgrad_dptr = outputs[1].dptr<DType>();
  DType* ograd_dptr = inputs[0].dptr<DType>();
  DType* lhs_dptr = inputs[1].dptr<DType>();
  DType* rhs_dptr = inputs[2].dptr<DType>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req,
    {Kernel<BinaryOpBackwardUseIn<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr,
      lhs_dptr, rhs_dptr);});
  MXNET_ASSIGN_REQ_SWITCH(req[1], Req,
    {Kernel<BinaryOpBackwardUseIn<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr,
      lhs_dptr, rhs_dptr);});
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseInWithHalf2(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
  });
}

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

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template<typename OP>
struct BinaryOp {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const OpReqType req, const DType* lhs,
    const DType* rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }
};

// Version of BinaryCompute() that supports half2 type
template<typename xpu, typename OP>
void BinaryComputeWithHalf2(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
      /DataType<DType>::kPack);
    DType* out_dptr = outputs[0].dptr<DType>();
    DType* lhs_dptr = inputs[0].dptr<DType>();
    DType* rhs_dptr = inputs[1].dptr<DType>();
    Kernel<BinaryOp<OP>, xpu>::Launch(s, size, out_dptr, req[0], lhs_dptr, rhs_dptr);
  });
}

template<typename xpu, typename OP>
void BinaryCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rhs = inputs[1].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, rhs));
  });
}

template<typename LOP, typename ROP>
struct BinaryOpBackwardUseNone {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* lgrad, DType* rgrad,
    const OpReqType req0, const OpReqType req1, const DType* ograd) {
    KERNEL_ASSIGN(lgrad[i], req0, LOP::Map(ograd[i]));
    KERNEL_ASSIGN(rgrad[i], req1, ROP::Map(ograd[i]));
  }
};

// Version of BinaryBackwardUseNone that supports half2 type
template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNoneWithHalf2(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
      /DataType<DType>::kPack);
    DType* lgrad_dptr = outputs[0].dptr<DType>();
    DType* rgrad_dptr = outputs[1].dptr<DType>();
    DType* ograd_dptr = inputs[0].dptr<DType>();
    Kernel<BinaryOpBackwardUseNone<LOP, ROP>, xpu>::Launch(s, size, lgrad_dptr, rgrad_dptr,
      req[0], req[1], ograd_dptr);
  });
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNone(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
    ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
  });
}

// template<typename xpu, typename LOP, typename ROP>
// void BinaryBackwardUseOut(const nnvm::NodeAttrs& attrs,
//                           const OpContext& ctx,
//                           const std::vector<TBlob>& inputs,
//                           const std::vector<OpReqType>& req,
//                           const std::vector<TBlob>& outputs) {
//   using namespace mshadow;
//   using namespace mshadow::expr;
//   Stream<xpu> *s = ctx.get_stream<xpu>();
//   MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
//     Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> out = inputs[1].FlatTo1D<xpu, DType>(s);
//     ASSIGN_DISPATCH(lgrad, req[0], ograd*F<LOP>(out));
//     ASSIGN_DISPATCH(rgrad, req[1], ograd*F<ROP>(out));
//   });
// }

template<typename LOP, typename ROP>
struct BinaryOpBackwardUseIn {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* lgrad, DType* rgrad,
    const OpReqType req0, const OpReqType req1, const DType* ograd,
    const DType* lhs, const DType* rhs) {
    KERNEL_ASSIGN(lgrad[i], req0, ograd[i]*LOP::Map(lhs[i], rhs[i]));
    KERNEL_ASSIGN(rgrad[i], req1, ograd[i]*ROP::Map(lhs[i], rhs[i]));
  }
};

// Version of BinaryBackwardUseIn that supports half2 type
template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseInWithHalf2(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
    int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
      /DataType<DType>::kPack);
    DType* lgrad_dptr = outputs[0].dptr<DType>();
    DType* rgrad_dptr = outputs[1].dptr<DType>();
    DType* ograd_dptr = inputs[0].dptr<DType>();
    DType* lhs_dptr = inputs[1].dptr<DType>();
    DType* rhs_dptr = inputs[2].dptr<DType>();
    Kernel<BinaryOpBackwardUseIn<LOP, ROP>, xpu>::Launch(s, size, lgrad_dptr, rgrad_dptr,
      req[0], req[1], ograd_dptr, lhs_dptr, rhs_dptr);
  });
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rhs = inputs[2].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(lgrad, req[0], ograd*F<LOP>(lhs, rhs));
    ASSIGN_DISPATCH(rgrad, req[1], ograd*F<ROP>(lhs, rhs));
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
  .add_argument("lhs", "NDArray-or-Symbol", "first input")                    \
  .add_argument("rhs", "NDArray-or-Symbol", "second input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

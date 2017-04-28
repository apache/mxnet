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
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

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
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
    /DataType<DType>::kPack);
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

// template<typename xpu, typename OP>
// void BinaryCompute(const nnvm::NodeAttrs& attrs,
//                    const OpContext& ctx,
//                    const std::vector<TBlob>& inputs,
//                    const std::vector<OpReqType>& req,
//                    const std::vector<TBlob>& outputs) {
//   using namespace mshadow;
//   using namespace mshadow::expr;
//   Stream<xpu> *s = ctx.get_stream<xpu>();
//   MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
//     Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> rhs = inputs[1].FlatTo1D<xpu, DType>(s);
//     ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, rhs));
//   });
// }

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
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
    /DataType<DType>::kPack);
  DType* lgrad_dptr = outputs[0].dptr<DType>();
  DType* rgrad_dptr = outputs[1].dptr<DType>();
  DType* ograd_dptr = inputs[0].dptr<DType>();
  if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
    CHECK_EQ(ograd_dptr, lgrad_dptr);
  } else if (req[0] != kNullOp) {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req,
      {Kernel<BinaryOpBackwardUseNone<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr,
        ograd_dptr);}
      );
  }
  if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
    CHECK_EQ(ograd_dptr, rgrad_dptr);
  } else if (req[1] != kNullOp) {
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req,
      {Kernel<BinaryOpBackwardUseNone<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr,
        ograd_dptr);}
      );
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

// template<typename xpu, typename LOP, typename ROP>
// void BinaryBackwardUseNone(const nnvm::NodeAttrs& attrs,
//                            const OpContext& ctx,
//                            const std::vector<TBlob>& inputs,
//                            const std::vector<OpReqType>& req,
//                            const std::vector<TBlob>& outputs) {
//   using namespace mshadow;
//   using namespace mshadow::expr;
//   Stream<xpu> *s = ctx.get_stream<xpu>();
//   MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
//     Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
//     if (std::is_same<LOP, mshadow_op::identity>::value) {
//       if (req[0] == kWriteInplace) {
//         CHECK_EQ(ograd.dptr_, lgrad.dptr_);
//       } else {
//         ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
//       }
//     } else {
//       ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
//     }
//     if (std::is_same<ROP, mshadow_op::identity>::value) {
//       if (req[1] == kWriteInplace) {
//         CHECK_EQ(ograd.dptr_, rgrad.dptr_);
//       } else {
//         ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
//       }
//     } else {
//       ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
//     }
//   });
// }

template<typename LOP, typename ROP, int Req0, int Req1>
struct BinaryOpBackwardUseIn {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* lgrad, DType* rgrad,
    const DType* ograd, const DType* lhs, const DType* rhs) {
    KERNEL_ASSIGN(lgrad[i], Req0, ograd[i]*LOP::Map(lhs[i], rhs[i]));
    KERNEL_ASSIGN(rgrad[i], Req1, ograd[i]*ROP::Map(lhs[i], rhs[i]));
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
  int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kPack - 1)
    /DataType<DType>::kPack);
  DType* lgrad_dptr = outputs[0].dptr<DType>();
  DType* rgrad_dptr = outputs[1].dptr<DType>();
  DType* ograd_dptr = inputs[0].dptr<DType>();
  DType* lhs_dptr = inputs[1].dptr<DType>();
  DType* rhs_dptr = inputs[2].dptr<DType>();
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req0,
    {MXNET_ASSIGN_REQ_SWITCH(req[1], Req1,
      {Kernel<BinaryOpBackwardUseIn<LOP, ROP, Req0, Req1>, xpu>::Launch(s, size, lgrad_dptr,
        rgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);}
      );}
    );
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

// template<typename xpu, typename LOP, typename ROP>
// void BinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
//                          const OpContext& ctx,
//                          const std::vector<TBlob>& inputs,
//                          const std::vector<OpReqType>& req,
//                          const std::vector<TBlob>& outputs) {
//   using namespace mshadow;
//   using namespace mshadow::expr;
//   Stream<xpu> *s = ctx.get_stream<xpu>();
//   MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
//     Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> lhs = inputs[1].FlatTo1D<xpu, DType>(s);
//     Tensor<xpu, 1, DType> rhs = inputs[2].FlatTo1D<xpu, DType>(s);
//     ASSIGN_DISPATCH(lgrad, req[0], ograd*F<LOP>(lhs, rhs));
//     ASSIGN_DISPATCH(rgrad, req[1], ograd*F<ROP>(lhs, rhs));
//   });
// }

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

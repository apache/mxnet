/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_broadcast_op.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "../operator_common.h"
#include "../broadcast_reduce_op_common.h"

namespace mxnet {
namespace op {
inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& lhs = (*in_attrs)[0];
  TShape& rhs = (*in_attrs)[1];
  if (lhs.ndim() == 0) {
    if (rhs.ndim() == 0) {
      if ((*out_attrs)[0].ndim() == 0) return false;
      rhs = (*out_attrs)[0];
    }
    lhs = rhs;
  } else if (rhs.ndim() == 0) {
    rhs = lhs;
  }
  if (lhs == rhs) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  int bl = out.ndim() - lhs.ndim();
  int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl) l = lhs[i-bl];
    if (i >= br) r = rhs[i-br];
    if (l != r) {
      CHECK(l == 1 || r == 1) 
        << "operands could not be broadcast together with shapes " << lhs << " " << rhs;
      out[i] = std::max(l, r);
    } else {
      out[i] = l;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return true;
}

bool BinaryBroadcastShapeCompact(const TShape& lshape, const TShape& rshape, const TShape& oshape,
                                 TShape *new_lshape, TShape *new_rshape, TShape *new_oshape) {
  if (lshape == rshape) return false;
  *new_lshape = TShape(oshape.ndim());
  *new_rshape = TShape(oshape.ndim());
  *new_oshape = TShape(oshape.ndim());
  int bl = oshape.ndim() - lshape.ndim();
  int br = oshape.ndim() - rshape.ndim();
  int j = 0, prod = 1;
  for (int i = 0; i < oshape.ndim(); ++i) {
    int l = 1, r = 1, o = oshape[i];
    if (i >= bl) l = lshape[i-bl];
    if (j >= br) r = rshape[i-br];
    if (l != r) {
      if (prod > 1) {
        (*new_lshape)[j] = (*new_rshape)[j] = (*new_oshape)[j] = prod;
        prod = 1; ++j;
      }
      (*new_lshape)[j] = l;
      (*new_rshape)[j] = r;
      (*new_oshape)[j] = o;
      ++j;
    } else {
      prod *= l;
    }
  }
  if (prod > 1) {
    (*new_lshape)[j] = (*new_rshape)[j] = (*new_oshape)[j] = prod;
    ++j;
  }
  LOG(INFO) << "o:" << lshape << rshape << oshape;
  LOG(INFO) << "n:" << *new_lshape << *new_rshape << *new_oshape;
  if (j != oshape.ndim()) {
    new_lshape->assign(&(*new_lshape)[0], &(*new_lshape)[j]);
    new_rshape->assign(&(*new_rshape)[0], &(*new_rshape)[j]);
    new_oshape->assign(&(*new_oshape)[0], &(*new_oshape)[j]);
  }
  LOG(INFO) << "o:" << lshape << rshape << oshape;
  LOG(INFO) << "n:" << *new_lshape << *new_rshape << *new_oshape;
  return true;
}

template<typename xpu, typename OP>
void BinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  TShape new_lshape, new_rshape, new_oshape;
  bool need_bc = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!need_bc) {
    BinaryCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MXNET_NDIM_SWITCH(new_oshape.ndim(), ndim, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, ndim, DType> out =
          outputs[0].get_with_shape<xpu, ndim, DType>(new_oshape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> lhs =
          inputs[0].get_with_shape<xpu, ndim, DType>(new_lshape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> rhs =
          inputs[1].get_with_shape<xpu, ndim, DType>(new_rshape.get<ndim>(), s);
        ASSIGN_DISPATCH(out, req[0], F<OP>(broadcast_to(lhs, new_oshape), broadcast_to(rhs, new_oshape)));
      });
    })
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  TShape new_lshape, new_rshape, new_oshape;
  bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!need_bc) {
    BinaryBackwardUseNone<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MXNET_NDIM_SWITCH(new_oshape.ndim(), ndim, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, ndim, DType> ograd =
          inputs[0].get_with_shape<xpu, ndim, DType>(new_oshape.get<ndim>(), s);
        Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
        ReduceToAssign<red::sum>(lgrad, req[0], new_lshape, F<LOP>(ograd));
        ReduceToAssign<red::sum>(rgrad, req[1], new_rshape, F<ROP>(ograd));
      });
    })
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseIn(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  TShape new_lshape, new_rshape, new_oshape;
  bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!need_bc) {
    BinaryBackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MXNET_NDIM_SWITCH(new_oshape.ndim(), ndim, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, ndim, DType> ograd =
          inputs[0].get_with_shape<xpu, ndim, DType>(new_oshape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> lhs =
          inputs[1].get_with_shape<xpu, ndim, DType>(new_lshape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> rhs =
          inputs[2].get_with_shape<xpu, ndim, DType>(new_rshape.get<ndim>(), s);
        Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
        ReduceToAssign<red::sum>(lgrad, req[0], new_lshape,
          ograd*F<LOP>(broadcast_to(lhs, new_oshape), broadcast_to(rhs, new_oshape)));
        ReduceToAssign<red::sum>(rgrad, req[1], new_rshape,
          ograd*F<ROP>(broadcast_to(lhs, new_oshape), broadcast_to(rhs, new_oshape)));
      });
    })
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseOut(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  TShape new_lshape, new_rshape, new_oshape;
  bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!need_bc) {
    BinaryBackwardUseOut<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MXNET_NDIM_SWITCH(new_oshape.ndim(), ndim, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, ndim, DType> ograd =
          inputs[0].get_with_shape<xpu, ndim, DType>(new_oshape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> out =
          inputs[1].get_with_shape<xpu, ndim, DType>(new_oshape.get<ndim>(), s);
        Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
        ReduceToAssign<red::sum>(lgrad, req[0], new_lshape, ograd*F<LOP>(out));
        ReduceToAssign<red::sum>(rgrad, req[1], new_rshape, ograd*F<ROP>(out));
      });
    })
  }
}


#define MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(name)           \
  NNVM_REGISTER_OP(name)                                         \
  .set_num_inputs(2)                                             \
  .set_num_outputs(1)                                            \
  .attr<nnvm::FInferShape>("FInferShape", BinaryBroadcastShape)  \
  .attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)      \
  .attr<nnvm::FInplaceOption>("FInplaceOption",                  \
    [](const NodeAttrs& attrs){                                  \
      return std::vector<std::pair<int, int> >{{0, 1}};          \
    })                                                           \
  .add_argument("lhs", "NDArray", "first input")                 \
  .add_argument("rhs", "NDArray", "second input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

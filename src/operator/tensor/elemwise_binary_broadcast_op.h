/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_broadcast_op.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "../operator_common.h"
#include "broadcast_reduce-inl.h"

namespace mxnet {
namespace op {
inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& lhs = (*in_attrs)[0];
  TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (lhs.ndim() == 0 || rhs.ndim() == 0) return false;

  if (lhs == rhs) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  index_t bl = out.ndim() - lhs.ndim();
  index_t br = out.ndim() - rhs.ndim();
  for (index_t i = 0; i < out.ndim(); ++i) {
    index_t l = 1, r = 1;
    if (i >= bl) l = lhs[i-bl];
    if (i >= br) r = rhs[i-br];
    if (l != r) {
      if (l == 0 || r == 0) {
        out[i] = 0;
      } else {
        CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << lhs << " " << rhs;
        out[i] = std::max(l, r);
      }
    } else {
      out[i] = l;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return true;
}

#define BROADCAST_NDIM_SWITCH(ndim, NDim, ...)  \
  if (ndim <= 2) {                    \
    const int NDim = 2;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= 4) {             \
    const int NDim = 4;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= broadcast::MAX_DIM) {  \
    const int NDim = broadcast::MAX_DIM;    \
    {__VA_ARGS__}                     \
  } else {                            \
    LOG(FATAL) << "NDim too large ";  \
  }

inline int BinaryBroadcastShapeCompact(const TShape& lshape, const TShape& rshape,
                                       const TShape& oshape, TShape *new_lshape,
                                       TShape *new_rshape, TShape *new_oshape) {
  if (lshape == rshape) return 0;
  index_t odim = std::max<index_t>(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = TShape(odim);
  *new_rshape = TShape(odim);
  *new_oshape = TShape(odim);
  index_t bl = oshape.ndim() - lshape.ndim();
  index_t br = oshape.ndim() - rshape.ndim();
  index_t j = 0, lprod = 1, rprod = 1, oprod = 1;
  for (index_t i = 0; i < oshape.ndim(); ++i) {
    index_t l = 1, r = 1, o = oshape[i];
    if (i >= bl) l = lshape[i-bl];
    if (i >= br) r = rshape[i-br];
    if ((lprod != rprod || l != r) &&
        lprod*l > 1 && rprod*r > 1) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1; ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  if (j <= broadcast::MAX_DIM) {
    BROADCAST_NDIM_SWITCH(j, NDim, {
      new_lshape->assign(&(*new_lshape)[0], &(*new_lshape)[NDim]);
      new_rshape->assign(&(*new_rshape)[0], &(*new_rshape)[NDim]);
      new_oshape->assign(&(*new_oshape)[0], &(*new_oshape)[NDim]);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape << " " << rshape;
  }
  return j;
}

template<typename xpu, typename OP>
void BinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace broadcast;
  TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    BinaryCompute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        BinaryBroadcastComputeImpl<NDim, DType, OP>(s, req[0], inputs[0].reshape(new_lshape),
          inputs[1].reshape(new_rshape), outputs[0].reshape(new_oshape));
      });
    });
  }
}

template<typename Reducer, typename xpu, typename SrcExp, int ndim, typename DType>
void ReduceToAssign(mshadow::Tensor<xpu, ndim, DType> out,
                    const OpReqType req, const SrcExp &src_) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Shape<ndim> src_shape = ShapeCheck<ndim, SrcExp>::Check(src_);
  Shape<ndim> axes;
  index_t reducing_size = 1, remaining_size = 1;
  int i = 0;
  for (int k = 0; k < ndim; ++k)
    if (src_shape[k] != out.shape_[k])
      ++i;
  for (int j = ndim-1, k = ndim-1; k >= 0; --k) {
    if (src_shape[k] == out.shape_[k]) {
      axes[j--] = k;
      remaining_size *= src_shape[k];
    } else {
      axes[--i] = k;
      reducing_size *= src_shape[k];
    }
  }
  if (reducing_size == 1) {
    ASSIGN_DISPATCH(out, req, F<mshadow_op::identity>(src_));
  } else {
    ASSIGN_DISPATCH(out.FlatTo1D(), req,
      (reduce_except_dim<1, Reducer>(reshape(transpose(src_, axes),
      Shape2(reducing_size, remaining_size)))));
  }
}

template<typename Reducer, typename xpu, typename SrcExp, typename DType>
void ReduceToAssign(mshadow::Tensor<xpu, 2, DType> out, const OpReqType req, const SrcExp &src_) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Shape<2> src_shape = ShapeCheck<2, SrcExp>::Check(src_);
  if (src_shape == out.shape_) {
    ASSIGN_DISPATCH(out, req, F<mshadow_op::identity>(src_));
  } else if (src_shape[0] == out.shape_[0]) {
    ASSIGN_DISPATCH(out.FlatTo1D(), req, (reduce_except_dim<0, Reducer>(src_)));
  } else if (src_shape[1] == out.shape_[1]) {
    ASSIGN_DISPATCH(out.FlatTo1D(), req, (reduce_except_dim<1, Reducer>(src_)));
  } else {
    ASSIGN_DISPATCH(out.FlatTo1D(), req,
      (reduce_except_dim<1, Reducer>(reshape(src_,
      Shape2(src_shape.Size(), 1)))));
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace broadcast;
  TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    BinaryBackwardUseNone<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob lhs = outputs[0].reshape(new_lshape);
      const TBlob rhs = outputs[1].reshape(new_rshape);
      const TBlob out = inputs[0].reshape(new_oshape);
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        // Request temporary storage
        size_t workspace_size_l = ReduceWorkspaceSize<NDim, DType>(s, lhs, req[0], out);
        size_t workspace_size_r = ReduceWorkspaceSize<NDim, DType>(s, rhs, req[1], out);
        size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
        Tensor<xpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
        Reduce<red::sum, NDim, DType, LOP>(s, lhs, req[0], workspace, out);
        Reduce<red::sum, NDim, DType, ROP>(s, rhs, req[1], workspace, out);
      });
    });
  }
}

template<typename xpu, int ndim, typename DType, typename LOP, typename ROP>
inline void BinaryBroadcastBackwardUseInImpl(const OpContext& ctx,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs,
                                             const TShape& new_lshape,
                                             const TShape& new_rshape,
                                             const TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob lhs = inputs[1].reshape(new_lshape);
  const TBlob rhs = inputs[2].reshape(new_rshape);
  size_t workspace_size_l = ReduceWorkspaceSize<ndim, DType>(s, lgrad, req[0], ograd, lhs, rhs);
  size_t workspace_size_r = ReduceWorkspaceSize<ndim, DType>(s, rgrad, req[1], ograd, lhs, rhs);
  size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  Reduce<red::sum, ndim, DType, mshadow::op::mul, LOP>(s, lgrad, req[0], workspace,
    ograd, lhs, rhs);
  Reduce<red::sum, ndim, DType, mshadow::op::mul, ROP>(s, rgrad, req[0], workspace,
    ograd, lhs, rhs);
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseIn(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  TShape new_lshape, new_rshape, new_oshape;
  bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                             &new_lshape, &new_rshape, &new_oshape);
  if (!need_bc) {
    BinaryBackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
        BinaryBroadcastBackwardUseInImpl<xpu, NDim, DType, LOP, ROP>(
          ctx, inputs, req, outputs, new_lshape, new_rshape, new_oshape);
      });
    });
  }
}

#undef BROADCAST_NDIM_SWITCH

#define MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(name)                \
  NNVM_REGISTER_OP(name)                                              \
  .set_num_inputs(2)                                                  \
  .set_num_outputs(1)                                                 \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                 \
    [](const NodeAttrs& attrs) {                                      \
      return std::vector<std::string>{"lhs", "rhs"};                  \
    })                                                                \
  .set_attr<nnvm::FInferShape>("FInferShape", BinaryBroadcastShape)   \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)       \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                   \
    [](const NodeAttrs& attrs){                                       \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};       \
    })                                                                \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")                      \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

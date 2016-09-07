/*!
* Copyright (c) 2016 by Contributors
* \file elemwise_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#define MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
template<int n_in, int n_out, typename AttrType,
         bool (*is_none)(const AttrType&), bool reverse_infer>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in);
  CHECK_EQ(out_attrs->size(), n_out);
  bool found = false;
  AttrType dattr;
  for (int i = 0; i < n_in; ++i) {
    if (!is_none((*in_attrs)[i])) {
      dattr = (*in_attrs)[i];
      found = true;
      break;
    }
  }
  if (reverse_infer && !found) {
    for (int i = 0; i < n_out; ++i) {
      if (!is_none((*out_attrs)[i])) {
        dattr = (*out_attrs)[i];
        found = true;
        break;
      }
    }
  }
  if (!found) {
    return false;
  }
  for (int i = 0; i < n_in; ++i) {
    if (is_none((*in_attrs)[i])) {
      (*in_attrs)[i] = dattr;
    } else if ((*in_attrs)[i] != dattr) {
      LOG(FATAL) << "Incompatible attr in node " << attrs.name << " at " << i << "-th input: "
                 << "expected " << dattr << ", got " << (*in_attrs)[i];
    }
  }
  for (int i = 0; i < n_out; ++i) {
    if (is_none((*out_attrs)[i])) {
      (*out_attrs)[i] = dattr;
    } else if ((*out_attrs)[i] != dattr) {
      LOG(FATAL) << "Incompatible attr in node " << attrs.name << " at " << i << "-th output: "
                 << "expected " << dattr << ", got " << (*out_attrs)[i];
    }
  }
  return true;
}

inline bool shape_is_none(const TShape& x) {
  return  x.ndim() == 0;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  return ElemwiseAttr<n_in, n_out, TShape, shape_is_none, true>(
    attrs, in_attrs, out_attrs);
}

inline bool type_is_none(const int& x) {
  return x == -1;
}

template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  return ElemwiseAttr<n_in, n_out, int, type_is_none, true>(
    attrs, in_attrs, out_attrs);
}

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
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
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  CHECK_EQ(inputs[1].type_flag_, outputs[0].type_flag_);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rhs = inputs[1].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, rhs));
  });
}

#define MXNET_OPERATOR_REGISTER_UNARY(name)                     \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                 \
      return std::vector<std::pair<int, int> >{{0, 0}};         \
    })                                                          \
  .add_argument("src", "NDArray", "Source input")

#define MXNET_OPERATOR_REGISTER_BINARY(name)                    \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(2)                                            \
  .set_num_outputs(1)                                           \
  .attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
  .attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
  .attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                 \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}}; \
    })                                                          \
  .add_argument("lhs", "NDArray", "first input")                \
  .add_argument("rhs", "NDArray", "second input")

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_

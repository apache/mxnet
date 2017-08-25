/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_fully_connected-inl.h"
#if MXNET_USE_MKLDNN == 1 
#include <mkl_memory.h>
#include "../mkl/mkldnn_memory-inl.h"
#include "../mkl/mkl_util-inl.h"
#include "./mkl/mkldnn_quantized_fully_connected-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizedFullyConnectedParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedFullyConnectedParam& param) {
  //LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
#if MXNET_USE_MKLDNN == 1 
        return new MKLDNNQuantFullyConnectedOp<uint8_t, int8_t, int32_t, uint8_t>(param);
        //return new MKLDNNQuantFullyConnectedOp<uint8_t, int32_t, uint8_t>(param);
      if (enableMKLDNNWarnGenerated())
        LOG(INFO) << "MKLDNNQuantFullyConnectedOp Skip MKL DNN optimization";
#endif
  
  return op;
}

Operator *QuantizedFullyConnectedProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(quantized_fully_connected, QuantizedFullyConnectedProp)
.add_argument("data", "NDArray-or-Symbol", "matrix a")
.add_argument("weight", "NDArray-or-Symbol", "matrix b")
.add_argument("bias", "NDArray-or-Symbol", "matrix b")
.add_argument("min_data", "NDArray-or-Symbol", "minimum value of matrix a")
.add_argument("max_data", "NDArray-or-Symbol", "maximum value of matrix a")
.add_argument("min_weight", "NDArray-or-Symbol", "minimum value of matrix b")
.add_argument("max_weight", "NDArray-or-Symbol", "maximum value of matrix b")
.add_argument("min_bias", "NDArray-or-Symbol", "minimum value of matrix b")
.add_argument("max_bias", "NDArray-or-Symbol", "maximum value of matrix b")
.add_arguments(QuantizedFullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(quantized_fully_connected)
.set_attr<TQuantizationNeedShrink>("TQuantizationNeedShrink", false);

NNVM_REGISTER_OP(FullyConnected)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const nnvm::NodeAttrs& attrs = n->attrs;
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("quantized_fully_connected");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet

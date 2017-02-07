/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"
#include <nnvm/op_attr_types.h>
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNormParam param, int dtype) {
#if MXNET_USE_MKL2017 == 1
  if (!param.use_global_stats) {
    return new MKLBatchNormOp<cpu, float>(param);
  } else {
    if (enableMKLWarnGenerated())
      LOG(INFO) << MKLBatchNormOp<cpu, float>::getName() << " Skip MKL optimization";
  }
#endif
  return new BatchNormOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
.describe("Apply batch normalization to input.")
.add_argument("data", "Symbol", "Input data to batch normalization")
.add_argument("gamma", "Symbol", "gamma matrix")
.add_argument("beta", "Symbol", "beta matrix")
.add_arguments(BatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(BatchNorm)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 3) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      } else if (index == 4) {
        var->attrs.dict["__init__"] = "[\"one\", {}]";
      }
    });

}  // namespace op
}  // namespace mxnet


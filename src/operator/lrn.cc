/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn.cc
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_lrn-inl.h"
#endif
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(LRNParam param, int dtype) {
#if MXNET_USE_MKL2017 == 1
  return new MKLLRNOp<cpu, float>(param);
#endif
  return new LocalResponseNormOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* LocalResponseNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(LRNParam);

MXNET_REGISTER_OP_PROPERTY(LRN, LocalResponseNormProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_arguments(LRNParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

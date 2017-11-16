/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution_v1.cc
 * \brief
 * \author Bing Xu
*/

#include "./convolution_v1-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ConvolutionV1Param);

template<>
Operator* CreateOp<cpu>(ConvolutionV1Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionV1Op<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionV1Prop::CreateOperatorEx(Context ctx,
                                              std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(Convolution_v1, ConvolutionV1Prop)
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionV1Op.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(ConvolutionV1Param::__FIELDS__())
.describe("This operator is DEPRECATED."
          " Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

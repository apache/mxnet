/*!
 * Copyright (c) 2015 by Contributors
 * \file Ifft-inl.h
 * \brief
 * \author Chen Zhu
*/

#include "./ifft-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(IFFTParam param, int dtype) {
    LOG(FATAL) << "ifft is only available for GPU.";
    return NULL;
}

Operator *IFFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(IFFTParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_ifft, IFFTProp)
.describe("Apply IFFT to input.")
.add_argument("data", "NDArray-or-Symbol", "Input data to the IFFTOp.")
.add_arguments(IFFTParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

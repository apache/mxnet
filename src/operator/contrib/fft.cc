/*!
 * Copyright (c) 2015 by Contributors
 * \file fft-inl.h
 * \brief
 * \author Chen Zhu
*/
#include "./fft-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(FFTParam param, int dtype) {
    LOG(FATAL) << "fft is only available for GPU.";
    return NULL;
}

Operator *FFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(FFTParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_fft, FFTProp)
.describe("Apply FFT to input.")
.add_argument("data", "NDArray-or-Symbol", "Input data to the FFTOp.")
.add_arguments(FFTParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

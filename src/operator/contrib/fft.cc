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
.describe(R"code(Apply 1D FFT to input"

.. note:: `fft` is only available on GPU.

Currently accept 2 input data shapes: (N, d) or (N1, N2, N3, d), data can only be real numbers.
The output data has shape: (N, 2*d) or (N1, N2, N3, 2*d). The format is: [real0, imag0, real1, imag1, ...].

Example::
   data = np.random.normal(0,1,(3,4))
   out = mx.contrib.ndarray.fft(data = mx.nd.array(data,ctx = mx.gpu(0)))

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the FFTOp.")
.add_arguments(FFTParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

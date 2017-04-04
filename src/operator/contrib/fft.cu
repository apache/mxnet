/*!
 * Copyright (c) 2017 by Contributors
 * \file fft-inl.h
 * \brief
 * \author Chen Zhu
 */
#include "./fft-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(FFTParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new FFTOp<gpu, DType>(param);
    })
    return op;
}
}  // namespace op
}  // namespace mxnet

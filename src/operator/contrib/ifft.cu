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
Operator* CreateOp<gpu>(IFFTParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new IFFTOp<gpu, DType>(param);
    })
    return op;
}
}  // namespace op
}  // namespace mxnet

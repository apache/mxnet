/*!
 * Copyright (c) 2017 by Contributors
 * \file yolo_output.cu
 * \brief yolo output layer
 * \author Joshua Zhang
 */
#include "./yolo_output-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(YoloOutputParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new YoloOutputOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

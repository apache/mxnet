/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_convolution.cu
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
 */

#include "./deformable_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {

template <>
Operator* CreateOp<gpu>(DeformableConvolutionParam param,
                        int dtype,
                        mxnet::ShapeVector* in_shape,
                        mxnet::ShapeVector* out_shape,
                        Context ctx) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new DeformableConvolutionOp<gpu, DType>(param); })
  return op;
}

}  // namespace op
}  // namespace mxnet

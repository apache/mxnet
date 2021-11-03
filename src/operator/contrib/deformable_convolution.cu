/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_convolution.cu
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
 *
 * Code from https://github.com/msracver/Deformable-ConvNets/blob/d51075968c5fd40b37a55d20c8e945c1f181d529/rfcn/operator_cxx/deformable_convolution.cu
 */

#include "./deformable_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(DeformableConvolutionParam param, int dtype,
    mxnet::ShapeVector *in_shape,
    mxnet::ShapeVector *out_shape,
    Context ctx) {
    Operator *op = nullptr;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeformableConvolutionOp<gpu, DType>(param);
    })
      return op;
  }

}  // namespace op
}  // namespace mxnet


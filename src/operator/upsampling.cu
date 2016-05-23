/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling_nearest.cc
 * \brief
 * \author Bing Xu
*/

#include "./deconvolution-inl.h"
#include "./upsampling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(UpSamplingParam param) {
  if (param.sample_type == up_enum::kNearest) {
    return new UpSamplingNearestOp<gpu>(param);
  } else if (param.sample_type == up_enum::kBilinear) {
    DeconvolutionParam p = DeconvolutionParam();
    int kernel = 2 * param.scale - param.scale % 2;
    int stride = param.scale;
    int pad = static_cast<int>(ceil((param.scale - 1) / 2.));
    p.workspace = param.workspace;
    p.num_group = param.num_filter;
    p.num_filter = param.num_filter;
    p.no_bias =  true;
    int shape[] = {1, 1};
    shape[0] = shape[1] = kernel;
    p.kernel = TShape(shape, shape + 2);
    shape[0] = shape[1] = stride;
    p.stride = TShape(shape, shape + 2);
    shape[0] = shape[1] = pad;
    p.pad = TShape(shape, shape + 2);
    return new DeconvolutionOp<gpu>(p);
  } else {
    LOG(FATAL) << "Unknown sample type";
    return NULL;
  }
}

}  // namespace op
}  // namespace mxnet

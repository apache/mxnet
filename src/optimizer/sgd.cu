/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include "./sgd-inl.h"

namespace mxnet {
namespace opt {

struct sgd_clip {
  MSHADOW_XINLINE static real_t Map(real_t x, real_t bound) {
    if (x > bound) {
      return bound;
    } else if (x < -bound) {
      return -bound;
    } else {
      return x;
    }
  }
};

template<typename xpu>
void sgd_mom_update(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                float lr, const SGDParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> mom2d = mom.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  if (param.clip_gradient >= 0.0f) {
    mom2d = param.momentum*mom2d - lr*(param.rescale_grad*F<sgd_clip>(grad2d, param.clip_gradient) + param.wd*weight2d);
  } else {
    mom2d = param.momentum*mom2d - lr*(param.rescale_grad*grad2d + param.wd*weight2d);
  }
  weight2d += mom2d;
}

template<typename xpu>
void sgd_update(RunContext ctx, TBlob weight, const TBlob grad,
                float lr, const SGDParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  if (param.clip_gradient >= 0.0f) {
    weight2d -= lr*(param.rescale_grad*F<sgd_clip>(grad2d, param.clip_gradient) + param.wd*weight2d);
  } else {
    weight2d -= lr*(param.rescale_grad*grad2d + param.wd*weight2d);
  }
}

template void sgd_mom_update<gpu>(RunContext, TBlob, const TBlob, TBlob,
                              float, const SGDParam&);
template void sgd_mom_update<cpu>(RunContext, TBlob, const TBlob, TBlob,
                              float, const SGDParam&);
template void sgd_update<gpu>(RunContext, TBlob, const TBlob,
                              float, const SGDParam&);
template void sgd_update<cpu>(RunContext, TBlob, const TBlob,
                              float, const SGDParam&);

}  // namespace opt
}  // namespace mxnet

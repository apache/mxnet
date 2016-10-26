/*!
 *  Copyright (c) 2015 by Contributors
 * \file rmsprop-inl.h
 * \brief Implements RMSProp adaptive learning rate optimizer using moving
 * average squared gradients. See http://cs231n.github.io/neural-networks-3/#ada
 * for details
 * \author Rodrigo Castro
 */
#ifndef MXNET_OPTIMIZER_RMSPROP_INL_H_
#define MXNET_OPTIMIZER_RMSPROP_INL_H_

#include <mshadow/tensor.h>
#include "../operator/mshadow_op.h"
#include <mxnet/optimizer.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <map>
#include <utility>

namespace mxnet {
namespace opt {

struct RMSPropParam : public dmlc::Parameter<RMSPropParam> {
  float decay_rate;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(RMSPropParam) {
    DMLC_DECLARE_FIELD(decay_rate)
    .set_range(0.0f, 1.0f)
    .set_default(0.95f)
    .describe("gamma1. typical values are [0.9, 0.99, 0.999]");
    //DMLC_DECLARE_FIELD(gamma1)
    //set_range(0.0f, 1.0f)
    //.set_default(0.9f)
    //.describe("gamma1. decay factor of moving average for gradient, gradient^2."
    //          "Default value is set to 0.95")
    //DMLC_DECLARE_FIELD(gamma1)
    //set_range(0.0f, 1.0f)
    //.set_default(0.9f)
    //.describe("gamma2. 'momentum' factor"
    //          "Default value if set to 0.9")
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("epsilon. Smoothing term (1e-8) that avoids division by zero.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(1.0f)
    .describe("weight decay.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient in range"
              "[-clip_gradient, clip_gradient]");
  }
};

template<typename xpu>
void rmsprop_update(RunContext ctx, TBlob weight, const TBlob grad, TBlob cache,
                float lr, const RMSPropParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> cache2d = cache.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2dSource = grad.FlatTo2D<xpu, real_t>(s);

  Tensor<xpu, 2> grad2d = grad2dSource;
  if (param.clip_gradient >= 0.0f) {
    // grad2d = F<op::mshadow_op::clip>(param.rescale_grad * grad2dSource, param.clip_gradient);
    cache2d = param.decay_rate * cache2d + (1.0f - param.decay_rate)
      * F<op::mshadow_op::clip>(param.rescale_grad * grad2dSource, param.clip_gradient)
      * F<op::mshadow_op::clip>(param.rescale_grad * grad2dSource, param.clip_gradient);
    weight2d -= lr
      * F<op::mshadow_op::clip>(param.rescale_grad * grad2dSource, param.clip_gradient) 
      * F<op::mshadow_op::reciprocal_square_root>(cache2d + param.epsilon);
  } else {
    //grad2d = param.rescale_grad * grad2dSource;
    cache2d = param.decay_rate * cache2d + (1.0f - param.decay_rate) * grad2d * grad2d;
    weight2d -= lr * grad2d * F<op::mshadow_op::reciprocal_square_root>(cache2d + param.epsilon);
  }
}

void call_rmsprop_update_cpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob cache,
                float lr, const RMSPropParam& param);
#if MXNET_USE_CUDA
void call_rmsprop_update_gpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob cache,
                float lr, const RMSPropParam& param);
#endif  // MXNET_USE_CUDA

#if DMLC_USE_CXX11

class RMSPropOpt : public Optimizer {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  void CreateState(const int index, const NDArray *weight) override {
    if (cache.find(index) == cache.end()) {
      cache[index] = NDArray(weight->shape(), weight->ctx());
      cache[index] = 0.0f;
    }
  }

  void Update(const int index, NDArray *weight,
              const NDArray *grad, const float lr) override {
    NDArray w = *weight, g = *grad;
    CreateState(index, weight);
    switch (w.ctx().dev_type) {
     case Context::kCPU:
     case Context::kCPUPinned:
      Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
        call_rmsprop_update_cpu(ctx, w.data(), g.data(), cache[index].data(), lr, param_);
      }, w.ctx(), {g.var()}, {w.var(), cache[index].var()}, FnProperty::kNormal);
      break;
     case Context::kGPU:
#if MXNET_USE_CUDA
      Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
        call_rmsprop_update_gpu(ctx, w.data(), g.data(), cache[index].data(), lr, param_);
      }, w.ctx(), {g.var()}, {w.var(), cache[index].var()}, FnProperty::kNormal);
      break;
#else
        LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif  // MXNET_USE_CUDA
     default:
      LOG(FATAL) << "Unsupported device type for sgd optimizer: " << w.ctx().dev_type;
    }
  }

 private:
  RMSPropParam param_;
  std::map<int, NDArray> cache;
};

#endif  // DMLC_USE_CXX11

}  // namespace opt
}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_RMSPROP_INL_H_

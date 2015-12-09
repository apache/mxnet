/*!
 *  Copyright (c) 2015 by Contributors
 * \file sgd-inl.h
 * \brief Operator interface of mxnet.
 * \author Junyuan Xie
 */
#ifndef MXNET_OPTIMIZER_SGD_INL_H_
#define MXNET_OPTIMIZER_SGD_INL_H_

#include <mshadow/tensor.h>
#include <mxnet/optimizer.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <map>
#include <utility>

namespace mxnet {
namespace opt {

struct SGDParam : public dmlc::Parameter<SGDParam> {
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SGDParam) {
    DMLC_DECLARE_FIELD(momentum)
    .set_range(0.0f, 1.0f)
    .set_default(0.0f)
    .describe("momentum");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0001f)
    .describe("weight decay");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
  }
};

template<typename xpu>
void sgd_mom_update(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                float lr, const SGDParam& param);
template<typename xpu>
void sgd_update(RunContext ctx, TBlob weight, const TBlob grad,
                float lr, const SGDParam& param);

#if DMLC_USE_CXX11
#include <mxnet/ndarray.h>

class SGDOpt : public Optimizer {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  void CreateState(const int index, const NDArray *weight) override {
    if (param_.momentum > 0.0f && mom.find(index) == mom.end()) {
      mom[index] = NDArray(weight->shape(), weight->ctx());
      mom[index] = 0.0f;
    }
  }

  void Update(const int index, NDArray *weight,
              const NDArray *grad, const float lr) override {
    NDArray w = *weight, g = *grad;
    if (param_.momentum > 0.0f) {
      CreateState(index, weight);
      if (w.ctx().dev_type == Context::kCPU ||
          w.ctx().dev_type == Context::kCPUPinned) {
        Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
          sgd_mom_update<cpu>(ctx, w.data(), g.data(), mom[index].data(), lr, param_);
        }, w.ctx(), {g.var()}, {w.var(), mom[index].var()}, FnProperty::kNormal);
      } else if (w.ctx().dev_type == Context::kGPU) {
        Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
          sgd_mom_update<gpu>(ctx, w.data(), g.data(), mom[index].data(), lr, param_);
        }, w.ctx(), {g.var()}, {w.var(), mom[index].var()}, FnProperty::kNormal);
      } else {
        LOG(FATAL) << "Unsupported device type for sgd optimizer: " << w.ctx().dev_type;
      }
    } else {
      if (w.ctx().dev_type == Context::kCPU ||
          w.ctx().dev_type == Context::kCPUPinned) {
        Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
          sgd_update<cpu>(ctx, w.data(), g.data(), lr, param_);
        }, w.ctx(), {g.var()}, {w.var()}, FnProperty::kNormal);
      } else if (w.ctx().dev_type == Context::kGPU) {
        Engine::Get()->PushSync([this, index, w, g, lr](RunContext ctx) {
          sgd_update<gpu>(ctx, w.data(), g.data(), lr, param_);
        }, w.ctx(), {g.var()}, {w.var()}, FnProperty::kNormal);
      } else {
        LOG(FATAL) << "Unsupported device type for sgd optimizer: " << w.ctx().dev_type;
      }
    }
  }

 private:
  SGDParam param_;
  std::map<int, NDArray> mom;
};

#endif  // DMLC_USE_CXX11

}  // namespace opt
}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_SGD_INL_H_

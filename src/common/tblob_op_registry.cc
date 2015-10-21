/*!
 *  Copyright (c) 2015 by Contributors
 * \file tblob_op_registry.cc
 * Implementation of tblob op registry
 */
#include <mxnet/ndarray.h>
#include <mxnet/engine.h>
#include <vector>
#include <mutex>
#include "./tblob_op_registry.h"

namespace mxnet {
namespace common {


class TBlobOpRegEntryImpl : public TBlobOpRegEntry {
 public:
  TSelf& set_function(int dev_mask, UnaryFunction funary) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reg_counter_;
    if (funary_.size() <= static_cast<size_t>(dev_mask)) {
      funary_.resize(dev_mask + 1, nullptr);
    }
    if (funary_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_[dev_mask] = funary;
    // return if it is already registered.
    if (reg_counter_ != 1) return *this;

    // The body to be registered
    auto body = [this] (NDArray **used_vars,
                        real_t *s,
                        NDArray **mutate_vars) {
      NDArray src = *used_vars[0];
      NDArray *out = mutate_vars[0];

      if (out->is_none()) {
        *out = NDArray(src.shape(), src.ctx(), true);
      } else {
        CHECK(out->ctx() == src.ctx()) << "target context mismatch";
        CHECK(out->shape() == src.shape()) << "target shape mismatch";
      }
      // important: callback must always capture by value
      NDArray ret = *out;
      // get the const variables
      std::vector<Engine::VarHandle> const_vars;
      if (src.var() != ret.var()) const_vars.push_back(src.var());
      // check if the function exist
      int dev_mask = src.ctx().dev_mask();
      if (static_cast<size_t>(dev_mask) >= funary_.size() ||
          funary_[dev_mask] == nullptr) {
        if (dev_mask == gpu::kDevMask) LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        LOG(FATAL) << "Function " << this->name << "not registered for device " << dev_mask;
      }
      // invoke the function
      UnaryFunction fun = funary_[dev_mask];
      Engine::Get()->PushSync([src, ret, fun, dev_mask](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          (*fun)(src.data(), &tmp, ctx);
#if MXNET_USE_CUDA
          if (dev_mask == gpu::kDevMask) {
            ctx.get_stream<gpu>()->Wait();
          }
#endif
        }, src.ctx(), const_vars, {ret.var()});
    };
    // register the function.
    NDArrayReg()
        .set_body(body)
        .set_num_use_vars(1)
        .set_num_mutate_vars(1)
        .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
        .add_argument("src", "NDArray", "Source input to the function");
    return *this;
  }

  TSelf& describe(const std::string &description) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reg_counter_ != 1) return *this;
    NDArrayReg().describe(description);
    return *this;
  }

  GenericTBlobOp *GetOp() const override {
    return nullptr;
  }

 private:
  // internal mutex
  std::mutex mutex_;
  // unary functions on each device mask
  std::vector<UnaryFunction> funary_;
  // registration counter
  int reg_counter_{0};
  // NDArray registry
  NDArrayFunctionReg *ndarray_reg_{nullptr};
  // internal function to register NDArray function.
  inline NDArrayFunctionReg &NDArrayReg() {
    if (ndarray_reg_ == nullptr) {
      NDArrayFunctionReg &reg =
          ::dmlc::Registry<NDArrayFunctionReg>::Get()->__REGISTER__(this->name);
      ndarray_reg_ = &reg;
    }
    return *ndarray_reg_;
  }
};


TBlobOpRegEntry& TBlobOpRegistry::__REGISTER_OR_FIND__(const std::string &name) {
  if (fmap_.count(name) != 0) return *fmap_.at(name);
  TBlobOpRegEntry *e = new TBlobOpRegEntryImpl();
  e->name = name;
  fmap_[name] = e;
  return *e;
}

TBlobOpRegistry* TBlobOpRegistry::Get() {
  static TBlobOpRegistry inst;
  return &inst;
}

TBlobOpRegistry::~TBlobOpRegistry() {
  for (auto kv : fmap_) {
    delete kv.second;
  }
}

}  // namespace common
}  // namespace mxnet

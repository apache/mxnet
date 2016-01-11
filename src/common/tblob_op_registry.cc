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
class TBlobUnaryOpProp;

class TBlobOpRegEntryImpl : public TBlobOpRegEntry {
 public:
  // functions
  TSelf& set_function(int dev_mask,
                      UnaryFunction funary,
                      bool inplace_in_out,
                      bool register_symbolic) override {
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
    inplace_in0_out_forward_ = inplace_in_out;
    if (reg_counter_ == 1) {
      this->RegisterUnary();
      register_symbolic_ = register_symbolic;
      if (register_symbolic) {
        this->RegisterUnarySymbolic();
      }
    }
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradType1 fgrad,
                      bool inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (funary_grad_t1_.size() <= static_cast<size_t>(dev_mask)) {
      funary_grad_t1_.resize(dev_mask + 1, nullptr);
    }
    if (funary_grad_t1_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device gradient function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_grad_t1_[dev_mask] = fgrad;
    inplace_out_in0_grad_ = inplace_out_in_grad;
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradType2 fgrad,
                      bool inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (funary_grad_t2_.size() <= static_cast<size_t>(dev_mask)) {
      funary_grad_t2_.resize(dev_mask + 1, nullptr);
    }
    if (funary_grad_t2_[dev_mask] != nullptr) {
      LOG(FATAL) << "Device gradient function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    funary_grad_t2_[dev_mask] = fgrad;
    inplace_out_in0_grad_ = inplace_out_in_grad;
    return *this;
  }

  TSelf& set_shape_infer(UnaryShapeInfer fshapeinfer) override {
    std::lock_guard<std::mutex> lock(mutex_);
    unary_infer_ = fshapeinfer;
    return *this;
  }

  TSelf& describe(const std::string &description) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reg_counter_ != 1) return *this;
    NDArrayReg().describe(description);
    if (register_symbolic_) {
      OpReg().describe(description);
    }
    return *this;
  }

 private:
  // make friend with unary op
  friend class TBlobUnaryOpProp;
  // internal mutex
  std::mutex mutex_;
  // registration counter
  int reg_counter_{0};
  bool register_symbolic_{true};
  // unary shape inferencer
  UnaryShapeInfer unary_infer_{nullptr};
  // unary functions on each device mask
  std::vector<UnaryFunction> funary_;
  // type 1 gradient function
  std::vector<UnaryGradType1> funary_grad_t1_;
  // type 2 gradient function
  std::vector<UnaryGradType2> funary_grad_t2_;
  // whether do inplace optimization of in 0 and output
  bool inplace_in0_out_forward_{true};
  // whether do inplace optimization of out_grad and in_grad0
  bool inplace_out_in0_grad_{false};
  // NDArray registry
  NDArrayFunctionReg *ndarray_reg_{nullptr};
  OperatorPropertyReg *op_reg_{nullptr};
  // internal function to register NDArray function.
  inline NDArrayFunctionReg &NDArrayReg() {
    if (ndarray_reg_ == nullptr) {
      NDArrayFunctionReg &reg =
          ::dmlc::Registry<NDArrayFunctionReg>::Get()->__REGISTER__(this->name);
      ndarray_reg_ = &reg;
    }
    return *ndarray_reg_;
  }
  // internal function to register NDArray function.
  inline OperatorPropertyReg &OpReg() {
    if (op_reg_ == nullptr) {
      OperatorPropertyReg &reg =
          ::dmlc::Registry<OperatorPropertyReg>::Get()->__REGISTER__(this->name);
      op_reg_ = &reg;
    }
    return *op_reg_;
  }
  // start registering all stuffs
  void RegisterUnary();
  void RegisterUnarySymbolic();
};

// Unary operator to invoke generic TBlob function.
struct TBlobUnaryOperator : public Operator {
  TBlobOpRegEntry::UnaryFunction forward;
  TBlobOpRegEntry::UnaryGradType1 backward1{nullptr};
  TBlobOpRegEntry::UnaryGradType2 backward2{nullptr};

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    TBlob out = out_data[0];
    (*forward)(in_data[0], &out, req[0], ctx.run_ctx);
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad,
                const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    arg::OutGrad ograd; ograd.data = out_grad[0];
    TBlob igrad = in_grad[0];
    if (backward1 != nullptr) {
      arg::OutValue out_value; out_value.data = out_data[0];
      (*backward1)(ograd, out_value, &igrad, req[0], ctx.run_ctx);
    } else if (backward2 != nullptr) {
      arg::Input0 in0; in0.data = in_data[0];
      (*backward2)(ograd, in0, &igrad, req[0], ctx.run_ctx);
    } else {
      LOG(FATAL) << "Backward is not supported";
    }
  }
};  // class UnaryOperator

class TBlobUnaryOpProp : public OperatorProperty {
 public:
  std::string name;
  TBlobOpRegEntryImpl* source;

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    if (source->unary_infer_ == nullptr) {
      out_shape->push_back(dshape);
    } else {
      out_shape->push_back((*(source->unary_infer_))(dshape));
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TBlobUnaryOpProp();
    ptr->source = source;
    ptr->name = name;
    return ptr;
  }

  std::string TypeString() const override {
    return name;
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (source->funary_grad_t1_.size() != 0) {
      return {out_grad[0], out_data[0]};
    } else if (source->funary_grad_t2_.size() != 0) {
      return {out_grad[0], in_data[0]};
    } else {
      LOG(FATAL) << "Backward of " << name << " is not decalred";
      return {};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    if (source->inplace_out_in0_grad_) {
      return {{out_grad[0], in_grad[0]}};
    } else {
      return {};
    }
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (source->inplace_in0_out_forward_) {
      return {{in_data[0], out_data[0]}};
    } else {
      return {};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    size_t dev_mask = ctx.dev_mask();
    TBlobUnaryOperator *op = new TBlobUnaryOperator();
    CHECK(dev_mask < source->funary_.size() && source->funary_[dev_mask] != nullptr);
    op->forward = source->funary_[dev_mask];
    if (dev_mask < source->funary_grad_t1_.size()) {
      op->backward1 = source->funary_grad_t1_[dev_mask];
    }
    if (dev_mask < source->funary_grad_t2_.size()) {
      op->backward2 = source->funary_grad_t2_[dev_mask];
    }
    return op;
  }
};

void TBlobOpRegEntryImpl::RegisterUnary() {
  CHECK_EQ(reg_counter_, 1);
  // The body to be registered
  auto body = [this] (NDArray **used_vars,
                      real_t *s,
                      NDArray **mutate_vars,
                      int num_params,
                      char **param_keys,
                      char **param_vals) {
    NDArray src = *used_vars[0];
    NDArray *out = mutate_vars[0];
    TShape dshape = src.shape();
    if (unary_infer_ != nullptr) dshape = unary_infer_(dshape);

    if (out->is_none()) {
      *out = NDArray(dshape, src.ctx(), true, src.dtype());
    } else {
      CHECK(out->ctx() == src.ctx()) << "target context mismatch";
      CHECK(out->dtype() == src.dtype()) << "target data type mismatch";
      CHECK(out->shape() == dshape) << "target shape mismatch "
      << out->shape() << " vs. " << dshape;
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
        (*fun)(src.data(), &tmp, kWriteTo, ctx);
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
}

void TBlobOpRegEntryImpl::RegisterUnarySymbolic() {
  // register the operator
  auto op_factory = [this]() {
    TBlobUnaryOpProp *prop = new TBlobUnaryOpProp();
    prop->name = this->name;
    prop->source = this;
    return prop;
  };
  OpReg()
      .set_body(op_factory)
      .add_argument("src", "Symbol", "Source symbolic input to the function");
}
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

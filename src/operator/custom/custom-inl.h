/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#define MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <queue>
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct CustomOpParam {
  std::string op_type;
  std::vector<std::pair<std::string, std::string> > kwargs;
};

template<typename xpu>
class CustomOp : public Operator {
 public:
  explicit CustomOp(MXCallbackList* op_info) {
    op_info_.reset(op_info, [](MXCallbackList *ptr){
        reinterpret_cast<CustomOpDelFunc>(ptr->callbacks[kCustomOpDelete])(
          ptr->contexts[kCustomOpDelete]);
        delete ptr;
      });
    if (std::string("NaiveEngine") == dmlc::GetEnv("MXNET_ENGINE_TYPE", std::string())) {
      sync_mode_ = true;
    } else {
      sync_mode_ = false;
      destructing_ = false;
      worker_ = std::thread([&]() {
          std::unique_lock<std::mutex> lock(mtx_);
          while (!q_.empty() || !destructing_) {
            cv_.wait(lock, [&] {return !q_.empty() || destructing_;});
            while (!q_.empty()) {
              q_.front()();
              q_.pop();
            }
          }
        });
    }
  }

  ~CustomOp() {
    if (!sync_mode_) {
      {
        std::unique_lock<std::mutex> lock(mtx_);
        destructing_ = true;
        cv_.notify_all();
      }
      worker_.join();
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args);

  virtual ExecType exec_type() const {
    return kAsync;
  }

 private:
  Context get_ctx();
  std::shared_ptr<MXCallbackList> op_info_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::thread worker_;
  std::queue<std::function<void(void)> > q_;
  bool destructing_;
  bool sync_mode_;
};  // CustomOp

template<typename xpu>
Operator* CreateOp(MXCallbackList *op_info);

class CustomOpProp : public OperatorProperty {
 public:
  static void Register(const std::string &op_type, CustomOpPropCreator creator) {
    if (registry_.find(op_type) != registry_.end()) {
      LOG(WARNING) << "New registration is overriding existing custom operator " << op_type;
    }
    registry_[op_type] = creator;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    kwargs_ = kwargs;
    param_.op_type = "";
    param_.kwargs.clear();
    std::vector<const char*> keys, vals;
    for (auto &p : kwargs) {
      if (p.first == "op_type") {
        param_.op_type = p.second;
      } else {
        param_.kwargs.push_back(p);
        keys.push_back(p.first.c_str());
        vals.push_back(p.second.c_str());
      }
    }
    CHECK_NE(param_.op_type, "") << "Custom operator type missing";
    CHECK(registry_.find(param_.op_type) != registry_.end())
      << "Cannot find custom operator type " << param_.op_type;
    CustomOpPropCreator creator = registry_[param_.op_type];
    info_.reset(new MXCallbackList, [](MXCallbackList* ptr){
        reinterpret_cast<CustomOpDelFunc>(ptr->callbacks[kCustomOpPropDelete])(
          ptr->contexts[kCustomOpPropDelete]);
        delete ptr;
      });
    CHECK(creator(param_.op_type.c_str(), keys.size(), keys.data(), vals.data(), info_.get()));
    num_inputs_ = ListArguments().size();
    num_outputs_ = ListOutputs().size();
    num_auxs_ = ListAuxiliaryStates().size();
  }

  std::vector<std::string> ListArguments() const override {
    char ** args = NULL;
    CHECK(reinterpret_cast<CustomOpListFunc>(info_->callbacks[kCustomOpPropListArguments])(
      &args, info_->contexts[kCustomOpPropListArguments]));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    char ** args = NULL;
    CHECK(reinterpret_cast<CustomOpListFunc>(info_->callbacks[kCustomOpPropListOutputs])(
      &args, info_->contexts[kCustomOpPropListOutputs]));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    char ** args = NULL;
    CHECK(reinterpret_cast<CustomOpListFunc>(info_->callbacks[kCustomOpPropListAuxiliaryStates])(
      &args, info_->contexts[kCustomOpPropListAuxiliaryStates]));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  int NumOutputs() const override {
    return ListOutputs().size();
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>(kwargs_.begin(), kwargs_.end());
  }


  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    std::vector<uint32_t*> shapes;
    std::vector<int> ndims;
    size_t size = 0;
    for (const auto& s : *in_shape) size += s.ndim();
    std::vector<uint32_t> shapes_buffer(size);
    shapes_buffer.resize(size);
    uint32_t *ptr = shapes_buffer.data();
    for (auto iter = in_shape->begin(); iter != in_shape->end(); ++iter) {
      shapes.push_back(ptr);
      ndims.push_back(iter->ndim());
      ptr = nnvm::ShapeTypeCast(iter->begin(), iter->end(), ptr);
    }
    shapes.resize(num_inputs_+num_outputs_+num_auxs_);
    ndims.resize(num_inputs_+num_outputs_+num_auxs_);

    CHECK(reinterpret_cast<CustomOpInferShapeFunc>(info_->callbacks[kCustomOpPropInferShape])(
      shapes.size(), ndims.data(), shapes.data(), info_->contexts[kCustomOpPropInferShape]));
    for (unsigned i = 0; i < in_shape->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape(shapes[i], shapes[i]+ndims[i]));
    }
    out_shape->clear();
    for (unsigned i = num_inputs_; i < num_inputs_+num_outputs_; ++i) {
      out_shape->push_back(TShape(shapes[i], shapes[i]+ndims[i]));
    }
    aux_shape->clear();
    for (unsigned i = num_inputs_+num_outputs_; i < shapes.size(); ++i) {
      aux_shape->push_back(TShape(shapes[i], shapes[i]+ndims[i]));
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    if (info_->num_callbacks <= kCustomOpPropInferType) {
      return OperatorProperty::InferType(in_type, out_type, aux_type);
    }

    std::vector<int> types;
    for (const auto &i : *in_type) types.push_back(i);
    for (const auto &i : *out_type) types.push_back(i);
    for (const auto &i : *aux_type) types.push_back(i);

    CHECK(reinterpret_cast<CustomOpInferTypeFunc>(info_->callbacks[kCustomOpPropInferType])(
      types.size(), types.data(), info_->contexts[kCustomOpPropInferType]));
    for (unsigned i = 0; i < num_inputs_; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, types[i]);
    }
    for (unsigned i = 0; i < num_outputs_; ++i) {
      TYPE_ASSIGN_CHECK(*out_type, i, types[i+num_inputs_]);
    }
    for (unsigned i = 0; i < num_auxs_; ++i) {
      TYPE_ASSIGN_CHECK(*aux_type, i, types[i+num_inputs_+num_outputs_]);
    }
    return true;
  }


  OperatorProperty* Copy() const override {
    CustomOpProp *prop_sym = new CustomOpProp();
    prop_sym->Init(kwargs_);
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Custom";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    int num_dep;
    int *rdeps;
    CHECK(reinterpret_cast<CustomOpBwdDepFunc>(
      info_->callbacks[kCustomOpPropDeclareBackwardDependency])(
        out_grad.data(), in_data.data(), out_data.data(), &num_dep,
        &rdeps, info_->contexts[kCustomOpPropDeclareBackwardDependency]));
    std::vector<int> deps;
    deps.insert(deps.end(), rdeps, rdeps+num_dep);
    return deps;
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  static std::map<std::string, CustomOpPropCreator> registry_;

  CustomOpParam param_;
  std::shared_ptr<MXCallbackList> info_;
  std::vector<std::pair<std::string, std::string> > kwargs_;
  unsigned num_inputs_, num_outputs_, num_auxs_;
  mutable std::vector<uint32_t> shapes_buffer_;
};  // class CustomOpProp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_

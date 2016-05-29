/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUSTOM_INL_H_
#define MXNET_OPERATOR_CUSTOM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include "./operator_common.h"

namespace mxnet {
namespace op {

struct CustomOpParam {
  std::string op_type;
  std::vector<std::pair<std::string, std::string> > kwargs;
};

template<typename xpu>
class CustomOp : public Operator {
 public:
  explicit CustomOp(CustomOpInfo* op_info) {
    op_info_.reset(op_info, [](CustomOpInfo *ptr){ ptr->del(ptr->p_del); });
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
  std::shared_ptr<CustomOpInfo> op_info_;
};  // CustomOp

template<typename xpu>
Operator* CreateOp(CustomOpInfo *op_info);

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
    CHECK_NE(registry_.find(param_.op_type), registry_.end())
      << "Cannot find custom operator type " << param_.op_type;
    CustomOpPropCreator creator = registry_[param_.op_type];
    info_.reset(new CustomOpPropInfo, [](CustomOpPropInfo* ptr){ptr->del(ptr->p_del);});
    CHECK(creator(param_.op_type.c_str(), keys.size(), keys.data(), vals.data(), info_.get()));
    num_inputs_ = ListArguments().size();
    num_outputs_ = ListOutputs().size();
    num_auxs_ = ListAuxiliaryStates().size();
  }

  std::vector<std::string> ListArguments() const override {
    char ** args = NULL;
    CHECK(info_->list_arguments(&args, info_->p_list_arguments));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    char ** args = NULL;
    CHECK(info_->list_outputs(&args, info_->p_list_outputs));
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    char ** args = NULL;
    CHECK(info_->list_auxiliary_states(&args, info_->p_list_auxiliary_states));
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
    std::vector<unsigned*> shapes;
    std::vector<int> ndims;
    for (auto iter = in_shape->begin(); iter != in_shape->end(); ++iter) {
      shapes.push_back(iter->data());
      ndims.push_back(iter->ndim());
    }
    shapes.resize(num_inputs_+num_outputs_+num_auxs_);
    ndims.resize(num_inputs_+num_outputs_+num_auxs_);
    CHECK(info_->infer_shape(shapes.size(), ndims.data(), shapes.data(), info_->p_infer_shape));
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
    CHECK(info_->declare_backward_dependency(out_grad.data(), in_data.data(),
                                             out_data.data(), &num_dep, &rdeps,
                                             info_->p_declare_backward_dependency));
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
  std::shared_ptr<CustomOpPropInfo> info_;
  std::vector<std::pair<std::string, std::string> > kwargs_;
  unsigned num_inputs_, num_outputs_, num_auxs_;
};  // class CustomOpProp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_INL_H_

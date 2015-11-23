/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_NDARRAY_OP_INL_H_
#define MXNET_OPERATOR_NDARRAY_OP_INL_H_
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

struct NDArrayOpParam : public dmlc::Parameter<NDArrayOpParam> {
  void *info;
  bool need_top_grad;

  NDArrayOpInfo *pinfo;
  int num_inputs_, num_outputs_;
  DMLC_DECLARE_PARAMETER(NDArrayOpParam) {
    DMLC_DECLARE_FIELD(info);
    DMLC_DECLARE_FIELD(need_top_grad).set_default(true)
    .describe("Whether this layer needs out grad for backward. "
      "Should be false for loss layers.");
  }
};

template<typename xpu>
class NDArrayOp : public Operator {
 public:
  explicit NDArrayOp(NDArrayOpParam p) {
    this->param_ = p;
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

 private:
  NDArrayOpParam param_;
  Context get_ctx();
};  // NDArrayOp

template<typename xpu>
Operator* CreateOp(NDArrayOpParam param);

#if DMLC_USE_CXX11
class NDArrayOpProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    char ** args = NULL;
    param_.pinfo->list_arguments(&args, param_.pinfo->p_list_arguments);
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  std::vector<std::string> ListOutputs() const override {
    char ** args = NULL;
    param_.pinfo->list_outputs(&args, param_.pinfo->p_list_outputs);
    std::vector<std::string> ret;
    for (int i = 0; args[i] != NULL; ++i) {
      ret.push_back(args[i]);
    }
    return ret;
  }

  int NumOutputs() const override {
    return param_.num_outputs_;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    for (auto iter = kwargs.begin(); iter != kwargs.end(); ++iter) {
      if (iter->first == "info") {
        sscanf(iter->second.c_str(), "%p", &param_.pinfo);
      }
    }
    param_.num_inputs_ = ListArguments().size();
    param_.num_outputs_ = ListOutputs().size();
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
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
    shapes.resize(param_.num_inputs_+param_.num_outputs_);
    ndims.resize(param_.num_inputs_+param_.num_outputs_);
    param_.pinfo->infer_shape(shapes.size(), ndims.data(), shapes.data(),
          param_.pinfo->p_infer_shape);
    for (unsigned i = 0; i < in_shape->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape(shapes[i], shapes[i]+ndims[i]));
    }
    out_shape->clear();
    for (unsigned i = param_.num_inputs_; i < shapes.size(); ++i) {
      out_shape->push_back(TShape(shapes[i], shapes[i]+ndims[i]));
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    NDArrayOpProp *prop_sym = new NDArrayOpProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "_NDArray";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> deps;
    if (param_.need_top_grad) {
      deps.insert(deps.end(), out_grad.begin(), out_grad.end());
    }
    deps.insert(deps.end(), in_data.begin(), in_data.end());
    deps.insert(deps.end(), out_data.begin(), out_data.end());
    return deps;
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  NDArrayOpParam param_;
};  // class PythonProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NDARRAY_OP_INL_H_

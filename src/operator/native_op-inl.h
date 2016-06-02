/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie, Bing Xu
*/

#ifndef MXNET_OPERATOR_NATIVE_OP_INL_H_
#define MXNET_OPERATOR_NATIVE_OP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace nativeop {
enum NativeOpSyncDirection {kTensorToData, kDataToTensor};
enum NativeOpResource {kTempSpace};
}


struct NativeOpParam : public dmlc::Parameter<NativeOpParam> {
  void *info;
  bool need_top_grad;

  NativeOpInfo *pinfo;
  int num_inputs_, num_outputs_;
  DMLC_DECLARE_PARAMETER(NativeOpParam) {
    DMLC_DECLARE_FIELD(info);
    DMLC_DECLARE_FIELD(need_top_grad).set_default(true)
    .describe("Whether this layer needs out grad for backward. "
      "Should be false for loss layers.");
  }
};

template<typename xpu>
class NativeOpBase : public Operator {
 public:
  virtual ExecType exec_type() const {
    return kAsync;
  }

 protected:
  inline uint64_t _CalculateSpace(const std::vector<TBlob> &tblob_vec) {
    uint64_t size = 0;
    for (size_t i = 0; i < tblob_vec.size(); ++i) {
      size += tblob_vec[i].shape_.Size();
    }
    return size;
  }

  inline void _InitDataVector(const std::vector<TBlob> &tblob_vec,
                              std::vector<real_t*> *vec,
                              real_t *buf,
                              uint64_t *buf_size);


  inline void _InitForward(const OpContext &ctx,
                           const std::vector<TBlob> &in_data,
                           const std::vector<TBlob> &out_data,
                           const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    in_data_ptr_.resize(in_data.size());
    out_data_ptr_.resize(out_data.size());
    aux_args_ptr_.resize(aux_args.size());
    uint64_t buf_size = 0;
    buf_size += _CalculateSpace(in_data);
    buf_size += _CalculateSpace(out_data);
    buf_size += _CalculateSpace(aux_args);
    Tensor<cpu, 1> buf = ctx.requested[nativeop::kTempSpace].get_host_space_typed<1, real_t>(
      Shape1(buf_size));
    buf_size = 0;
    _InitDataVector(in_data, &in_data_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(out_data, &out_data_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(aux_args, &aux_args_ptr_, buf.dptr_, &buf_size);
  }

  inline void _InitBackward(const OpContext &ctx,
                            const std::vector<TBlob> &out_grad,
                            const std::vector<TBlob> &in_data,
                            const std::vector<TBlob> &out_data,
                            const std::vector<TBlob> &in_grad,
                            const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    out_grad_ptr_.resize(out_grad.size());
    in_data_ptr_.resize(in_data.size());
    out_data_ptr_.resize(out_data.size());
    in_grad_ptr_.resize(in_grad.size());
    aux_args_ptr_.resize(aux_args.size());
    uint64_t buf_size = 0;
    buf_size += _CalculateSpace(out_grad);
    buf_size += _CalculateSpace(in_data);
    buf_size += _CalculateSpace(out_data);
    buf_size += _CalculateSpace(in_grad);
    buf_size += _CalculateSpace(aux_args);
    Tensor<cpu, 1> buf = ctx.requested[nativeop::kTempSpace].get_host_space_typed<1, real_t>(
      Shape1(buf_size));
    buf_size = 0;
    _InitDataVector(out_grad, &out_grad_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(in_data, &in_data_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(out_data, &out_data_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(in_grad, &in_grad_ptr_, buf.dptr_, &buf_size);
    _InitDataVector(aux_args, &aux_args_ptr_, buf.dptr_, &buf_size);
  }

  inline void _SyncData(const std::vector<TBlob> &data,
                        const std::vector<real_t*> &vec,
                        mshadow::Stream<xpu> *s,
                        nativeop::NativeOpSyncDirection dir) {
    using namespace mshadow;
    for (size_t i = 0; i < data.size(); ++i) {
      Tensor<xpu, 2> tensor_data = data[i].FlatTo2D<xpu, real_t>(s);
      Tensor<cpu, 2> vector_data = Tensor<cpu, 2>(vec[i], tensor_data.shape_);
      if (tensor_data.dptr_ == vector_data.dptr_) {
        continue;
      }
      switch (dir) {
      case nativeop::kTensorToData:
        Copy(vector_data, tensor_data, s);
        break;
      case nativeop::kDataToTensor:
        Copy(tensor_data, vector_data, s);
        break;
      default:
        LOG(FATAL) << "Not reach";
      }
    }
    // s->Wait();
  }

 protected:
  std::vector<real_t*> in_data_ptr_;
  std::vector<real_t*> out_data_ptr_;
  std::vector<real_t*> aux_args_ptr_;
  std::vector<real_t*> out_grad_ptr_;
  std::vector<real_t*> in_grad_ptr_;
};  // NativeOpBase

template<>
inline void NativeOpBase<gpu>::_InitDataVector(const std::vector<TBlob> &tblob_vec,
                                               std::vector<real_t*> *vec,
                                               real_t *buf,
                                               uint64_t *buf_size) {
  for (size_t i = 0; i < tblob_vec.size(); ++i) {
    vec->at(i) = buf + (*buf_size);
    (*buf_size) += tblob_vec[i].shape_.Size();
  }
}

template<>
inline void NativeOpBase<cpu>::_InitDataVector(const std::vector<TBlob> &tblob_vec,
                                               std::vector<real_t*> *vec,
                                               real_t *buf,
                                               uint64_t *buf_size) {
  for (size_t i = 0; i < tblob_vec.size(); ++i) {
    vec->at(i) = static_cast<real_t*>(tblob_vec[i].dptr_);
  }
}


template<typename xpu>
class NativeOp : public NativeOpBase<xpu> {
 public:
  explicit NativeOp(NativeOpParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Parent::_InitForward(ctx, in_data, out_data, aux_args);
    Parent::_SyncData(in_data, Parent::in_data_ptr_, s, nativeop::kTensorToData);
    Parent::_SyncData(aux_args, Parent::aux_args_ptr_, s,  nativeop::kTensorToData);
    this->_InitNativeForward(in_data, out_data, aux_args);
    if (s!= NULL) s->Wait();
    param_.pinfo->forward(ptrs_.size(), ptrs_.data(),
                          ndims_.data(), shapes_.data(),
                          tags_.data(),
                          param_.pinfo->p_forward);
    Parent::_SyncData(out_data, Parent::out_data_ptr_, s, nativeop::kDataToTensor);
    Parent::_SyncData(aux_args, Parent::aux_args_ptr_, s, nativeop::kDataToTensor);
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Parent::_InitBackward(ctx, out_grad, in_data, out_data, in_grad, aux_args);
    if (param_.need_top_grad) {
      Parent::_SyncData(out_grad, Parent::out_grad_ptr_, s, nativeop::kTensorToData);
    }
    Parent::_SyncData(in_data, Parent::in_data_ptr_, s, nativeop::kTensorToData);
    Parent::_SyncData(out_data, Parent::out_data_ptr_, s, nativeop::kTensorToData);
    this->_InitNativeBackward(out_grad, in_data, out_data, in_grad, aux_args);
    if (s != NULL) s->Wait();
    param_.pinfo->backward(ptrs_.size(), ptrs_.data(),
                           ndims_.data(), shapes_.data(),
                           tags_.data(),
                           param_.pinfo->p_backward);
    Parent::_SyncData(in_grad, Parent::in_grad_ptr_, s, nativeop::kDataToTensor);
    Parent::_SyncData(aux_args, Parent::aux_args_ptr_, s, nativeop::kDataToTensor);
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

 private:
  typedef NativeOpBase<xpu> Parent;
  inline void _InitNativeEntry(const std::vector<TBlob> &tblob_vec,
                               const std::vector<real_t*> &vec,
                               int tag,
                               uint64_t *idx) {
    for (size_t i = 0; i < vec.size(); ++i) {
      ptrs_[*idx] = vec[i];
      ndims_[*idx] = tblob_vec[i].ndim();
      shapes_[*idx] = const_cast<index_t*>(tblob_vec[i].shape_.data());
      tags_[*idx] = tag;
      ++(*idx);
    }
  }
  inline void _InitNativeForward(const std::vector<TBlob> &in_data,
                                 const std::vector<TBlob> &out_data,
                                 const std::vector<TBlob> &aux_args) {
    uint64_t size = in_data.size() + out_data.size();
    ptrs_.resize(size);
    ndims_.resize(size);
    shapes_.resize(size);
    tags_.resize(size);
    uint64_t idx = 0;
    _InitNativeEntry(in_data, Parent::in_data_ptr_, 0, &idx);
    _InitNativeEntry(out_data, Parent::out_data_ptr_, 1, &idx);
    // _InitNativeEntry(aux_args, aux_args_ptr_, 4, &idx);
  }

  inline void _InitNativeBackward(const std::vector<TBlob> &out_grad,
                                  const std::vector<TBlob> &in_data,
                                  const std::vector<TBlob> &out_data,
                                  const std::vector<TBlob> &in_grad,
                                  const std::vector<TBlob> &aux_args) {
    uint64_t size = (param_.need_top_grad ? out_grad.size() : 0) +
                    in_data.size() +
                    out_data.size() +
                    in_grad.size();
                    // aux_args_ptr_.size();
    ptrs_.resize(size);
    ndims_.resize(size);
    shapes_.resize(size);
    tags_.resize(size);
    uint64_t idx = 0;
    _InitNativeEntry(in_data, Parent::in_data_ptr_, 0, &idx);
    _InitNativeEntry(out_data, Parent::out_data_ptr_, 1, &idx);
    _InitNativeEntry(in_grad, Parent::in_grad_ptr_, 2, &idx);
    if (param_.need_top_grad) {
      _InitNativeEntry(out_grad, Parent::out_grad_ptr_, 3, &idx);
    }
    // _InitNativeEntry(aux_args, aux_args_ptr_, 4, &idx);
  }

 private:
  NativeOpParam param_;
  std::vector<real_t*> ptrs_;
  std::vector<int> ndims_;
  std::vector<unsigned*> shapes_;
  std::vector<int> tags_;
};  // NativeOp

template<typename xpu>
Operator* CreateOp(NativeOpParam param);

#if DMLC_USE_CXX11
class NativeOpProp : public OperatorProperty {
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
    NativeOpProp *prop_sym = new NativeOpProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "_Native";
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

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  NativeOpParam param_;
};  // class PythonProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NATIVE_OP_INL_H_

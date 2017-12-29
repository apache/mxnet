/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUSTOM_NATIVE_OP_INL_H_
#define MXNET_OPERATOR_CUSTOM_NATIVE_OP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include "../operator_common.h"

namespace mxnet {
namespace op {

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
class NativeOp : public Operator {
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
    ptrs.clear();
    ndims.clear();
    shapes.clear();
    tags.clear();
    SyncVec(in_data, "in_data", s, 0);
    SyncVec(out_data, "out_data", s, 1);
    s->Wait();
    param_.pinfo->forward(ptrs.size(), ptrs.data(), ndims.data(), shapes.data(),
        tags.data(), param_.pinfo->p_forward);
    for (index_t i = 0; i < out_data.size(); ++i) {
      CHECK_NE(req[i], kAddTo) << "NativeOp doesn't support AddTo for output";
      if (req[i] != kNullOp) {
        std::stringstream ss;
        ss << std::string("out_data") << i;
        Copy(out_data[i].FlatTo2D<xpu, real_t>(s),
             buffer_map[ss.str()].second, s);
      }
    }
    s->Wait();
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
    ptrs.clear();
    ndims.clear();
    shapes.clear();
    tags.clear();
    SyncVec(in_data, "in_data", s, 0);
    SyncVec(out_data, "out_data", s, 1);
    SyncVec(in_grad, "in_grad", s, 2);
    if (param_.need_top_grad) {
      SyncVec(out_grad, "out_grad", s, 3);
    }
    s->Wait();
    param_.pinfo->backward(ptrs.size(), ptrs.data(), ndims.data(), shapes.data(),
        tags.data(), param_.pinfo->p_backward);
    for (index_t i = 0; i < in_grad.size(); ++i) {
      CHECK_NE(req[i], kAddTo) << "NativeOp doesn't support AddTo for output";
      if (req[i] != kNullOp) {
        std::stringstream ss;
        ss << std::string("in_grad") << i;
        Copy(in_grad[i].FlatTo2D<xpu, real_t>(s),
             buffer_map[ss.str()].second, s);
      }
    }
    s->Wait();
  }

 private:
  NativeOpParam param_;
  std::vector<real_t*> ptrs;
  std::vector<int> ndims;
  std::vector<uint32_t*> shapes;
  std::vector<uint32_t> shapes_buffer_;
  std::vector<int> tags;
  std::map<std::string, std::pair<TShape, mshadow::Tensor<cpu, 2> > > buffer_map;

  virtual void SyncBuffer(const TBlob &tblob,
                          const std::string &name,
                          mshadow::Stream<xpu> *stream) {
    using namespace mshadow;
    std::map<std::string, std::pair<TShape, mshadow::Tensor<cpu, 2> > >::iterator buffer =
      buffer_map.find(name);
    if (buffer == buffer_map.end() || buffer->second.first != tblob.shape_) {
      if (buffer != buffer_map.end()) {
        FreeSpace<2, real_t>(&(buffer->second.second));
        buffer_map.erase(buffer);
      }
      buffer_map[name] =
        std::pair<TShape, Tensor<cpu, 2> >(tblob.shape_,
                                         NewTensor<cpu>(tblob.shape_.FlatTo2D(),
                                                        0.0f,
                                                        false));
      buffer = buffer_map.find(name);
    }
    Copy(buffer->second.second, tblob.FlatTo2D<xpu, real_t>(stream), stream);
  }

  virtual void SyncVec(const std::vector<TBlob> &vec,
                       const std::string &prefix,
                       mshadow::Stream<xpu> *stream,
                       int tag) {
    size_t size = 0;
    for (const auto& tblob : vec) size += tblob.shape_.ndim();
    shapes_buffer_.resize(size);
    uint32_t *ptr = shapes_buffer_.data();
    for (size_t i = 0; i < vec.size(); ++i) {
      std::stringstream name;
      name << prefix << i;
      SyncBuffer(vec[i], name.str(), stream);
      ptrs.push_back(buffer_map[name.str()].second.dptr_);
      ndims.push_back(vec[i].ndim());
      shapes.push_back(ptr);
      ptr = nnvm::ShapeTypeCast(vec[i].shape_.begin(), vec[i].shape_.end(), ptr);
      tags.push_back(tag);
    }
  }
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
    std::vector<uint32_t*> shapes;
    std::vector<int> ndims;
    size_t size = 0;
    for (const auto& s : *in_shape) size += s.ndim();
    std::vector<uint32_t> shapes_buffer(size);
    uint32_t *ptr = shapes_buffer.data();
    for (auto iter = in_shape->begin(); iter != in_shape->end(); ++iter) {
      shapes.push_back(ptr);
      ndims.push_back(iter->ndim());
      ptr = nnvm::ShapeTypeCast(iter->begin(), iter->end(), ptr);
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

  Operator* CreateOperator(Context ctx) const override;

 private:
  NativeOpParam param_;
};  // class PythonProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_NATIVE_OP_INL_H_

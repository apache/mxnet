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
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author HPI-DeepLearning
*/

#ifndef MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"
#include "./xnor.h"
#include <type_traits>

#include <csignal>

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace binary_inference_fullc {
enum BinaryInferenceFullyConnectedOpInputs {kData, kWeight, kBias};
enum BinaryInferenceFullyConnectedOpOutputs {kOut};
enum BinaryInferenceFullyConnectedResource {kTempSpace};
}  // fullc

struct BinaryInferenceFullyConnectedParam : public dmlc::Parameter<BinaryInferenceFullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;
  DMLC_DECLARE_PARAMETER(BinaryInferenceFullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");    
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class BinaryInferenceFullyConnectedOp : public Operator {
 public:
  explicit BinaryInferenceFullyConnectedOp(BinaryInferenceFullyConnectedParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[binary_inference_fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[binary_inference_fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    const mxnet::TShape& ishape = in_data[binary_inference_fullc::kData].shape_;
    const mxnet::TShape& oshape = out_data[binary_inference_fullc::kOut].shape_;

    Tensor<xpu, 2, DType> data = in_data[binary_inference_fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);    

    //define binarized weights
    mxnet::op::xnor::BINARY_WORD* wmat_binarized = NULL;    
    wmat_binarized = (mxnet::op::xnor::BINARY_WORD*) in_data[binary_inference_fullc::kWeight].dptr_;

    Tensor<xpu, 2, DType> out = out_data[binary_inference_fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);

    int m = data.size(0);
    int n = data.size(1);
    int k = param_.num_hidden;
    Tensor<xpu, 1, DType> binary_inputs_workspace =
            ctx.requested[binary_inference_fullc::kTempSpace].get_space_typed<xpu, 1, DType>(
                    Shape1(n * m / (sizeof(DType) * CHAR_BIT)), s);

    //====== testing code =======//
    // using ns = std::chrono::nanoseconds;
    // using get_time = std::chrono::steady_clock;
    // auto start = std::chrono::high_resolution_clock::now();
    
    BinaryInferenceFullyConnectedForward(m, n, k, data, binary_inputs_workspace, wmat_binarized, out);

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "Binary FC elapsed time: " << elapsed.count() << " s\n"; 
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    // nothing to do in backward pass
  }

 private:
  BinaryInferenceFullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(BinaryInferenceFullyConnectedParam param, int dtype,
                   mxnet::ShapeVector *in_shape,
                   mxnet::ShapeVector *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class BinaryInferenceFullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const mxnet::TShape &dshape = (*in_shape)[binary_inference_fullc::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());

    // defines shape of binarized weights
    SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_fullc::kWeight, Shape2(param_.num_hidden, num_input / mxnet::op::xnor::BITS_PER_BINARY_WORD));
    // this is the old 1-D version of binarized weights, will be removed in the final version
    // SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_fullc::kWeight, Shape1(param_.num_hidden * num_input / mxnet::op::xnor::BITS_PER_BINARY_WORD));


    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, binary_inference_fullc::kBias, Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.num_hidden));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        if (i == binary_inference_fullc::kWeight) {
          continue;
        }
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }    
    
    // get binary weight variable type  
    (*in_type)[binary_inference_fullc::kWeight] = mxnet::op::xnor::corresponding_dtype();
    
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    BinaryInferenceFullyConnectedProp* fc_sym = new BinaryInferenceFullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "BinaryInferenceFullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[binary_inference_fullc::kOut], in_data[binary_inference_fullc::kData], in_data[binary_inference_fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[binary_inference_fullc::kData], in_grad[binary_inference_fullc::kData]}};
  }

  std::vector<ResourceRequest> ForwardResource(
          const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BinaryInferenceFullyConnectedParam param_;
};  // class BinaryInferenceFullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_BINARY_INFERENCE_FULLY_CONNECTED_INL_H_

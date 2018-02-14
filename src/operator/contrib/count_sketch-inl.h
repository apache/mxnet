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
 * \file count_sketch-inl.h
 * \brief count_sketch operator and symbol
 * \author Chen Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_COUNT_SKETCH_INL_H_
#define MXNET_OPERATOR_CONTRIB_COUNT_SKETCH_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>

#include <string>
#include <utility>
#include "../operator_common.h"
namespace mxnet {
namespace op {
//  Declare enumeration of input order to make code more intuitive.
//  These enums are only visible within this header
namespace CountSketch {
enum  CountSketchOpInputs{kData, kH, kS};
enum  CountSketchOpOutputs{kOut};
}  //  namespace CountSketch

// seems that we can infer all the parameters from data shapes at the moment
struct CountSketchParam : public dmlc::Parameter<CountSketchParam> {
    int out_dim;
    int processing_batch_size;
    DMLC_DECLARE_PARAMETER(CountSketchParam) {
        DMLC_DECLARE_FIELD(out_dim)
        .describe("The output dimension.");
        DMLC_DECLARE_FIELD(processing_batch_size).set_default(32)
        .describe("How many sketch vectors to process at one time.");
    }
};

template<typename xpu, typename DType>
class CountSketchOp : public Operator {
 public:
    explicit CountSketchOp(CountSketchParam param) {
        this->param_ = param;
    }

    virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        CHECK_EQ(in_data.size(), 3);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        // use FlatTo2D to preseve the possible 4D shape
        // h and s should be 1d vectors
        Tensor<xpu, 2, DType> data = in_data[CountSketch::kData].FlatTo2D<xpu, DType>(s);

        const TShape& hshape = in_data[CountSketch::kH].shape_;
        const TShape& sshape = in_data[CountSketch::kS].shape_;
        Tensor<xpu, 1, DType> h = in_data[CountSketch::kH].get_with_shape<xpu, 1, DType>(
            Shape1(hshape.ProdShape(0, hshape.ndim())), s);
        Tensor<xpu, 1, DType> ss = in_data[CountSketch::kS].get_with_shape<xpu, 1, DType>(
            Shape1(sshape.ProdShape(0, sshape.ndim())), s);
        Tensor<xpu, 2, DType> out = out_data[CountSketch::kOut].FlatTo2D<xpu, DType>(s);
        n_samples = data.shape_[0];
        in_dim = data.shape_[1];
    // firstly set out to zero as we will use sum
    out = 0;
        CountSketchForward(out, data, h, ss, n_samples,
                           this->param_.processing_batch_size, in_dim, this->param_.out_dim);
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
    Tensor<xpu, 2, DType> ograd = out_grad[CountSketch::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> dgrad = in_grad[CountSketch::kData].FlatTo2D<xpu, DType>(s);

    const TShape& hshape = in_data[CountSketch::kH].shape_;
    const TShape& sshape = in_data[CountSketch::kS].shape_;
        Tensor<xpu, 1, DType> h = in_data[CountSketch::kH].get_with_shape<xpu, 1, DType>(
                                            Shape1(hshape.ProdShape(0, hshape.ndim())), s);
    Tensor<xpu, 1, DType> ss = in_data[CountSketch::kS].get_with_shape<xpu, 1, DType>(
                                            Shape1(sshape.ProdShape(0, sshape.ndim())), s);

    CountSketchBackward(dgrad, ograd, h, ss, n_samples,
            this->param_.processing_batch_size, in_dim, this->param_.out_dim);
    }

 private:
    CountSketchParam param_;
    int n_samples;
    int in_dim;
};  // class CountSketchOp

// Declare Factory Function
template<typename xpu>
Operator* CreateOp(CountSketchParam param, int dtype);

#if DMLC_USE_CXX11
class CountSketchProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
        return {"data", "h", "s"};
    }
    std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  int NumOutputs() const override {
    return 1;
  }
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) <<"Input:[data, h, s]";
    const TShape &dshape = (*in_shape)[CountSketch::kData];
    // require data to be known
    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    if (dshape.ndim() == 4) {
      // check the shapes of h and s
        CHECK_EQ((*in_shape)[CountSketch::kH][1], dshape[3])
            << "H should be 2D tensor with same length as input shape[3], "
                        << (*in_shape)[CountSketch::kH][1] << " v.s. " << dshape[3];
        CHECK_EQ((*in_shape)[CountSketch::kS][1], dshape[3])
            << "S should be 2D tensor with same length as input shape[3], "
                        << (*in_shape)[CountSketch::kS][1] << " v.s. " << dshape[3];

        out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], param_.out_dim));
    } else if (dshape.ndim() == 2) {
        CHECK_EQ((*in_shape)[CountSketch::kH][1], dshape[1])
           << "H should be 2D tensor with same length as input shape[1], "
                        << (*in_shape)[CountSketch::kH][1] << " v.s. " << dshape[1];
        CHECK_EQ((*in_shape)[CountSketch::kS][1], dshape[1])
            << "S should be 2D tensor with same length as input shape[1], "
                        << (*in_shape)[CountSketch::kS][1] << " v.s. " << dshape[1];
        out_shape->push_back(Shape2(dshape[0], param_.out_dim));
    } else {
        CHECK_EQ(dshape.ndim(), 2) <<"Data should be 2D or 4D!";
    return false;
    }
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
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    CountSketchProp* cs_sym = new CountSketchProp();
    cs_sym->param_ = this->param_;
    return cs_sym;
  }

  std::string TypeString() const override {
    return "_contrib_count_sketch";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[CountSketch::kOut], in_data[CountSketch::kData],
            in_data[CountSketch::kH], in_data[CountSketch::kS]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[CountSketch::kData], in_grad[CountSketch::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
    CountSketchParam param_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_COUNT_SKETCH_INL_H_

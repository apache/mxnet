/*!
 * Copyright (c) 2015 by Contributors
 * \file swapaxis-inl.h
 * \brief
 * \author Ming Zhang
*/
#ifndef MXNET_OPERATOR_SWAPAXIS_INL_H_
#define MXNET_OPERATOR_SWAPAXIS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

#define SWAPAXIS_DBG 1

namespace mxnet {
namespace op {

struct SwapAxis{
enum SwapAxisOpInputs {kData};
enum SwapAxisOpOutputs {kOut};
};


struct SwapAxisParam : public dmlc::Parameter<SwapAxisParam> {
  // use int for enumeration
  uint32_t dim1, dim2;
  DMLC_DECLARE_PARAMETER(SwapAxisParam) {
    DMLC_DECLARE_FIELD(dim1)
    .set_default(0)
    .describe("the first axis to be swapped.");
    DMLC_DECLARE_FIELD(dim2)
    .set_default(0)
    .describe("the second axis to be swapped.");
  }
};

//a1 higher dimension to be swapped, assert a1 > a2
//a2 lower dimension to be swapped
template<typename xpu>
class SwapAxisOp : public Operator {
 public:
  explicit SwapAxisOp(SwapAxisParam p) {
    CHECK_LT(p.dim1, p.dim2) << "dim1 must be lower than dim2.";
#if SWAPAXIS_DBG
    printf("hello swapaxis SwapAxisOp:dim1:%d, dim2:%d!\n", p.dim1, p.dim2);
#endif
  	this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
#if SWAPAXIS_DBG
    printf("hello swapaxis Forward!\n");
#endif
    Stream<xpu> *s = ctx.get_stream<xpu>();

    uint32_t dim1 = param_.dim1;
    uint32_t dim2 = param_.dim2;
    
    TBlob data_in = in_data[SwapAxis::kData];
    TBlob data_out = out_data[SwapAxis::kData];
#if 0 //free dimension
    int ndim_in = data_in.ndim();

    int aDims_in = new int[ndim_in];
    int aDims5[5];
    int si;
    for (si = 0; si < ndim_in; si++)
    {
      aDims_in[si] = data_in.size(si);
    }
    for (si = 0; si < 5; si++)
    {
      aDims5[si] = 1;
    }
    //dim_0
    for (si = 0; si < dim1; si++)
    {
      aDims5[0] *= aDims_in[si];
    }
    //dim_1
    aDims5[1] = aDims_in[dim1];
    //dim_2
    for (si = dim1+1; si < dim2; si++)
    {
      aDims5[2] *= aDims_in[si];
    }
    //dim_3
    aDims5[3] = aDims_in[dim2];
    //dim_4
    for (si = dim2+1; si < ndim_in; si++)
    {
      aDims5[4] *= aDims_in[si];
    }
    
    Shape<5> inter_shape = Shape5(aDims5[0], aDims5[1], aDims5[2], aDims5[3], aDims5[4]);
#else //fix 4 dimension
    Shape<4> shape_in = data_in.shape_.get<4>();
    Shape<5> inter_shape = Shape5(shape_in.ProdShape(0, dim1), shape_in[dim1], 
                                 shape_in.ProdShape(dim1+1, dim2), shape_in[dim2],
                                 shape_in.ProdShape(dim2+1, 4));
#endif    
    Tensor<xpu, 5> inter_data_in = data_in.get_with_shape<xpu, 5, real_t>(inter_shape, s);
//    Tensor<xpu, 5> inter_data = swapaxis<3, 1>(inter_data_in);
//    swapaxis<3, 1>(inter_data_in);
    
    Tensor<xpu, 4> out = data_out.get<xpu, 4, real_t>(s);
    Shape<4> shape_out = data_out.shape_.get<4>();
    int dwTmp = 0;
    Shape<4> shape_in_tmp = shape_in;
    dwTmp = shape_in_tmp[dim1];
    shape_in_tmp[dim1] = shape_in_tmp[dim2];
    shape_in_tmp[dim2] = dwTmp;
    
    assert(shape_out == shape_in_tmp);
    
    out = reshape(swapaxis<3, 1>(inter_data_in), shape_out);
    
#if 0
    delete []aDims_in;
#endif
  }

  virtual void Backward(const OpContext &ctx,
                       const std::vector<TBlob> &out_grad,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &in_grad,
                       const std::vector<TBlob> &aux_args) {
  	
  }
  
  SwapAxisParam param_;
};


template<typename xpu>
Operator* CreateOp(SwapAxisParam param);


#if DMLC_USE_CXX11
class SwapAxisProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
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
    int input_num = in_shape->size();
    if (input_num == 0)
    {
      std::cout << "Have no input data.\n";
      return false;
    }
    TShape &shape0 = (*in_shape)[SwapAxis::kData];
#if SWAPAXIS_DBG
    printf("in_shape_num:%d\n", input_num);
    printf("in_shape_0, dim:%d, size:%d\n", (int)shape0.ndim(), (int)shape0.Size());
#endif
    if (shape0.ndim() !=  4)
    {
      std::cout << "Input data should be 4D.\n";
      return false;
    }
    out_shape->clear();
    out_shape->push_back(shape0);
    TShape &shape1 = (*out_shape)[SwapAxis::kOut];
#if 1
    int tmp = 0;
    tmp = shape1[param_.dim1];
    shape1[param_.dim1] = shape1[param_.dim2];
    shape1[param_.dim2] = tmp;
#endif
#if SWAPAXIS_DBG
    for (int i = 0; i < 4; i++)
    {
      printf("%d[%d], ", shape1[i], shape0[i]);
    }
    printf("\n");
#endif
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SwapAxisProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SwapAxis";
  }
/*
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override;

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override;

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override;
*/
  Operator* CreateOperator(Context ctx) const override;

 private:
  SwapAxisParam param_;
};  // class SwapAxisProp
#endif  // DMLC_USE_CXX11


}
}

#endif



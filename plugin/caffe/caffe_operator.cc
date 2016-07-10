/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator.cc
 * \brief caffe operator
 * \author Haoran Wang 
*/
#include "./caffe_operator-inl.h"
#include "./caffe_operator_util.h"
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(CaffeOperatorParam param) {
  return new CaffeOperator<cpu>(param);
}

template<> void CaffeOperator<cpu>::CaffeForward(std::vector<::caffe::Blob<float>*> bottom, \
    std::vector<::caffe::Blob<float>*> top) {
  caffeOp_->Forward(bottom, top);
}

template<> void CaffeOperator<cpu>::CaffeBackward(std::vector<::caffe::Blob<float>*> top, \
    std::vector<bool> bp_flags, std::vector<::caffe::Blob<float>*> bottom) {
  caffeOp_->Backward(top, bp_flags, bottom);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* CaffeOperatorProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CaffeOperatorParam);

MXNET_REGISTER_OP_PROPERTY(CaffeOperator, CaffeOperatorProp)
.describe("Apply caffe operator")
.add_argument("data", "Symbol[]", "List of tensors")
.add_arguments(CaffeOperatorParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

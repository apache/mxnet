/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file channel_operator.cc
 * \brief 
 * \author Haozhi Qi, Yi Li, Guodong Zhang, Jifeng Dai
*/
#include "./channel_operator-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
  template<typename DType>
  inline void GroupMaxForward(const Tensor<cpu, 4, DType> &out,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 4, DType> &max_idx,
    const int group) {
    // NOT_IMPLEMENTED;
    return;
  }
  template<typename DType>
  inline void GroupPickForward(const Tensor<cpu, 4, DType> &out,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 4, DType> &pick_idx,
    const int group) {
    // NOT_IMPLEMENTED;
    return;
  }

  template<typename DType>
  inline void GroupMaxBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
    const Tensor<cpu, 4, DType> &out_grad,
    const Tensor<cpu, 4, DType> &max_idx,
    const int group) {
    // NOT_IMPLEMENTED;
    return;
  }

  template<typename DType>
  inline void GroupPickBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
    const Tensor<cpu, 4, DType> &out_grad,
    const Tensor<cpu, 4, DType> &pick_idx,
    const int group) {
    // NOT_IMPLEMENTED;
    return;
  }

  template<typename DType>
  inline void GetMaxIdx(const Tensor<cpu, 4, DType> &pick_score,
    const Tensor<cpu, 4, DType> &argmax,
    const int group) {
    // NOT_IMPLEMENTED;
    return;
  }
}  // namespace mshadow

namespace mxnet {
  namespace op {

    template<>
    Operator *CreateOp<cpu>(ChannelOperatorParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new ChannelOperatorOp<cpu, DType>(param);
      });
      return op;
    }

    Operator *ChannelOperatorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
    }

    DMLC_REGISTER_PARAMETER(ChannelOperatorParam);

    MXNET_REGISTER_OP_PROPERTY(_contrib_ChannelOperator, ChannelOperatorProp)
      .describe("Performs channel operation on inputs, including GroupMax, GroupSoftmax,  GroupPick "
        "and ChannelPick. This layer is designed for FCIS ")
      .add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
      .add_argument("pick_idx", "Symbol", "In GroupPick or ChannelPick mode, pick_idx is used to"
        "pick specific group or channel")
      .add_arguments(ChannelOperatorParam::__FIELDS__());
  }  // namespace op
}  // namespace mxnet
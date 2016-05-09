/*!
 *  Copyright (c) 2015 by Contributors
 * \file loss_binary_op-inl.h
 * \brief Loss functions
 */
#ifndef MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_
#define MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

// return a shape of scalar
inline TShape SoftmaxCrossEntropyShape_(const TShape& lshape,
                                        const TShape& rshape,
                                        const EnvArguments& env) {
  mshadow::index_t shape[] = {1};
  CHECK_EQ(lshape.ndim(), 2)
      << "SoftmaxCrossEntropy only accept 2D data";
  CHECK_EQ(rshape.ndim(), 1)
      << "SoftmaxCrossEntropy only accept 1D label";
  CHECK_EQ(lshape[0], rshape[0])
      << "SoftmaxCrossEntropy: data label shape mismatch";
  return TShape(shape, shape + 1);
}

template<typename xpu>
void SoftmaxCrossEntropyForward_(const TBlob& data,
                                 const TBlob& label,
                                 const EnvArguments& env,
                                 TBlob *ret,
                                 OpReqType req,
                                 RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(env.resource.size(), 1);
  CHECK_EQ(ret->type_flag_, data.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, label.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> out = ret->get<xpu, 1, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlabel = label.get<xpu, 1, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mdata = data.get<xpu, 2, DType>(s);
      mshadow::Tensor<xpu, 1, DType> workspace = env.resource[0].get_space_typed<xpu, 1, DType>(
          mshadow::Shape1(mdata.shape_.Size() + mlabel.size(0)), s);
      mshadow::Tensor<xpu, 2, DType> temp1(
          workspace.dptr_, mdata.shape_, s);
      mshadow::Tensor<xpu, 2, DType> temp2(
          workspace.dptr_ + mdata.shape_.Size(),
          mshadow::Shape2(1, mlabel.size(0)), s);
      // calculate softmax on temp
      // TODO(tqchen): change to SoftmaxLog later
      mshadow::Softmax(temp1, mdata);
      // choose the softmax rows
      mshadow::Tensor<xpu, 1, DType> tdst = temp2[0];
      tdst = F<mshadow_op::negation>(
          F<mshadow_op::log>(
              F<mshadow_op::maximum>(mat_choose_row_element(temp1, mlabel),
                                     scalar<DType>(1e-8f))));
      ASSIGN_DISPATCH(out, req, sumall_except_dim<0>(temp2));
    });
}

template<typename xpu>
void SoftmaxCrossEntropyBackward_(const OutputGrad& scale,
                                  const Input0& data,
                                  const Input1& label,
                                  const EnvArguments& env,
                                  TBlob* data_grad,
                                  TBlob* label_grad,
                                  OpReqType req_data_grad,
                                  OpReqType req_label_grad,
                                  RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(env.resource.size(), 1);
  CHECK_EQ(req_label_grad, kNullOp)
      << "SoftmaxCrossEntropy: Cannot take gradient wrt label";
  MSHADOW_TYPE_SWITCH(data_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mlabel = label.data.get<xpu, 1, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mdata = data.data.get<xpu, 2, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mdata_grad = data_grad->get<xpu, 2, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mscale = scale.data.get<xpu, 1, DType>(s);
      mshadow::Tensor<xpu, 2, DType> temp = env.resource[0].get_space_typed<xpu, 2, DType>(
          mdata.shape_, s);
      mshadow::Softmax(temp, mdata);
      mshadow::SoftmaxGrad(temp, temp, mlabel);
      ASSIGN_DISPATCH(mdata_grad, req_data_grad,
                      broadcast_scalar(mscale, temp.shape_) * temp);
    });
}

MXNET_REGISTER_SIMPLE_OP(softmax_cross_entropy, XPU)
.set_function(XPU::kDevMask, SoftmaxCrossEntropyForward_<XPU>, kNoInplace)
.set_gradient(XPU::kDevMask, SoftmaxCrossEntropyBackward_<XPU>, kNoInplace)
.set_resource_request(ResourceRequest::kTempSpace)
.set_shape_function(SoftmaxCrossEntropyShape_)
.describe("Calculate cross_entropy(lhs, one_hot(rhs))");
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LOSS_BINARY_OP_INL_H_

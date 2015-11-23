/*!
 * Copyright (c) 2015 by Contributors
 * \file ndarray_op.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./ndarray_op-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#define APPEND(source, tag)                                                       \
do {                                                                              \
  for (auto& blob : source) {                                                     \
        ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id))); \
        tags.push_back(tag);                                                      \
  }                                                                               \
}while(0)

namespace mxnet {
namespace op {
template<>
Context NDArrayOp<cpu>::get_ctx() {
  return Context::CPU();
}

template<>
Operator *CreateOp<cpu>(NDArrayOpParam param) {
  return new NDArrayOp<cpu>(param);
}

#if MXNET_USE_CUDA
template<>
Context NDArrayOp<gpu>::get_ctx() {
  int dev_id;
  CHECK_EQ(cudaGetDevice(&dev_id), cudaSuccess);
  return Context::GPU(dev_id);
}

template<>
Operator* CreateOp<gpu>(NDArrayOpParam param) {
  return new NDArrayOp<gpu>(param);
}
#endif  // MXNET_USE_CUDA

template<typename xpu>
void NDArrayOp<xpu>::Forward(const OpContext &ctx,
                   const std::vector<TBlob> &in_data,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &out_data,
                   const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);
  APPEND(in_data, 0);
  APPEND(out_data, 1);
  param_.pinfo->forward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_forward);
}

template<typename xpu>
void NDArrayOp<xpu>::Backward(const OpContext &ctx,
                    const std::vector<TBlob> &out_grad,
                    const std::vector<TBlob> &in_data,
                    const std::vector<TBlob> &out_data,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &in_grad,
                    const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<int> tags;
  for (auto& i : req) CHECK_NE(i, kAddTo);
  APPEND(in_data, 0);
  APPEND(out_data, 1);
  APPEND(in_grad, 2);
  APPEND(out_grad, 3);
  param_.pinfo->backward(ptrs.size(), ptrs.data(), tags.data(), param_.pinfo->p_backward);
}

Operator* NDArrayOpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(NDArrayOpParam);

MXNET_REGISTER_OP_PROPERTY(_NDArray, NDArrayOpProp)
.describe("Stub for implementing an operator implemented in native frontend language with ndarray.")
.add_arguments(NDArrayOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

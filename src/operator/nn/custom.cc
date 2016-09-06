/*!
 * Copyright (c) 2015 by Contributors
 * \file custom.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./custom-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace op {
std::map<std::string, CustomOpPropCreator> CustomOpProp::registry_;

template<>
Context CustomOp<cpu>::get_ctx() {
  return Context::CPU();
}

template<>
Operator *CreateOp<cpu>(CustomOpInfo *op_info) {
  return new CustomOp<cpu>(op_info);
}

#if MXNET_USE_CUDA
template<>
Context CustomOp<gpu>::get_ctx() {
  int dev_id;
  CHECK_EQ(cudaGetDevice(&dev_id), cudaSuccess);
  return Context::GPU(dev_id);
}

template<>
Operator* CreateOp<gpu>(CustomOpInfo *op_info) {
  return new CustomOp<gpu>(op_info);
}
#endif  // MXNET_USE_CUDA

template<typename xpu>
void CustomOp<xpu>::Forward(const OpContext &ctx,
                           const std::vector<TBlob> &in_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &out_data,
                           const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<NDArray> ndcpy;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  std::vector<int> reqs(req.begin(), req.end());

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndcpy.push_back(*nd);
    ndvar.push_back(nd->var());
    tags.push_back(1);
  }
  for (auto& blob : aux_args) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndcpy.push_back(*nd);
    ndvar.push_back(nd->var());
    tags.push_back(4);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());

  CHECK(
    op_info_->forward(ptrs.size(),
    ptrs.data(), tags.data(),
    reqs.data(),
    ctx.is_train,
    op_info_->p_forward));

  // NDArray* in ptrs is freed by frontend side. We keep a copy in ndcpy to keep ndvar alive
  Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx) {
      ctx.async_on_complete();
    }, ndctx, ndvar, {});
}

template<typename xpu>
void CustomOp<xpu>::Backward(const OpContext &ctx,
                            const std::vector<TBlob> &out_grad,
                            const std::vector<TBlob> &in_data,
                            const std::vector<TBlob> &out_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &in_grad,
                            const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  Context ndctx = get_ctx();
  std::vector<void*> ptrs;
  std::vector<NDArray> ndcpy;
  std::vector<Engine::VarHandle> ndvar;
  std::vector<int> tags;
  std::vector<int> reqs(req.begin(), req.end());

  for (auto& blob : in_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(0);
  }
  for (auto& blob : out_data) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(1);
  }
  for (auto& blob : in_grad) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndcpy.push_back(*nd);
    ndvar.push_back(nd->var());
    tags.push_back(2);
  }
  for (auto& blob : aux_args) {
    NDArray* nd = new NDArray(blob, ndctx.dev_id);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    ndcpy.push_back(*nd);
    ndvar.push_back(nd->var());
    tags.push_back(4);
  }
  std::sort(ndvar.begin(), ndvar.end());
  ndvar.resize(std::unique(ndvar.begin(), ndvar.end()) - ndvar.begin());
  for (auto& blob : out_grad) {
    ptrs.push_back(reinterpret_cast<void*>(new NDArray(blob, ndctx.dev_id)));
    tags.push_back(3);
  }

  CHECK(
    op_info_->backward(ptrs.size(),
    ptrs.data(),
    tags.data(),
    reqs.data(),
    true,
    op_info_->p_backward));
  // NDArray* in ptrs is freed by frontend side. We keep a copy in ndcpy to keep ndvar alive
  Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx){
      ctx.async_on_complete();
    }, ndctx, ndvar, {});
}

Operator* CustomOpProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<unsigned*> shapes;
  std::vector<int> ndims;
  for (auto iter = in_shape->begin(); iter != in_shape->end(); ++iter) {
    shapes.push_back(iter->data());
    ndims.push_back(iter->ndim());
  }
  std::string str_ctx;
  if (ctx.dev_mask() == cpu::kDevMask) {
    str_ctx = "cpu";
  } else {
    str_ctx = "gpu";
  }
  CustomOpInfo *op_info = new CustomOpInfo;
  CHECK(info_->create_operator(str_ctx.c_str(), shapes.size(), shapes.data(),
                              ndims.data(), in_type->data(), op_info, info_->p_create_operator));
  DO_BIND_DISPATCH(CreateOp, op_info);
}

MXNET_REGISTER_OP_PROPERTY(Custom, CustomOpProp)
.describe("Custom operator implemented in frontend.")
.add_argument("op_type", "string", "Type of custom operator. Must be registered first.");

}  // namespace op
}  // namespace mxnet

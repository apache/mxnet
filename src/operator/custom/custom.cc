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
Operator *CreateOp<cpu>(MXCallbackList *op_info) {
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
Operator* CreateOp<gpu>(MXCallbackList *op_info) {
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

  auto compute = [=]() mutable {
      CHECK(reinterpret_cast<CustomOpFBFunc>(op_info_->callbacks[kCustomOpForward])(
        ptrs.size(), ptrs.data(), tags.data(), reqs.data(),
        static_cast<int>(ctx.is_train), op_info_->contexts[kCustomOpForward]));

      // NDArray* in ptrs is freed by frontend side. We keep a copy in ndcpy to keep ndvar alive
      Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx) {
          ctx.async_on_complete();
        }, ndctx, ndvar, {},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("CustomOpForward"));
    };

  if (sync_mode_) {
    compute();
  } else {
    std::unique_lock<std::mutex> lock(mtx_);
    q_.push(compute);
    cv_.notify_all();
  }
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

  auto compute = [=]() mutable {
      CHECK(reinterpret_cast<CustomOpFBFunc>(op_info_->callbacks[kCustomOpBackward])(
        ptrs.size(), ptrs.data(), tags.data(), reqs.data(), 1,
        op_info_->contexts[kCustomOpBackward]));

      // NDArray* in ptrs is freed by frontend side. We keep a copy in ndcpy to keep ndvar alive
      Engine::Get()->PushSync([ndcpy, ctx](RunContext rctx){
          ctx.async_on_complete();
        }, ndctx, ndvar, {},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("CustomOpBackward"));
    };

  if (sync_mode_) {
    compute();
  } else {
    std::unique_lock<std::mutex> lock(mtx_);
    q_.push(compute);
    cv_.notify_all();
  }
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
  MXCallbackList *op_info = new MXCallbackList;

  CHECK(reinterpret_cast<CustomOpCreateFunc>(info_->callbacks[kCustomOpPropCreateOperator])(
    str_ctx.c_str(), shapes.size(), shapes.data(), ndims.data(), in_type->data(), op_info,
    info_->contexts[kCustomOpPropCreateOperator]));
  DO_BIND_DISPATCH(CreateOp, op_info);
}

MXNET_REGISTER_OP_PROPERTY(Custom, CustomOpProp)
.describe(R"code(Apply a custom operator implemented in a frontend language (like Python).

Custom operators should override required methods like `forward` and `backward`.
The custom operator must be registered before it can be used.
Please check the tutorial here: http://mxnet.io/how_to/new_op.html.

)code")
.add_argument("op_type", "string", "Name of the custom operator. "
              "This is the name that is passed to `mx.operator.register` "
              "to register the operator.")
.add_argument("data", "NDArray-or-Symbol", "Input data for the custom operator.");


}  // namespace op
}  // namespace mxnet

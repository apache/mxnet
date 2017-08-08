/*!
 * Copyright (c) 2015 by Contributors
 * \file custom.cc
 * \brief
 * \author Junyuan Xie
*/
#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include "./c_api_common.h"
#include "../ndarray/autograd.h"

namespace mxnet {
namespace custom_function {

struct CustomFunctionParam {
  size_t num_args, num_outs;
  std::shared_ptr<MXCallbackList> info;
};

std::vector<nnvm::NodeEntry> Gradient(
    const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& out_grads) {
  const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(n->attrs.parsed);

  nnvm::NodePtr g = nnvm::Node::Create();
  g->attrs.op = nnvm::Op::Get("_backward_CustomFunction");
  g->attrs.name = n->attrs.name + "_backward";
  g->attrs.parsed = params;
  g->control_deps.emplace_back(n);

  g->inputs = out_grads;

  std::vector<nnvm::NodeEntry> ret;
  for (index_t i = 0; i < g->num_outputs(); ++i) {
    ret.emplace_back(nnvm::NodeEntry{g, i, 0});
  }

  return ret;
}

void Backward(const OpStatePtr& state,
              const OpContext& ctx,
              const std::vector<NDArray>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<NDArray>& outputs) {
  const CustomFunctionParam& params = state.get_state<CustomFunctionParam>();

  std::vector<NDArrayHandle> ptrs(params.num_outs + params.num_args);

  for (const auto& i : inputs) {
    NDArray* nd = new NDArray(i.Detach());
    ptrs.push_back(reinterpret_cast<void*>(nd));
  }
  for (const auto& i : outputs) {
    NDArray* nd = new NDArray(i.Detach());
    ptrs.push_back(reinterpret_cast<void*>(nd));
  }


  bool prev_recording = autograd::AutogradRuntime::Get()->SetIsRecording(false);
  bool prev_training = autograd::AutogradRuntime::Get()->SetIsTraining(ctx.is_train);

  CHECK(reinterpret_cast<CustomFunctionBwdFunc>(
      params.info->callbacks[kCustomFunctionBackward])(
          inputs.size(), outputs.size(), ptrs.data(),
          reinterpret_cast<const int*>(req.data()), ctx.is_train,
          params.info->contexts[kCustomFunctionBackward]));

  autograd::AutogradRuntime::Get()->SetIsTraining(prev_training);
  autograd::AutogradRuntime::Get()->SetIsRecording(prev_recording);
}


NNVM_REGISTER_OP(_CustomFunction)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    return params.num_args;
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    return params.num_outs;
  })
.set_attr<nnvm::FInferShape>("FInferShape",
  [](const NodeAttrs& attrs, std::vector<TShape> *in_shape,
     std::vector<TShape> *out_shape) {
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const NodeAttrs& attrs, std::vector<int> *in_type,
     std::vector<int> *out_type) {
    return true;
  })
.set_attr<nnvm::FGradient>("FGradient", Gradient);


NNVM_REGISTER_OP(_backward_CustomFunction)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    return params.num_outs;
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    return params.num_args;
  })
.set_attr<bool>("TIsBackward", true)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
    return ExecType::kLocal;
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", Backward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", Backward);

}  // namespace custom_function
}  // namespace mxnet

int MXCustomFunctionRecord(int num_inputs, NDArrayHandle *inputs,
                           int num_outputs, NDArrayHandle *outputs,
                           MXCallbackList *callbacks) {
  using namespace mxnet;
  using namespace mxnet::custom_function;
  using mxnet::autograd::AutogradRuntime;
  API_BEGIN();
  CHECK(AutogradRuntime::Get()->IsRecording());
  std::vector<NDArray> ndinputs, ndoutputs;
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.emplace_back(*reinterpret_cast<NDArray*>(inputs[i]));
  }
  for (int i = 0; i < num_outputs; ++i) {
    ndoutputs.emplace_back(*reinterpret_cast<NDArray*>(outputs[i]));
  }
  CustomFunctionParam params;
  params.num_args = num_inputs;
  params.num_outs = num_outputs;
  params.info.reset(callbacks, [](MXCallbackList* ptr){
      reinterpret_cast<CustomFunctionDelFunc>(ptr->callbacks[kCustomFunctionDelete])(
        ptr->contexts[kCustomFunctionDelete]);
    });
  nnvm::NodeAttrs attrs;
  attrs.parsed = params;
  auto state = OpStatePtr::Create<CustomFunctionParam>(params);
  auto op = nnvm::Op::Get("_CustomFunction");
  AutogradRuntime::Get()->RecordImperativeOperator(
      state, op, attrs, &ndinputs, &ndoutputs);

  API_END();
}

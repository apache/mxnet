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
 * \file custom.cc
 * \brief
 * \author Junyuan Xie
*/
#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/imperative.h>

#include "./c_api_common.h"
#include "../operator/operator_common.h"
#include "../operator/custom/custom-inl.h"

namespace mxnet {
namespace custom_function {

struct CustomFunctionParam {
  size_t num_args, num_outs;
  std::shared_ptr<MXCallbackList> info;
  std::vector<mxnet::TShape> out_shapes;
  std::vector<int> out_dtypes;
};

std::vector<nnvm::NodeEntry> Gradient(
    const nnvm::ObjectPtr& n,
    const std::vector<nnvm::NodeEntry>& out_grads) {
  const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(n->attrs.parsed);

  nnvm::ObjectPtr g = nnvm::Node::Create();
  g->attrs.op = nnvm::Op::Get("_backward_CustomFunction");
  g->attrs.name = n->attrs.name + "_backward";
  g->attrs.parsed = params;
  g->control_deps.emplace_back(n);

  g->inputs = out_grads;

  std::vector<nnvm::NodeEntry> ret;
  for (uint32_t i = 0; i < g->num_outputs(); ++i) {
    ret.emplace_back(g, i, 0);
  }

  return ret;
}

OpStatePtr CreateState(const nnvm::NodeAttrs& attrs,
                       Context ctx,
                       const mxnet::ShapeVector& ishape,
                       const std::vector<int>& itype) {
  LOG(FATAL) << "Not reached";
  return OpStatePtr::Create<void*>(nullptr);
}

void Forward(const OpStatePtr& state,
             const OpContext& ctx,
             const std::vector<NDArray>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<NDArray>& outputs) {
  LOG(FATAL) << "Not reached";
}

void Backward(const OpStatePtr& state,
              const OpContext& ctx,
              const std::vector<NDArray>& inputs,
              const std::vector<OpReqType>& req,
              const std::vector<NDArray>& outputs) {
  const CustomFunctionParam& params = state.get_state<CustomFunctionParam>();

  std::vector<NDArrayHandle> ptrs;
  std::vector<NDArray> cpys;
  std::vector<int> tags;
  std::unordered_set<int> input_tags({0});
  std::unordered_set<int> output_tags({1});

  auto dev_id = ctx.run_ctx.ctx.dev_id;

  for (const auto& i : inputs) {
    NDArray* nd = new NDArray(i.data(), dev_id);
    ptrs.push_back(reinterpret_cast<NDArrayHandle>(nd));
    cpys.push_back(*nd);
    tags.push_back(0);
  }
  for (const auto& i : outputs) {
    NDArray* nd = new NDArray(i.data(), dev_id);
    ptrs.push_back(reinterpret_cast<NDArrayHandle>(nd));
    cpys.push_back(*nd);
    tags.push_back(1);
  }

  op::custom::CustomOperator::Get()->Push(
    [=]() {
      CHECK(reinterpret_cast<CustomFunctionBwdFunc>(
          params.info->callbacks[kCustomFunctionBackward])(
              inputs.size(), outputs.size(),
              const_cast<NDArrayHandle*>(ptrs.data()),
              reinterpret_cast<const int*>(req.data()), ctx.is_train,
              params.info->contexts[kCustomFunctionBackward]));
    }, ctx, false, ctx.is_train, cpys, tags, output_tags, outputs);
}

inline bool InferStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                             DispatchMode* dispatch_mode,
                             std::vector<int>* iattr, std::vector<int>* oattr) {
  using namespace op;

  for (size_t i = 0; i < iattr->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*iattr, i, kDefaultStorage);
  }
  for (size_t i = 0; i < oattr->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*oattr, i, kDefaultStorage);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
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
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const NodeAttrs& attrs, mxnet::ShapeVector *in_shape,
     mxnet::ShapeVector *out_shape) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    *out_shape = params.out_shapes;
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const NodeAttrs& attrs, std::vector<int> *in_type,
     std::vector<int> *out_type) {
    const CustomFunctionParam& params = nnvm::get<CustomFunctionParam>(attrs.parsed);
    *out_type = params.out_dtypes;
    return true;
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateState)
.set_attr<nnvm::FGradient>("FGradient", Gradient)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", Forward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", Forward)
.set_attr<FInferStorageType>("FInferStorageType", InferStorageType);


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
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
    return ExecType::kAsync;
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", Backward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", Backward)
.set_attr<FInferStorageType>("FInferStorageType", InferStorageType);

}  // namespace custom_function
}  // namespace mxnet

int MXCustomFunctionRecord(int num_inputs, NDArrayHandle *inputs,
                           int num_outputs, NDArrayHandle *outputs,
                           MXCallbackList *callbacks) {
  using namespace mxnet;
  using namespace mxnet::custom_function;
  API_BEGIN();
  CHECK(Imperative::Get()->is_recording());
  auto state = OpStatePtr::Create<CustomFunctionParam>();
  CustomFunctionParam& params = state.get_state<CustomFunctionParam>();
  params.num_args = num_inputs;
  params.num_outs = num_outputs;
  params.info.reset(callbacks, [](MXCallbackList* ptr){
      reinterpret_cast<CustomFunctionDelFunc>(ptr->callbacks[kCustomFunctionDelete])(
        ptr->contexts[kCustomFunctionDelete]);
    });
  std::vector<NDArray*> ndinputs, ndoutputs;
  ndinputs.reserve(num_inputs);
  ndoutputs.reserve(num_outputs);
  params.out_shapes.reserve(num_outputs);
  params.out_dtypes.reserve(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.emplace_back(reinterpret_cast<NDArray*>(inputs[i]));
  }
  for (int i = 0; i < num_outputs; ++i) {
    NDArray* arr = reinterpret_cast<NDArray*>(outputs[i]);
    ndoutputs.emplace_back(arr);
    params.out_shapes.emplace_back(arr->shape());
    params.out_dtypes.emplace_back(arr->dtype());
  }
  nnvm::NodeAttrs attrs;
  attrs.op = nnvm::Op::Get("_CustomFunction");
  attrs.parsed = params;
  Imperative::Get()->RecordOp(
      std::move(attrs), ndinputs, ndoutputs, state);

  API_END();
}

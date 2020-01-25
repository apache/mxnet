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
 * \file custom.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./custom-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include "../elemwise_op_common.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace custom {

struct CustomParam {
  std::string op_type;
  size_t num_args, num_outs, num_auxs;
  std::vector<int> bwd_idx;
  std::shared_ptr<MXCallbackList> info;
};

/*! \brief allocate ndarrays from existing ndarrays
 */
inline void AllocateNDArrayCopy(NDArray** nd,
                                const std::vector<NDArray>& inputs,
                                size_t idx, int dev_id) {
  std::vector<TBlob> aux;
  NDArrayStorageType stype = inputs[idx].storage_type();
  switch (stype) {
    case kUndefinedStorage:
    case kDefaultStorage:
      *nd = new NDArray(inputs[idx].data(), dev_id);
      break;
    case kRowSparseStorage:
      aux.push_back(inputs[idx].aux_data(rowsparse::kIdx));
      *nd = new NDArray(stype, inputs[idx].shape(), inputs[idx].data(), aux,
                        dev_id);
      break;
    case kCSRStorage:
      aux.push_back(inputs[idx].aux_data(csr::kIndPtr));
      aux.push_back(inputs[idx].aux_data(csr::kIdx));
      *nd = new NDArray(stype, inputs[idx].shape(), inputs[idx].data(), aux,
                        dev_id);
      break;
  }
}

template<CustomOpPropCallbacks Type>
std::vector<std::string> List(const NodeAttrs& attrs) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
  char ** args = nullptr;
  CHECK(reinterpret_cast<CustomOpListFunc>(
    params.info->callbacks[Type])(
      &args, params.info->contexts[Type]));
  std::vector<std::string> ret;
  for (int i = 0; args[i] != nullptr; ++i) {
    ret.emplace_back(args[i]);
  }
  return ret;
}

void AttrParser(NodeAttrs* attrs) {
  attrs->parsed = CustomParam();
  CustomParam& params = nnvm::get<CustomParam>(attrs->parsed);

  std::vector<const char*> keys, vals;
  for (auto& p : attrs->dict) {
    if (p.first == "op_type") {
      params.op_type = p.second;
    } else {
      keys.push_back(p.first.c_str());
      vals.push_back(p.second.c_str());
    }
  }
  CHECK(!params.op_type.empty()) << "Required argument `op_type` is missing.";
  CustomOpPropCreator creator = CustomOperator::Get()->Find(params.op_type);
  CHECK(CustomOperator::Get()->Find(params.op_type) != nullptr)
      << "Cannot find custom operator " << params.op_type;
  params.info.reset(new MXCallbackList, [](MXCallbackList* ptr){
      reinterpret_cast<CustomOpDelFunc>(ptr->callbacks[kCustomOpPropDelete])(
        ptr->contexts[kCustomOpPropDelete]);
      delete ptr;
    });
  CHECK(creator(params.op_type.c_str(), keys.size(), keys.data(),
                vals.data(), params.info.get()));

  params.num_args = List<kCustomOpPropListArguments>(*attrs).size();
  params.num_outs = List<kCustomOpPropListOutputs>(*attrs).size();
  params.num_auxs = List<kCustomOpPropListAuxiliaryStates>(*attrs).size();

  int num_dep, *rdeps, counter = 0;
  std::vector<int> out_grad, in_data, out_data;
  for (size_t i = 0; i < params.num_outs; ++i) out_grad.push_back(counter++);
  for (size_t i = 0; i < params.num_args; ++i) in_data.push_back(counter++);
  for (size_t i = 0; i < params.num_outs; ++i) out_data.push_back(counter++);
  CHECK(reinterpret_cast<CustomOpBwdDepFunc>(
    params.info->callbacks[kCustomOpPropDeclareBackwardDependency])(
      out_grad.data(), in_data.data(), out_data.data(), &num_dep,
      &rdeps, params.info->contexts[kCustomOpPropDeclareBackwardDependency]));
  params.bwd_idx.insert(params.bwd_idx.end(), rdeps, rdeps+num_dep);
}

bool InferShape(const NodeAttrs& attrs,
                mxnet::ShapeVector *in_shape,
                mxnet::ShapeVector *out_shape) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);

  size_t total = params.num_args + params.num_outs + params.num_auxs;
  std::vector<int*> shapes(total);
  std::vector<int> ndims(total);
  size_t buff_size = 0;
  for (const auto& i : *in_shape) {
    if (i.ndim() > 0) {
      buff_size += i.ndim();
    }
  }
  std::vector<int> buff(buff_size);
  int *ptr = buff.data();
  for (size_t i = 0; i < in_shape->size(); ++i) {
    shapes[i] = ptr;
    ndims[i] = (*in_shape)[i].ndim();
    for (int j = 0; j < (*in_shape)[i].ndim(); ++j, ++ptr) {
      *ptr = (*in_shape)[i][j];
    }
  }

  CHECK(reinterpret_cast<CustomOpInferShapeFunc>(
      params.info->callbacks[kCustomOpPropInferShape])(
          shapes.size(), ndims.data(), shapes.data(),
          params.info->contexts[kCustomOpPropInferShape]));

  for (size_t i = 0; i < params.num_args; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape(shapes[i], shapes[i]+ndims[i]));
  }

  size_t base = params.num_args;
  for (size_t i = 0; i < params.num_outs; ++i) {
    SHAPE_ASSIGN_CHECK(*out_shape, i,
        mxnet::TShape(shapes[base+i], shapes[base+i]+ndims[base+i]));
  }

  base = params.num_args + params.num_outs;
  for (size_t i = 0; i < params.num_auxs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, params.num_args+i,
        mxnet::TShape(shapes[base+i], shapes[base+i]+ndims[base+i]));
  }
  return true;
}

bool InferType(const NodeAttrs& attrs,
               std::vector<int> *in_type,
               std::vector<int> *out_type) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);

  if (params.info->num_callbacks <= kCustomOpPropInferType) {
    return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
        attrs, in_type, out_type, -1);
  }

  std::vector<int> types;
  types.reserve(params.num_args + params.num_outs + params.num_auxs);
  for (size_t i = 0; i < params.num_args; ++i) {
    types.push_back((*in_type)[i]);
  }
  for (const auto& i : *out_type) {
    types.push_back(i);
  }
  for (size_t i = 0; i < params.num_auxs; ++i) {
    types.push_back((*in_type)[params.num_args+i]);
  }

  CHECK(reinterpret_cast<CustomOpInferTypeFunc>(
      params.info->callbacks[kCustomOpPropInferType])(
          types.size(), types.data(), params.info->contexts[kCustomOpPropInferType]));

  for (size_t i = 0; i < params.num_args; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, types[i]);
  }
  for (size_t i = 0; i < params.num_outs; ++i) {
    TYPE_ASSIGN_CHECK(*out_type, i, types[params.num_args+i]);
  }
  for (size_t i = 0; i < params.num_auxs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, params.num_args+i,
                      types[params.num_args+params.num_outs+i]);
  }
  return true;
}

std::vector<nnvm::NodeEntry> Gradient(
    const nnvm::ObjectPtr& n,
    const std::vector<nnvm::NodeEntry>& out_grads) {
  const CustomParam& params = nnvm::get<CustomParam>(n->attrs.parsed);

  nnvm::ObjectPtr g = nnvm::Node::Create();
  g->attrs.op = nnvm::Op::Get("_backward_Custom");
  g->attrs.name = n->attrs.name;
  g->attrs.parsed = params;
  g->control_deps.emplace_back(n);

  g->inputs.reserve(params.bwd_idx.size());
  for (const int& t : params.bwd_idx) {
    size_t i = static_cast<size_t>(t);
    if (i >= params.num_outs + params.num_args) {
      uint32_t idx = static_cast<uint32_t>(i-params.num_outs-params.num_args);
      g->inputs.emplace_back(n, idx, 0);
    } else if (i >= params.num_outs) {
      g->inputs.push_back(n->inputs[i-params.num_outs]);
    } else {
      g->inputs.push_back(out_grads[i]);
    }
  }

  for (size_t i = 0; i < params.num_auxs; ++i) {
    g->inputs.push_back(n->inputs[i+params.num_args]);
  }

  std::vector<nnvm::NodeEntry> ret;
  for (size_t i = 0; i < params.num_args; ++i) {
    ret.emplace_back(g, static_cast<uint32_t>(i), 0);
  }
  if (params.num_auxs) {
    nnvm::ObjectPtr ng = nnvm::Node::Create();
    ng->attrs.op = nnvm::Op::Get("_NoGradient");
    ng->attrs.name = "NoGradient";
    for (size_t i = 0; i < params.num_auxs; ++i) {
      ret.emplace_back(ng, 0, 0);
    }
  }

  return ret;
}


OpStatePtr CreateState(const NodeAttrs& attrs, Context ctx,
                       const mxnet::ShapeVector& in_shape,
                       const std::vector<int>& in_type) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);

  std::vector<uint32_t*> shapes(in_shape.size());
  std::vector<int> ndims(in_shape.size());
  size_t buff_size = 0;
  for (const auto& i : in_shape) buff_size += i.ndim();
  std::vector<uint32_t> buff(buff_size);
  uint32_t *ptr = buff.data();
  for (size_t i = 0; i < in_shape.size(); ++i) {
    shapes[i] = ptr;
    ndims[i] = in_shape[i].ndim();
    for (int j = 0; j < in_shape[i].ndim(); ++j, ++ptr) {
      *ptr = static_cast<uint32_t>(in_shape[i][j]);
    }
  }

  std::ostringstream os;
  os << ctx;

  MXCallbackList *op_info = new MXCallbackList;
  CHECK(reinterpret_cast<CustomOpCreateFunc>(
      params.info->callbacks[kCustomOpPropCreateOperator])(
          os.str().c_str(), shapes.size(), shapes.data(), ndims.data(), in_type.data(),
          op_info, params.info->contexts[kCustomOpPropCreateOperator]));

  CustomParam state = params;
  state.info.reset(op_info, [](MXCallbackList *ptr){
    reinterpret_cast<CustomOpDelFunc>(ptr->callbacks[kCustomOpDelete])(
      ptr->contexts[kCustomOpDelete]);
    delete ptr;
  });

  return OpStatePtr::Create<CustomParam>(state);
}

void ForwardEx(const OpStatePtr& state, const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs) {
  const CustomParam& params = state.get_state<CustomParam>();
  std::vector<void*> ptrs;
  // Tags are provided to the callback to provide the frontend
  std::vector<int> tags;
  std::vector<NDArray> cpys;

  // info on what ndarray is at each position in the input and output vector
  // 0 - Input
  // 1 - Output
  // 4 - aux
  std::unordered_set<int> input_tags({0, 4});
  std::unordered_set<int> output_tags({1});

  auto dev_id = ctx.run_ctx.ctx.dev_id;

  for (size_t i = 0; i < params.num_args; ++i) {
    NDArray* nd;
    AllocateNDArrayCopy(&nd, inputs, i, dev_id);
    cpys.push_back(*nd);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    tags.push_back(0);
  }

  for (size_t i = 0; i < params.num_outs; ++i) {
    NDArray* nd;
    AllocateNDArrayCopy(&nd, outputs, i, dev_id);
    cpys.push_back(*nd);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    tags.push_back(1);
  }

  for (size_t i = 0; i < params.num_auxs; ++i) {
    size_t idx = i + params.num_args;
    NDArray* nd;
    AllocateNDArrayCopy(&nd, inputs, idx, dev_id);
    cpys.push_back(*nd);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    tags.push_back(4);
  }

  CustomOperator::Get()->Push(
      [=]() {
        CHECK(reinterpret_cast<CustomOpFBFunc>(
            params.info->callbacks[kCustomOpForward])(
            ptrs.size(), const_cast<void**>(ptrs.data()),
            const_cast<int*>(tags.data()),
            reinterpret_cast<const int*>(req.data()),
            static_cast<int>(ctx.is_train),
            params.info->contexts[kCustomOpForward]));
      },
      ctx, false, ctx.is_train, cpys, tags, output_tags, outputs, params.op_type);
}

void BackwardEx(const OpStatePtr& state, const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
  const CustomParam& params = state.get_state<CustomParam>();

  size_t total = 2 * params.num_args + 2 * params.num_outs + params.num_auxs;
  std::vector<void*> ptrs(params.num_args + 2 * params.num_outs, nullptr);

  std::vector<int> tags;
  std::vector<NDArray> cpys;

  ptrs.reserve(total);
  tags.reserve(total);
  cpys.reserve(total);

  // info on what ndarray is at each position in the input and output vector
  // 3 - out grads
  // 0 - inputs
  // 1 - outputs
  // 4 - auxs
  // 2 - in grads
  std::unordered_set<int> input_tags({3, 0, 1, 4});
  std::unordered_set<int> output_tags({2});

  for (size_t i = 0; i < params.num_outs; ++i) tags.push_back(3);
  for (size_t i = 0; i < params.num_args; ++i) tags.push_back(0);
  for (size_t i = 0; i < params.num_outs; ++i) tags.push_back(1);

  auto dev_id = ctx.run_ctx.ctx.dev_id;


  for (size_t i = 0; i < params.bwd_idx.size(); ++i) {
    NDArray* nd;
    AllocateNDArrayCopy(&nd, inputs, i, dev_id);
    cpys.push_back(*nd);
    ptrs[params.bwd_idx[i]] = reinterpret_cast<void*>(nd);
  }
  for (auto& ptr : ptrs) {
    NDArray* nd;
    if (ptr == nullptr) {
        nd = new NDArray();
      ptr = reinterpret_cast<void*>(nd);
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    NDArray* nd;
    AllocateNDArrayCopy(&nd, outputs, i, dev_id);
    cpys.push_back(*nd);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    tags.push_back(2);
  }

  for (size_t i = 0; i < params.num_auxs; ++i) {
    size_t idx = inputs.size() - params.num_auxs + i;
    NDArray* nd;
    AllocateNDArrayCopy(&nd, inputs, idx, dev_id);
    cpys.push_back(*nd);
    ptrs.push_back(reinterpret_cast<void*>(nd));
    tags.push_back(4);
  }
  CustomOperator::Get()->Push(
    [=]() {
      CHECK(reinterpret_cast<CustomOpFBFunc>(params.info->callbacks[kCustomOpBackward])(
        ptrs.size(), const_cast<void**>(ptrs.data()), const_cast<int*>(tags.data()),
        reinterpret_cast<const int*>(req.data()), static_cast<int>(ctx.is_train),
        params.info->contexts[kCustomOpBackward]));
    }, ctx, false, ctx.is_train, cpys, tags, output_tags, outputs, "_backward_" + params.op_type);
}

// infer storage backward function for custom op which assigns kDefaultStorage for
// all undefined stypes and dispatches on DispatchMode::kFComputeEx.
inline bool BackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* iattr,
                                     std::vector<int>* oattr) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);

  if (params.info->num_callbacks <= kCustomOpPropBackwardInferStorageType) {
    for (size_t i = 0; i < iattr->size(); i++) {
      STORAGE_TYPE_ASSIGN_CHECK(*iattr, i, kDefaultStorage);
    }
    for (size_t i = 0; i < oattr->size(); i++) {
      STORAGE_TYPE_ASSIGN_CHECK(*oattr, i, kDefaultStorage);
    }
    DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
    return true;
  }

  size_t total = 2 * params.num_args + 2 * params.num_outs + params.num_auxs;
  size_t bwd_deps_size = params.bwd_idx.size();
  std::vector<int> stypes(bwd_deps_size, -1);
  std::vector<int> tags;
  stypes.reserve(total);
  tags.reserve(total);

  for (size_t i = 0; i < bwd_deps_size; i++) {
    if (params.bwd_idx[i] < static_cast<int>(params.num_outs))
      tags.push_back(3);
    else if (params.bwd_idx[i] <
             static_cast<int>(params.num_outs + params.num_args))
      tags.push_back(0);
    else
      tags.push_back(1);
    stypes[i] = (*iattr)[i];
  }

  for (int i : *oattr) {
    stypes.push_back(i);
    tags.push_back(2);
  }

  for (size_t i = (iattr->size() - params.num_auxs); i < iattr->size(); i++) {
    stypes.push_back((*iattr)[i]);
    tags.push_back(4);
  }

  CHECK(reinterpret_cast<CustomOpBackwardInferStorageTypeFunc>(
      params.info->callbacks[kCustomOpPropBackwardInferStorageType])(
      stypes.size(), stypes.data(), tags.data(),
      params.info->contexts[kCustomOpPropBackwardInferStorageType]));

  for (size_t i = 0; i < bwd_deps_size; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*iattr, i, stypes[i]);
  }
  for (size_t i = 0; i < oattr->size(); ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*oattr, i, stypes[i + bwd_deps_size]);
  }
  for (size_t i = 0; i < params.num_auxs; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*iattr, (i + iattr->size() - params.num_auxs),
                              stypes[i + params.num_outs + bwd_deps_size]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

// infer storage function for custom op which assigns kDefaultStorage for
// all undefined stypes and dispatches on DispatchMode::kFComputeEx.
inline bool InferStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                             DispatchMode* dispatch_mode,
                             std::vector<int>* iattr, std::vector<int>* oattr) {
  const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);

  if (params.info->num_callbacks <= kCustomOpPropInferStorageType) {
    for (size_t i = 0; i < iattr->size(); i++) {
      STORAGE_TYPE_ASSIGN_CHECK(*iattr, i, kDefaultStorage);
    }
    for (size_t i = 0; i < oattr->size(); i++) {
      STORAGE_TYPE_ASSIGN_CHECK(*oattr, i, kDefaultStorage);
    }
    DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
    return true;
  }

  std::vector<int> stypes;
  stypes.reserve(params.num_args + params.num_outs + params.num_auxs);
  for (size_t i = 0; i < params.num_args; ++i) {
    stypes.push_back((*iattr)[i]);
  }
  for (const auto& i : *oattr) {
    stypes.push_back(i);
  }
  for (size_t i = 0; i < params.num_auxs; ++i) {
    stypes.push_back((*iattr)[params.num_args + i]);
  }

  CHECK(reinterpret_cast<CustomOpInferStorageTypeFunc>(
      params.info->callbacks[kCustomOpPropInferStorageType])(
      stypes.size(), stypes.data(),
      params.info->contexts[kCustomOpPropInferStorageType]));

  for (size_t i = 0; i < params.num_args; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*iattr, i, stypes[i]);
  }
  for (size_t i = 0; i < params.num_outs; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*oattr, i, stypes[params.num_args + i]);
  }
  for (size_t i = 0; i < params.num_auxs; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*iattr, params.num_args + i,
                              stypes[params.num_args + params.num_outs + i]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

NNVM_REGISTER_OP(Custom)
.add_alias("_npi_Custom")
.describe(R"code(Apply a custom operator implemented in a frontend language (like Python).

Custom operators should override required methods like `forward` and `backward`.
The custom operator must be registered before it can be used.
Please check the tutorial here: https://mxnet.incubator.apache.org/api/faq/new_op

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs){
    const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
    return params.num_args + params.num_auxs;
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
    return params.num_outs;
  })
.set_attr_parser(AttrParser)
.set_attr<mxnet::FInferShape>("FInferShape", InferShape)
.set_attr<nnvm::FInferType>("FInferType", InferType)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    std::vector<std::string> args = List<kCustomOpPropListArguments>(attrs);
    std::vector<std::string> auxs = List<kCustomOpPropListAuxiliaryStates>(attrs);
    args.insert(args.end(), auxs.begin(), auxs.end());
    return args;
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames", List<kCustomOpPropListOutputs>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
    std::vector<uint32_t> ret;
    for (size_t i = 0; i < params.num_auxs; ++i) ret.push_back(i+params.num_args);
    return ret;
  })
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
    return ExecType::kAsync;
  })
.set_attr<nnvm::FGradient>("FGradient", Gradient)
.set_attr<FCreateOpState>("FCreateOpState", CreateState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ForwardEx)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", ForwardEx)
.set_attr<FInferStorageType>("FInferStorageType", InferStorageType)
.add_argument("data", "NDArray-or-Symbol[]", "Input data for the custom operator.")
.add_argument("op_type", "string", "Name of the custom operator. "
              "This is the name that is passed to `mx.operator.register` "
              "to register the operator.");


NNVM_REGISTER_OP(_backward_Custom)
.set_num_inputs([](const NodeAttrs& attrs){
    const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
    return params.bwd_idx.size() + params.num_auxs;
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CustomParam& params = nnvm::get<CustomParam>(attrs.parsed);
    return params.num_args;
  })
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<bool>("TIsBackward", true)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
    return ExecType::kAsync;
  })
.set_attr<FInferStorageType>("FInferStorageType", BackwardInferStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", BackwardEx)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", BackwardEx);

}  // namespace custom
}  // namespace op
}  // namespace mxnet

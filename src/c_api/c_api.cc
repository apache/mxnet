
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
 *  Copyright (c) 2015 by Contributors
 * \file c_api.cc
 * \brief C API of mxnet
 */
#include <vector>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <functional>
#include <utility>
#include "dmlc/base.h"
#include "dmlc/logging.h"
#include "dmlc/io.h"
#include "dmlc/memory_io.h"
#include "dmlc/recordio.h"
#include "dmlc/omp.h"
#include "mxnet/base.h"
#include "mxnet/ndarray.h"
#include "mxnet/operator.h"
#include "mxnet/io.h"
#include "mxnet/c_api.h"
#include "mxnet/kvstore.h"
#include "mxnet/rtc.h"
#include "mxnet/storage.h"
#include "mxnet/libinfo.h"
#include "mxnet/imperative.h"
#include "mxnet/lib_api.h"
#include "../initialize.h"
#include "./c_api_common.h"
#include "../operator/custom/custom-inl.h"
#include "../operator/operator_common.h"
#include "../operator/tensor/matrix_op-inl.h"
#include "../operator/tvmop/op_module.h"
#include "../common/utils.h"
#include "nnvm/pass_functions.h"

using namespace mxnet;

// Internal function to get the information
// from function registry
// Used to implement MXSymbolGetAtomicSymbolInfo and MXFuncGetInfo
template<typename FunRegType>
inline int MXAPIGetFunctionRegInfo(const FunRegType *e,
                                   const char **name,
                                   const char **description,
                                   uint32_t *num_args,
                                   const char ***arg_names,
                                   const char ***arg_type_infos,
                                   const char ***arg_descriptions,
                                   const char **return_type) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();

  API_BEGIN();
  *name = e->name.c_str();
  *description = e->description.c_str();
  *num_args = static_cast<uint32_t>(e->arguments.size());
  if (return_type) *return_type = e->return_type.c_str();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].name.c_str());
  }
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].type_info_str.c_str());
  }
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].description.c_str());
  }
  *arg_names = dmlc::BeginPtr(ret->ret_vec_charp);
  *arg_type_infos = dmlc::BeginPtr(ret->ret_vec_charp) + e->arguments.size();
  *arg_descriptions = dmlc::BeginPtr(ret->ret_vec_charp) + (e->arguments.size() * 2);
  API_END();
}

// NOTE: return value is added in API_END

/*!
 * \brief Loads dynamic library and initializes it
 * \param path library path
 */
int MXLoadLib(const char *path) {
  API_BEGIN();
  void *lib = LibraryInitializer::Get()->lib_load(path);
  if (!lib)
    LOG(FATAL) << "Unable to load library";

  // check that library and MXNet use same version of library API
  opVersion_t opVersion = get_func<opVersion_t>(lib, const_cast<char*>(MXLIB_OPVERSION_STR));
  int libVersion =  opVersion();
  if (MX_LIBRARY_VERSION != libVersion)
    LOG(FATAL) << "Library version (" << libVersion << ") does not match MXNet version ("
               << MX_LIBRARY_VERSION << ")";

  // initialize library by passing MXNet version
  initialize_t initialize = get_func<initialize_t>(lib, const_cast<char*>(MXLIB_INITIALIZE_STR));
  if (!initialize(static_cast<int>(MXNET_VERSION)))
    LOG(FATAL) << "Library failed to initialize";

  // get C type interface functions
  opCallFree_t callFree = get_func<opCallFree_t>(lib, const_cast<char*>(MXLIB_OPCALLFREE_STR));

  opCallParseAttrs_t callParseAttrs =
    get_func<opCallParseAttrs_t>(lib, const_cast<char*>(MXLIB_OPCALLPARSEATTRS_STR));

  opCallInferShape_t callInferShape =
    get_func<opCallInferShape_t>(lib, const_cast<char*>(MXLIB_OPCALLINFERSHAPE_STR));

  opCallInferType_t callInferType =
    get_func<opCallInferType_t>(lib, const_cast<char*>(MXLIB_OPCALLINFERTYPE_STR));

  opCallFComp_t callFComp =
    get_func<opCallFComp_t>(lib, const_cast<char*>(MXLIB_OPCALLFCOMP_STR));

  opCallMutateInputs_t callMutateInputs =
    get_func<opCallMutateInputs_t>(lib, const_cast<char*>(MXLIB_OPCALLMUTATEINPUTS_STR));

  opCallCreateOpState_t callCreateOpState =
    get_func<opCallCreateOpState_t>(lib, const_cast<char*>(MXLIB_OPCALLCREATEOPSTATE_STR));

  opCallFStatefulComp_t callFStatefulComp =
    get_func<opCallFStatefulComp_t>(lib, const_cast<char*>(MXLIB_OPCALLFSTATEFULCOMP_STR));

  // get number of operators registered in the library
  opRegSize_t opRegSize = get_func<opRegSize_t>(lib, const_cast<char*>(MXLIB_OPREGSIZE_STR));
  int numOps = opRegSize();
  LOG(INFO) << "Found " << numOps << " operators in library";

  /*
   * Get all custom operators implementation from custom library
   * loop and register each operator in the library to NNVM
   */
  opRegGet_t opRegGet = get_func<opRegGet_t>(lib, const_cast<char*>(MXLIB_OPREGGET_STR));
  for (int i = 0; i < numOps; i++) {
    const char* name;
    // function pointers holding implementation from custom library
    fcomp_t fcomp_fp = nullptr;
    parseAttrs_t parse_fp = nullptr;
    inferType_t type_fp = nullptr;
    inferShape_t shape_fp = nullptr;
    // optional attributes
    fcomp_t fgrad_fp = nullptr;
    mutateInputs_t mutate_fp = nullptr;
    createOpState_t create_opstate_fp = nullptr;

    // get custom operator implemenation from the dynamic library
    opRegGet(i, &name, &fcomp_fp, &fgrad_fp, &parse_fp, &type_fp, &shape_fp,
                &mutate_fp, &create_opstate_fp);

    // validate custom operator functions from the dynamic library
    CHECK(fcomp_fp != nullptr || create_opstate_fp != nullptr) << "Error loading '" << name
                            << "' custom op, Forward or CreateOpState function was not set.";
    CHECK(parse_fp != nullptr) << "Error loading '" << name
                            << "' custom op, ParseAttrs function was not set.";
    CHECK(type_fp  != nullptr) << "Error loading '" << name
                            << "' custom op, InferType function was not set.";
    CHECK(shape_fp != nullptr) << "Error loading '" << name
                            << "' custom op, InferShape function was not set.";

    LOG(INFO) << "\tOp[" << i << "] " << name;
    std::string name_str(name);

    /*
     * Below are a series of lambda functions that will be registered in the NNVM op registration
     * Each one has the standard MXNet signature and converts to types supported by externally
     * registered operators. 
     */

    // lambda function to call parse attributes
    auto attr_parser = [=](const NodeAttrs* attrs) {
      // convert attributes to vector of char
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs->dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }
      // convert subgraph symbol from node attributes to char*
      std::string subgraph_json;
      if (!attrs->subgraphs.empty()) {
        nnvm::Graph g;
        g.outputs = attrs->subgraphs[0].get()->outputs;
        subgraph_json = nnvm::pass::SaveJSON(g);
        attr_keys.push_back(SUBGRAPH_SYM_JSON);
        attr_vals.push_back(subgraph_json.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      CHECK(callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           &num_in, &num_out))
      << "Error calling ParseAttrs for custom operator '" << name_str << "'";

      // return type void
    };

    // lambda function to call parse attributes and return the number of inputs
    auto num_inputs = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      CHECK(callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           &num_in, &num_out))
      << "Error calling ParseAttrs::num_inputs for custom operator '" << name_str << "'";

      return num_in;
    };

    // lambda function to call parse attributes and return the number of outputs
    auto num_outputs = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      CHECK(callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           &num_in, &num_out))
      << "Error calling ParseAttrs::num_outputs for custom operator '" << name_str << "'";

      return num_out;
    };

    // lambda function to call parse attributes and return the number of inputs and outputs
    // for backward computation
    auto num_inouts = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      CHECK(callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           &num_in, &num_out))
      << "Error calling ParseAttrs::num_outputs for custom operator '" << name_str << "'";

      return num_in + 2*num_out;
    };

    // lambda function to call infer shape
    auto infer_shape = [=] (const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_shape,
                            mxnet::ShapeVector *out_shape) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      std::vector<uint32_t*> inshapes(in_shape->size());
      std::vector<int> indims(in_shape->size());

      // determine amount of memory needed to store all the input shapes
      size_t buff_size = 0;
      for (const auto& i : *in_shape) buff_size += i.ndim();

      // copy input shapes from ShapeVector to raw memory layout
      std::vector<uint32_t> inbuff(buff_size);
      uint32_t *ptr = inbuff.data();
      for (size_t i = 0; i < in_shape->size(); ++i) {
        inshapes[i] = ptr;
        indims[i] = (*in_shape)[i].ndim();
        for (int j = 0; j < (*in_shape)[i].ndim(); ++j, ++ptr) {
          *ptr = static_cast<uint32_t>((*in_shape)[i][j]);
        }
      }

      // output shapes will be allocated by infer shape function
      uint32_t** outshapes = nullptr;
      int* outdims = nullptr;

      CHECK(callInferShape(shape_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           inshapes.data(), indims.data(), in_shape->size(),
                           &outshapes, &outdims, out_shape->size()))
      << "Error calling InferShape for custom operator '" << name_str << "'";

      std::vector<uint32_t*> out_shapes(out_shape->size());
      // determine amount of memory needed to store all the output shapes
      buff_size = 0;
      for (unsigned i = 0; i < out_shape->size(); i++) {
        buff_size += outdims[i];
      }

      // copy output shapes from custom op memory to MXNet memory
      std::vector<uint32_t> outbuff(buff_size);
      ptr = outbuff.data();
      for (unsigned i = 0; i < out_shape->size(); ++i) {
        out_shapes[i] = ptr;
        for (int j = 0; j < outdims[i]; ++j, ++ptr) {
          *ptr = static_cast<uint32_t>(outshapes[i][j]);
        }
      }

      // assign output shapes to ShapeVector
      for (unsigned i = 0; i < out_shape->size(); ++i) {
        SHAPE_ASSIGN_CHECK(*out_shape, i,
                           mxnet::TShape(out_shapes[i], out_shapes[i]+outdims[i]));
      }

      // free memory used by custom op to allocate shapes/dims
      callFree(outdims);
      for (unsigned i = 0; i < out_shape->size(); i++) {
        callFree(outshapes[i]);
      }
      callFree(outshapes);

      return true;
    };

    // lambda function to call infer type
    auto infer_type = [=] (const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type,
                            std::vector<int> *out_type) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // copy input types from in_type
      std::vector<int> intypes(*in_type);

      // output types will be populated by inferType function
      std::vector<int> outtypes(out_type->size());

      CHECK(callInferType(type_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           intypes.data(), in_type->size(),
                           outtypes.data(), out_type->size()))
      << "Error calling InferType for custom operator '" << name_str << "'";

      // copy and assign output types from custom op to MXNet memory
      for (size_t i = 0; i < out_type->size(); i++) {
        TYPE_ASSIGN_CHECK(*out_type, i, outtypes[i]);
      }

      return true;
    };

    // lambda function to convert from external fcompute to internal MXNet types
    auto fcomp_lambda = [=](fcomp_t fcomp_fp,
                            const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      std::vector<void*> in_data, out_data;
      std::vector<const int64_t *> in_shapes, out_shapes;
      std::vector<int> in_dims, out_dims;
      std::vector<int> in_types, out_types;

      // convert input tensors to constituent parts
      for (size_t i = 0; i < inputs.size(); i++) {
        in_data.push_back(inputs[i].data().dptr_);
        in_shapes.push_back(inputs[i].shape().data());
        in_dims.push_back(inputs[i].shape().ndim());
        in_types.push_back(inputs[i].dtype());
      }

      // convert output tensors to constituent parts
      for (size_t i = 0; i < outputs.size(); i++) {
        out_data.push_back(outputs[i].data().dptr_);
        out_shapes.push_back(outputs[i].shape().data());
        out_dims.push_back(outputs[i].shape().ndim());
        out_types.push_back(outputs[i].dtype());
      }

      // get memory resource
      const Resource &resource = ctx.requested[0];
      mshadow::Stream<mxnet::cpu> *cpu_stream = ctx.get_stream<mxnet::cpu>();

      // create lambda that captures stream & resource objects
      // this temp workspace holds memory allocated by custom library via OpResource
      auto cpu_alloc = [&](int size) {
        mshadow::Tensor<mxnet::cpu, 1, char> workspace =
          resource.get_space_typed<mxnet::cpu, 1, char>(mshadow::Shape1(size), cpu_stream);
        return workspace.dptr_;
      };

      // create lambda without captures so that we can cast it to function pointer
      // this needs to be a lambda function so that we can do the decltype cast
      typedef decltype(cpu_alloc) alloc_type;
      auto cpu_malloc = [](void* _cpu_alloc, int size) {
        // cast the void* argument to the type for the cpu_alloc lambda function
        alloc_type* cpualloc = static_cast<alloc_type*>(_cpu_alloc);
        // call cpu_alloc to actually allocate memory and get the pointer
        void* ptr = (*cpualloc)(size);
        return ptr;
      };

      // call fcompute function
      CHECK(callFComp(fcomp_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                      in_shapes.data(), in_dims.data(), in_data.data(),
                      in_types.data(), in_data.size(),
                      out_shapes.data(), out_dims.data(), out_data.data(),
                      out_types.data(), out_data.size(), cpu_malloc, &cpu_alloc))
      << "Error calling FCompute for custom operator '" << name_str << "'";

      // return type void
    };

    auto forward_lambda = [=](const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
      return fcomp_lambda(fcomp_fp, attrs, ctx, inputs, req, outputs);
    };

    auto backward_lambda = [=](const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
      return fcomp_lambda(fgrad_fp, attrs, ctx, inputs, req, outputs);
    };

    // lambda function to convert from external mutate_inputs to internal MXNet types
    auto mutate_inputs = [=](const nnvm::NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // C type placeholder for mutate input indices vector
      int* mutate_indices = nullptr;
      int indices_size = 0;

      // call mutate inputs function
      CHECK(callMutateInputs(mutate_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                      &mutate_indices, &indices_size))
      << "Error calling MutateInputs for custom operator '" << name_str << "'";

      std::vector<uint32_t> mutate_indices_list(indices_size);
      for (int i=0; i < indices_size; i++) {
        mutate_indices_list[i] = static_cast<uint32_t>(mutate_indices[i]);
      }

      return mutate_indices_list;
    };

    // lambda function to set storage types
    auto infer_storage_type = [=](const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int>* in_stypes,
                                std::vector<int>* out_stypes) {
      // TODO(ziyimu): remove this dense enforce check after supporting sparse tensor
      CHECK(mxnet::common::ContainsOnlyStorage(*in_stypes, mxnet::kDefaultStorage))
      << "Error input tensors are not dense for custom operator '" << name_str << "'";
      // set outputs as dense
      return op::storage_type_assign(out_stypes, mxnet::kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
    };

    // FGradient register lambda
    auto grad_reg = [=](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
        // copy gradients first
        std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
        // copy inputs second
        for (auto& h : n->inputs) {
          heads.push_back(h);
        }
        // copy outputs last
        uint32_t n_out = n->num_outputs();
        for (uint32_t i = 0; i < n_out; ++i) {
          heads.emplace_back(n, i, 0);
        }
        std::string grad_name = "_backward_" + name_str;
        return mxnet::op::MakeGradNode(grad_name.c_str(), n, heads, n->attrs.dict);
    };

    auto resc_req = [=](const NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
    };

    // library author should implement and return a 'state' which points to an instance
    // in lambda we create OpStatePtr using the returned 'state'
    auto create_opstate = [=] (const NodeAttrs& attrs,
                                Context ctx,
                                const std::vector<TShape>& in_shapes,
                                const std::vector<int>& in_types) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // convert subgraph symbol from node attributes to char*
      std::string subgraph_json;
      if (!attrs.subgraphs.empty()) {
        nnvm::Graph g;
        g.outputs = attrs.subgraphs[0].get()->outputs;
        subgraph_json = nnvm::pass::SaveJSON(g);
        attr_keys.push_back(SUBGRAPH_SYM_JSON);
        attr_vals.push_back(subgraph_json.c_str());
      }

      // create a pointer to hold custom op state object
      void* state_op_inst = nullptr;
      CHECK(callCreateOpState(create_opstate_fp, attr_keys.data(), attr_vals.data(),
                              attr_keys.size(), &state_op_inst))
      << "Error calling CreateOpState for custom operator '" << name_str << "'";

      CHECK(state_op_inst != nullptr)
      << "Error custom library failed to create stateful operator '" << name_str << "'";

      CustomStatefulOp* state_op = reinterpret_cast<CustomStatefulOp*>(state_op_inst);
      return OpStatePtr::Create<CustomStatefulOpWrapper>(state_op);
    };

    // stateful forward and backward
    auto fstateful_lambda = [=](bool is_forward,
                                const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
      std::vector<void*> in_data, out_data;
      std::vector<const int64_t *> in_shapes, out_shapes;
      std::vector<int> in_dims, out_dims;
      std::vector<int> in_types, out_types;

      // convert input tensors to constituent parts
      for (size_t i = 0; i < inputs.size(); i++) {
        in_data.push_back(inputs[i].data().dptr_);
        in_shapes.push_back(inputs[i].shape().data());
        in_dims.push_back(inputs[i].shape().ndim());
        in_types.push_back(inputs[i].dtype());
      }

      // convert output tensors to constituent parts
      for (size_t i = 0; i < outputs.size(); i++) {
        out_data.push_back(outputs[i].data().dptr_);
        out_shapes.push_back(outputs[i].shape().data());
        out_dims.push_back(outputs[i].shape().ndim());
        out_types.push_back(outputs[i].dtype());
      }

      // get memory resource
      const Resource &resource = ctx.requested[0];
      mshadow::Stream<mxnet::cpu> *cpu_stream = ctx.get_stream<mxnet::cpu>();

      // create lambda that captures stream & resource objects
      // this temp workspace holds memory allocated by custom library via OpResource
      auto cpu_alloc = [&](int size) {
        mshadow::Tensor<mxnet::cpu, 1, char> data =
        resource.get_space_typed<mxnet::cpu, 1, char>(mshadow::Shape1(size), cpu_stream);
        return data.dptr_;
      };

      // create lambda without captures so that we can cast it to function pointer
      // this needs to be a lambda function so that we can do the decltype cast
      typedef decltype(cpu_alloc) alloc_type;
      auto cpu_malloc = [](void* _cpu_alloc, int size) {
        // cast the void* argument to the type for the cpu_alloc lambda function
        alloc_type* cpualloc = static_cast<alloc_type*>(_cpu_alloc);
        // call cpu_alloc to actually allocate memory and get the pointer
        void* ptr = (*cpualloc)(size);
        return ptr;
      };

      // retrieve op state object created from CreateOpState
      CustomStatefulOpWrapper& op = state_ptr.get_state<CustomStatefulOpWrapper>();
      CustomStatefulOp* state_op_inst = op.get_instance();
      CHECK(state_op_inst != nullptr)
      << "Error MXNet cannot load custom stateful operator'" << name_str << "'";

      // call fcompute function
      CHECK(callFStatefulComp(is_forward, state_op_inst, in_shapes.data(), in_dims.data(),
                              in_data.data(), in_types.data(), in_data.size(),
                              out_shapes.data(), out_dims.data(), out_data.data(),
                              out_types.data(), out_data.size(), cpu_malloc, &cpu_alloc))
      << "Error calling FStatefulCompute for custom operator '" << name_str << "'";
    };

    auto fstateful_forward = [=](const OpStatePtr& state_ptr,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
      fstateful_lambda(true, state_ptr, ctx, inputs, req, outputs);
    };

    auto fstateful_backward = [=](const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
      fstateful_lambda(false, state_ptr, ctx, inputs, req, outputs);
    };

    // check if operator is already registered
    const nnvm::Op *regOpPtr = dmlc::Registry<nnvm::Op>::Get()->Find(name);
    nnvm::Op &regOp = dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(name);
    regOp.set_attr_parser(attr_parser);
    regOp.set_num_inputs(num_inputs);
    regOp.set_num_outputs(num_outputs);
    int plevel = 10;
    if (regOpPtr != nullptr) {
      // overwrite registration of existing op with custom op
      regOp.arguments.clear();
      // set attribute with higher plevel (11) to allow re-registering once
      // TODO(samskalicky): enable constant overwriting of registertion multiple times
      plevel++;
    }
    regOp.set_attr<nnvm::FInferType>("FInferType", infer_type, plevel);
    regOp.set_attr<mxnet::FInferShape>("FInferShape", infer_shape, plevel);
    regOp.set_attr<FInferStorageType>("FInferStorageType", infer_storage_type, plevel);
    regOp.set_attr<FResourceRequest>("FResourceRequest", resc_req, plevel);
    // optionally add stateful forward
    if (create_opstate_fp != nullptr) {
      regOp.set_attr<FCreateOpState>("FCreateOpState", create_opstate, plevel);
      regOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                        fstateful_forward, plevel);
    } else {
      regOp.set_attr<FComputeEx>("FComputeEx<cpu>", forward_lambda, plevel);
    }
    // optionally add fmutate inputs if user specified a function
    if (mutate_fp != nullptr)
      regOp.set_attr<nnvm::FMutateInputs>("FMutateInputs", mutate_inputs, plevel);
    // optionally add fgradient if user specified a function
    if (fgrad_fp != nullptr || create_opstate_fp != nullptr) {
      regOp.set_attr<nnvm::FGradient>("FGradient", grad_reg, plevel);
      std::string grad_name = "_backward_" + name_str;
      nnvm::Op &gradOp = dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(grad_name);
      gradOp.set_attr<nnvm::TIsBackward>("TIsBackward", true, plevel);
      gradOp.set_attr_parser(attr_parser);
      gradOp.set_num_inputs(num_inouts);
      gradOp.set_num_outputs(num_inputs);
      gradOp.set_attr<FInferStorageType>("FInferStorageType", infer_storage_type, plevel);
      gradOp.set_attr<FResourceRequest>("FResourceRequest", resc_req, plevel);
      if (create_opstate_fp != nullptr) {
        gradOp.set_attr<bool>("TIsLayerOpBackward", true, plevel);
        gradOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                            fstateful_backward, plevel);
      } else {
        gradOp.set_attr<FComputeEx>("FComputeEx<cpu>", backward_lambda, plevel);
      }
    }
    regOp.add_argument("data", "NDArray[]", "Source inputs");
  }
  API_END();
}

int MXLibInfoFeatures(const struct LibFeature **lib_features, size_t *size) {
  using namespace features;
  API_BEGIN();
  LibInfo* lib_info = LibInfo::getInstance();
  *lib_features = lib_info->getFeatures().data();
  *size = lib_info->getFeatures().size();
  API_END();
}

int MXRandomSeed(int seed) {
  API_BEGIN();
  mxnet::RandomSeed(seed);
  API_END();
}

int MXRandomSeedContext(int seed, int dev_type, int dev_id) {
  API_BEGIN();
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  mxnet::RandomSeed(ctx, seed);
  API_END();
}

int MXNotifyShutdown() {
  API_BEGIN();
  mxnet::op::custom::CustomOperator::Get()->Stop();
  Engine::Get()->NotifyShutdown();
  API_END();
}

int MXSetNumOMPThreads(int thread_num) {
  API_BEGIN();
  omp_set_num_threads(thread_num);
  API_END();
}

int MXEngineSetBulkSize(int bulk_size, int* prev_bulk_size) {
  API_BEGIN();
  *prev_bulk_size = Engine::Get()->set_bulk_size(bulk_size);
  API_END();
}

int MXGetGPUCount(int* out) {
  API_BEGIN();
  *out = Context::GetGPUCount();
  API_END();
}

// Deprecated: use MXGetGPUMemoryInformation64() instead.
int MXGetGPUMemoryInformation(int dev, int *free_mem, int *total_mem) {
  API_BEGIN();
  uint64_t free_mem64 = 0UL;
  uint64_t total_mem64 = 0UL;
  Context::GetGPUMemoryInformation(dev, &free_mem64, &total_mem64);
  *free_mem = static_cast<int>(free_mem64);
  *total_mem = static_cast<int>(total_mem64);
  API_END();
}

int MXGetGPUMemoryInformation64(int dev, uint64_t *free_mem, uint64_t *total_mem) {
  API_BEGIN();
  Context::GetGPUMemoryInformation(dev, free_mem, total_mem);
  API_END();
}

int MXGetVersion(int *out) {
  API_BEGIN();
  *out = static_cast<int>(MXNET_VERSION);
  API_END();
}

#if MXNET_USE_TVM_OP
int MXLoadTVMOp(const char *libpath) {
  API_BEGIN();
  tvm::runtime::TVMOpModule::Get()->Load(libpath);
  API_END();
}

int MXLoadTVMConfig(ConfigSpaces config) {
  API_BEGIN();
  for (int k = 0; k < config.spaces_size; ++k) {
    tvm::runtime::TVMOpConfig& entry = ::dmlc::Registry<tvm::runtime::TVMOpConfig>::Get()
      ->__REGISTER_OR_GET__(std::string(config.spaces_key[k]));
    const ConfigSpace& c = config.spaces_val[k];
    for (int i = 0; i < c.entity_map_size; ++i) {
      entry.add_entity(std::string(c.entity_map_key[i]), c.entity_map_val[i].val);
    }
    for (int i = 0; i < c.space_map_size; ++i) {
      std::string name = std::string(c.space_map_key[i]);
      std::vector<int> entities;
      for (int j = 0; j < c.space_map_val[i].entities_size; ++j) {
        int val = c.space_map_val[i].entities[j].val;
        entities.push_back(val);
      }
      entry.add_space(name, entities);
    }
  }
  API_END();
}

#endif  // MXNET_USE_TVM_OP

int MXNDArrayCreateNone(NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray();
  API_END();
}

template<typename DataType, typename dimtype>
void CreateNDArray(const DataType* shape,
                   dimtype ndim,
                   int dev_type,
                   int dev_id,
                   int delay_alloc,
                   int dtype,
                   NDArrayHandle* out) {
  mxnet::TShape requested_shape = mxnet::TShape(shape, shape + ndim);
  if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
    CHECK_LT(requested_shape.Size(), (int64_t{1} << 31) - 1) <<
              "[CreateNDArray] Size of tensor you are trying to allocate is larger than "
              "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
  }
  *out = new NDArray(requested_shape,
                     Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
                     delay_alloc != 0, dtype);
}

int MXNDArrayCreate(const uint32_t *shape,
                    uint32_t ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(mxnet::TShape(shape, shape + ndim),
                     Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
                     delay_alloc != 0);
  API_END();
}

int MXNDArrayCreateEx64(const int64_t *shape,
                        int ndim,
                        int dev_type,
                        int dev_id,
                        int delay_alloc,
                        int dtype,
                        NDArrayHandle *out) {
  API_BEGIN();
  CreateNDArray<int64_t, int>(shape, ndim, dev_type, dev_id, delay_alloc, dtype, out);
  API_END();
}

int MXNDArrayCreateEx(const uint32_t *shape,
                      uint32_t ndim,
                      int dev_type,
                      int dev_id,
                      int delay_alloc,
                      int dtype,
                      NDArrayHandle *out) {
  API_BEGIN();
  CreateNDArray<uint32_t, uint32_t>(shape, ndim, dev_type, dev_id, delay_alloc, dtype, out);
  API_END();
}

int MXNDArrayCreateSparseEx(int storage_type,
                            const uint32_t *shape,
                            uint32_t ndim,
                            int dev_type,
                            int dev_id,
                            int delay_alloc,
                            int dtype,
                            uint32_t num_aux,
                            int *aux_type,
                            uint32_t *aux_ndims,
                            const uint32_t *aux_shape,
                            NDArrayHandle *out) {
  API_BEGIN();
  std::vector<int> aux_types;
  mxnet::ShapeVector aux_shapes;
  auto shape_start = aux_shape;
  for (size_t i = 0; i < num_aux; i++) {
    // types
    aux_types.push_back(aux_type[i]);
    // shapes
    aux_shapes.emplace_back(shape_start, shape_start + aux_ndims[i]);
    shape_start += aux_ndims[i];
  }
  *out = new NDArray(
      NDArrayStorageType(storage_type),
      mxnet::TShape(shape, shape + ndim),
      Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
      delay_alloc != 0,
      dtype, aux_types, aux_shapes);
  API_END();
}


int MXNDArrayLoadFromRawBytes(const void *buf,
                              size_t size,
                              NDArrayHandle *out) {
  NDArray *ptr = nullptr;
  API_BEGIN();
  dmlc::MemoryFixedSizeStream strm((void*)buf, size); // NOLINT(*)
  ptr = new NDArray();
  if (!ptr->Load(&strm)) {
    throw dmlc::Error("Invalid NDArray serialization format");
  }
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArraySaveRawBytes(NDArrayHandle handle,
                          size_t *out_size,
                          const char **out_buf) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  ret->ret_str.resize(0);
  dmlc::MemoryStringStream strm(&ret->ret_str);
  static_cast<NDArray*>(handle)->Save(&strm);
  *out_size = ret->ret_str.length();
  *out_buf = ret->ret_str.c_str();
  API_END();
}

int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                             const void *data,
                             size_t size) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->SyncCopyFromCPU(data, size);
  API_END();
}

int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                           void *data,
                           size_t size) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->SyncCopyToCPU(data, size);
  API_END();
}

/*!
 * \brief Copy src.data() to dst.data() if i = -1, else dst.aux_data(i) if i >= 0
 * This function blocks. Do not use it in performance critical code.
 * \param handle_dst handle of a dst ndarray whose data/aux_data has been allocated
 * \param handle_src handle of a src ndarray which has default storage type
 * \param i dst data blob indicator
 */
int MXNDArraySyncCopyFromNDArray(NDArrayHandle handle_dst,
                                 const NDArrayHandle handle_src,
                                 const int i) {
  API_BEGIN();
  NDArray* dst = static_cast<NDArray*>(handle_dst);
  NDArray* src = static_cast<NDArray*>(handle_src);
  dst->SyncCopyFromNDArray(*src, -1, i);
  API_END();
}

int MXNDArraySyncCheckFormat(NDArrayHandle handle, const bool full_check) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  arr->SyncCheckFormat(full_check);
  API_END();
}

int MXNDArrayWaitToRead(NDArrayHandle handle) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->WaitToRead();
  API_END();
}

int MXNDArrayWaitToWrite(NDArrayHandle handle) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->WaitToWrite();
  API_END();
}

int MXNDArrayWaitAll() {
  API_BEGIN();
  Engine::Get()->WaitForAll();
  API_END();
}

int MXNDArraySave(const char* fname,
                  uint32_t num_args,
                  NDArrayHandle* args,
                  const char** keys) {
  API_BEGIN();
  std::vector<NDArray> data(num_args);
  std::vector<std::string> names;
  for (uint32_t i = 0; i < num_args; ++i) {
    data[i] = *static_cast<NDArray*>(args[i]);
  }
  if (keys != nullptr) {
    names.resize(num_args);
    for (uint32_t i = 0; i < num_args; ++i) {
      names[i] = keys[i];
    }
  }
  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
    mxnet::NDArray::Save(fo.get(), data, names);
  }
  API_END();
}

int MXNDArrayLoad(const char* fname,
                  uint32_t *out_size,
                  NDArrayHandle** out_arr,
                  uint32_t *out_name_size,
                  const char*** out_names) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_vec_str.clear();
  API_BEGIN();
  std::vector<NDArray> data;
  std::vector<std::string> &names = ret->ret_vec_str;
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
    mxnet::NDArray::Load(fi.get(), &data, &names);
  }
  ret->ret_handles.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = data[i];
    ret->ret_handles[i] = ptr;
  }
  ret->ret_vec_charp.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ret_vec_charp[i] = names[i].c_str();
  }
  *out_size = static_cast<uint32_t>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<uint32_t>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXNDArrayLoadFromBuffer(const void *ndarray_buffer,
                            size_t size,
                            uint32_t *out_size,
                            NDArrayHandle** out_arr,
                            uint32_t *out_name_size,
                            const char*** out_names) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_vec_str.clear();
  API_BEGIN();
  CHECK_NOTNULL(ndarray_buffer);
  std::vector<NDArray> data;
  std::vector<std::string> &names = ret->ret_vec_str;
  {
    std::unique_ptr<dmlc::MemoryFixedSizeStream> fi(new dmlc::MemoryFixedSizeStream(
        const_cast<void*>(ndarray_buffer), size));
    mxnet::NDArray::Load(fi.get(), &data, &names);
  }
  ret->ret_handles.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = data[i];
    ret->ret_handles[i] = ptr;
  }
  ret->ret_vec_charp.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ret_vec_charp[i] = names[i].c_str();
  }
  *out_size = static_cast<uint32_t>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<uint32_t>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXNDArrayFree(NDArrayHandle handle) {
  API_BEGIN();
  delete static_cast<NDArray*>(handle);
  API_END();
}

template<typename dtype>
void SliceArray(NDArrayHandle handle, dtype slice_begin, dtype slice_end, NDArray* ptr,
                NDArrayHandle* out) {
  *ptr = static_cast<NDArray*>(handle)->SliceWithRecord(slice_begin, slice_end);
  *out = ptr;
}

int MXNDArraySlice(NDArrayHandle handle,
                   uint32_t slice_begin,
                   uint32_t slice_end,
                   NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  SliceArray<uint32_t>(handle, slice_begin, slice_end, ptr, out);
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArraySlice64(NDArrayHandle handle,
                     int64_t slice_begin,
                     int64_t slice_end,
                     NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  SliceArray<int64_t>(handle, slice_begin, slice_end, ptr, out);
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt(NDArrayHandle handle,
                uint32_t idx,
                NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->AtWithRecord(idx);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt64(NDArrayHandle handle,
                  int64_t idx,
                  NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->AtWithRecord(idx);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

MXNET_DLL int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *dims,
                               NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  mxnet::TShape new_shape(dims, dims+ndim);
  int size = 1;
  int pos = -1;
  for (int i = 0; i < ndim; ++i) {
    int dim = dims[i];
    if (dim == -1) {
      CHECK_EQ(pos, -1)
        << "Invalid new shape " << new_shape
        << ": more than one dimensions are -1";
      pos = i;
    } else {
      if (dim == 0) {
        CHECK_LT(i, arr->shape().ndim())
          << "Invalid new shape " << new_shape
          << ": 0 dimension exceeds original shape " << arr->shape();
        dim = arr->shape()[i];
      }
      size *= dim;
      new_shape[i] = dim;
    }
  }
  if (pos >= 0) {
    new_shape[pos] = arr->shape().Size() / size;
  }
  *ptr = arr->ReshapeWithRecord(new_shape);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

MXNET_DLL int MXNDArrayReshape64(NDArrayHandle handle,
                                 int ndim,
                                 dim_t *dims,
                                 bool reverse,
                                 NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  mxnet::Tuple<dim_t> shape(dims, dims+ndim);
  mxnet::TShape new_shape = mxnet::op::InferReshapeShape(shape, arr->shape(), reverse);
  *ptr = arr->ReshapeWithRecord(new_shape);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayGetStorageType(NDArrayHandle handle,
                     int *out_storage_type) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_storage_type = arr->storage_type();
  } else {
    *out_storage_type = kUndefinedStorage;
  }
  API_END();
}

int MXNDArrayGetShape(NDArrayHandle handle,
                      uint32_t *out_dim,
                      const uint32_t **out_pdata) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const mxnet::TShape &s = arr->shape();
    *out_dim = s.ndim();
    std::vector<uint32_t>& buffer = ret->arg_shape_buffer;
    buffer.resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer.data());
    *out_pdata = buffer.data();
  } else {
    *out_dim = 0;
  }
  API_END();
}

template<typename dtype>
inline void GetShape(NDArrayHandle handle, const dtype** out_pdata, int* out_dim,
                     MXAPIThreadLocalEntry<dtype>* ret) {
  NDArray* arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      CHECK_LT(arr->shape().Size(), (int64_t{1} << 31) - 1) <<
                      "[Get Shape] Size of tensor you are trying to allocate is larger than "
                      "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
    }
    mxnet::TShape s = arr->shape();
    if (!Imperative::Get()->is_np_shape()) {
      common::ConvertToLegacyShape(&s);
    }
    *out_dim = s.ndim();
    if (s.ndim() >= 0) {
      std::vector<dtype> &buffer = ret->arg_shape_buffer_ex;
      buffer.resize(s.ndim());
      mxnet::ShapeTypeCast(s.begin(), s.end(), buffer.data());
      *out_pdata = buffer.data();
    }
  } else {
    if (Imperative::Get()->is_np_shape()) {
      *out_dim = -1;
    } else {
      *out_dim = 0;
    }
  }
}

int MXNDArrayGetShapeEx(NDArrayHandle handle,
                        int *out_dim,
                        const int **out_pdata) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  GetShape<int>(handle, out_pdata, out_dim, ret);
  API_END();
}

int MXNDArrayGetShapeEx64(NDArrayHandle handle,
                          int *out_dim,
                          const int64_t **out_pdata) {
  MXAPIThreadLocalEntry<int64_t> *ret = MXAPIThreadLocalStore<int64_t>::Get();
  API_BEGIN();
  GetShape<int64_t>(handle, out_pdata, out_dim, ret);
  API_END();
}

int MXNDArrayGetData(NDArrayHandle handle,
                     void **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
#if MXNET_USE_MKLDNN == 1
  if (arr->IsMKLDNNData()) {
    arr->Reorder2DefaultAsync();
    arr->WaitToRead();
  }
#endif
  if (!arr->is_none()) {
    *out_pdata = arr->data().dptr_;
  } else {
    *out_pdata = nullptr;
  }
  API_END();
}

int MXNDArrayToDLPack(NDArrayHandle handle,
                      DLManagedTensorHandle *out_dlpack) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out_dlpack = arr->ToDLPack();
  API_END();
}

int MXNDArrayFromDLPack(DLManagedTensorHandle dlpack,
                        NDArrayHandle *out_handle) {
  return MXNDArrayFromDLPackEx(dlpack, false, out_handle);
}

int MXNDArrayFromDLPackEx(DLManagedTensorHandle dlpack,
                          const bool transient_handle,
                          NDArrayHandle *out_handle) {
  API_BEGIN();
  *out_handle = new NDArray(NDArray::FromDLPack(
              static_cast<DLManagedTensor*>(dlpack),
              transient_handle));
  API_END();
}

int MXNDArrayCallDLPackDeleter(DLManagedTensorHandle dlpack) {
  API_BEGIN();
  if (dlpack != nullptr) {
    DLManagedTensor *p_dlpack = static_cast<DLManagedTensor*>(dlpack);
    p_dlpack->deleter(p_dlpack);
  }
  API_END();
}

int MXNDArrayGetDType(NDArrayHandle handle,
                     int *out_dtype) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_dtype = arr->dtype();
  } else {
    *out_dtype = -1;
  }
  API_END();
}

int MXNDArrayGetAuxType(NDArrayHandle handle,
                        uint32_t i,
                        int *out_type) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out_type = arr->aux_type(i);
  API_END();
}

/*!
 * \brief Get a deep copy of the ith aux data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
int MXNDArrayGetAuxNDArray(NDArrayHandle handle,
                           uint32_t i,
                           NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->aux_ndarray(i));
  API_END();
}

/*!
 * \brief Get a deep copy of the data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
int MXNDArrayGetDataNDArray(NDArrayHandle handle,
                            NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->data_ndarray());
  API_END();
}

int MXNDArrayGetContext(NDArrayHandle handle,
                        int *out_dev_type,
                        int *out_dev_id) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const Context &ctx = arr->ctx();
    *out_dev_type = ctx.dev_type;
    *out_dev_id = ctx.dev_id;
  } else {
    *out_dev_type = 0;
    *out_dev_id = 0;
  }
  API_END();
}


int MXNDArrayGetGrad(NDArrayHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  NDArray ret = arr->grad();
  if (ret.is_none()) {
    *out = NULL;
  } else {
    *out = new NDArray(ret);
  }
  API_END();
}

int MXNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->Detach());
  API_END();
}

int MXNDArraySetGradState(NDArrayHandle handle, int state) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  arr->set_fresh_out_grad(static_cast<bool>(state));
  API_END();
}

int MXNDArrayGetGradState(NDArrayHandle handle, int *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = arr->fresh_out_grad();
  API_END();
}

int MXListFunctions(uint32_t *out_size,
                    FunctionHandle **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<NDArrayFunctionReg>::List();
  *out_size = static_cast<uint32_t>(vec.size());
  *out_array = (FunctionHandle*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXGetFunction(const char *name,
                  FunctionHandle *out) {
  API_BEGIN();
  *out = dmlc::Registry<NDArrayFunctionReg>::Find(name);
  API_END();
}

int MXFuncGetInfo(FunctionHandle fun,
                  const char **name,
                  const char **description,
                  uint32_t *num_args,
                  const char ***arg_names,
                  const char ***arg_type_infos,
                  const char ***arg_descriptions,
                  const char **return_type) {
  return MXAPIGetFunctionRegInfo(static_cast<const NDArrayFunctionReg *>(fun),
                                 name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 return_type);
}

int MXFuncDescribe(FunctionHandle fun,
                   uint32_t *num_use_vars,
                   uint32_t *num_scalars,
                   uint32_t *num_mutate_vars,
                   int *type_mask) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  *num_use_vars = f->num_use_vars;
  *num_scalars = f->num_scalars;
  *num_mutate_vars = f->num_mutate_vars;
  *type_mask = f->type_mask;
  API_END();
}

int MXFuncInvoke(FunctionHandle fun,
                 NDArrayHandle *use_vars,
                 float *scalar_args,
                 NDArrayHandle *mutate_vars) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  f->body((NDArray**)(use_vars),  //  NOLINT(*)
          scalar_args,
          (NDArray**)(mutate_vars),  //  NOLINT(*)
          0,
          NULL,
          NULL);
  API_END();
}

int MXFuncInvokeEx(FunctionHandle fun,
                 NDArrayHandle *use_vars,
                 float *scalar_args,
                 NDArrayHandle *mutate_vars,
                 int num_params,
                 char **param_keys,
                 char **param_vals) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  f->body((NDArray**)(use_vars),  //  NOLINT(*)
          scalar_args,
          (NDArray**)(mutate_vars),  //  NOLINT(*)
          num_params,
          param_keys,
          param_vals);
  API_END();
}

//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
int MXListDataIters(uint32_t *out_size,
                    DataIterCreator **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<DataIteratorReg>::List();
  *out_size = static_cast<uint32_t>(vec.size());
  *out_array = (DataIterCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXDataIterGetIterInfo(DataIterCreator creator,
                          const char **name,
                          const char **description,
                          uint32_t *num_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions) {
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  return MXAPIGetFunctionRegInfo(e, name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 NULL);
}

int MXDataIterCreateIter(DataIterCreator creator,
                         uint32_t num_param,
                         const char **keys,
                         const char **vals,
                         DataIterHandle *out) {
  IIterator<DataBatch> *iter = nullptr;
  API_BEGIN();
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  iter = e->body();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (uint32_t i = 0; i < num_param; ++i) {
    kwargs.push_back({std::string(keys[i]), std::string(vals[i])});
  }
  iter->Init(kwargs);
  *out = iter;
  API_END_HANDLE_ERROR(delete iter);
}

int MXDataIterFree(DataIterHandle handle) {
  API_BEGIN();
  delete static_cast<IIterator<DataBatch> *>(handle);
  API_END();
}

int MXDataIterBeforeFirst(DataIterHandle handle) {
  API_BEGIN();
  static_cast<IIterator<DataBatch>* >(handle)->BeforeFirst();
  API_END();
}

int MXDataIterNext(DataIterHandle handle, int *out) {
  API_BEGIN();
  *out = static_cast<IIterator<DataBatch>* >(handle)->Next();
  API_END();
}

int MXDataIterGetLabel(DataIterHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  NDArray* pndarray = new NDArray();
  // temp hack to make label 1D
  // TODO(tianjun) make label 1D when label_width=0
  mxnet::TShape shape = db.data[1].shape();
  if (shape.ndim() > 1 && shape[1] == 1) {
    *pndarray = db.data[1].Reshape(mshadow::Shape1(shape[0]));
  } else {
    *pndarray = db.data[1];
  }
  *out = pndarray;
  API_END();
}

int MXDataIterGetIndex(DataIterHandle handle, uint64_t **out_index, uint64_t *out_size) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  *out_size = db.index.size();
  *out_index = const_cast<uint64_t*>(db.index.data());
  API_END();
}

int MXDataIterGetData(DataIterHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  NDArray* pndarray = new NDArray();
  *pndarray = db.data[0];
  *out = pndarray;
  API_END();
}

int MXDataIterGetPadNum(DataIterHandle handle, int *pad) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  *pad = db.num_batch_padd;
  API_END();
}

int MXKVStoreCreate(const char *type,
                    KVStoreHandle *out) {
  API_BEGIN();
  *out = KVStore::Create(type);
  API_END();
}

int MXKVStoreSetGradientCompression(KVStoreHandle handle, uint32_t num_params,
                                    const char** keys, const char** vals) {
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > params;
  for (uint32_t i = 0; i < num_params; ++i) {
    std::pair<std::string, std::string> p;
    p.first = keys[i];
    p.second = vals[i];
    params.push_back(p);
  }
  static_cast<KVStore*>(handle)->SetGradientCompression(params);
  API_END();
}

int MXKVStoreFree(KVStoreHandle handle) {
  API_BEGIN();
  delete static_cast<KVStore*>(handle);
  API_END();
}

int MXKVStoreInit(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
  API_END();
}

int MXKVStoreInitEx(KVStoreHandle handle,
                  uint32_t num,
                  const char** keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
  API_END();
}

int MXKVStorePush(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePushEx(KVStoreHandle handle,
                  uint32_t num,
                  const char** keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePull(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, true);
  API_END();
}

int MXKVStorePullEx(KVStoreHandle handle,
                    uint32_t num,
                    const char** keys,
                    NDArrayHandle* vals,
                    int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, true);
  API_END();
}

int MXKVStorePushPull(KVStoreHandle handle,
                      mx_uint vnum,
                      const int* vkeys,
                      mx_uint onum,
                      const int* okeys,
                      NDArrayHandle* vals,
                      NDArrayHandle* outs,
                      int priority) {
  API_BEGIN();
  std::vector<int> v_vkeys(vnum);
  std::vector<int> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->PushPull(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStorePushPullEx(KVStoreHandle handle,
                        mx_uint vnum,
                        const char** vkeys,
                        mx_uint onum,
                        const char** okeys,
                        NDArrayHandle* vals,
                        NDArrayHandle* outs,
                        int priority) {
  API_BEGIN();
  std::vector<std::string> v_vkeys(vnum);
  std::vector<std::string> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->PushPull(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStorePullWithSparse(KVStoreHandle handle,
                            uint32_t num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority,
                            bool ignore_sparse) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, ignore_sparse);
  API_END();
}

int MXKVStorePullWithSparseEx(KVStoreHandle handle,
                              uint32_t num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority,
                              bool ignore_sparse) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, ignore_sparse);
  API_END();
}

int MXKVStorePullRowSparse(KVStoreHandle handle,
                           uint32_t num,
                           const int* keys,
                           NDArrayHandle* vals,
                           const NDArrayHandle* row_ids,
                           int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_val_rowids[i] = std::make_pair(static_cast<NDArray*>(vals[i]),
                                     *static_cast<NDArray*>(row_ids[i]));
  }
  static_cast<KVStore*>(handle)->PullRowSparse(v_keys, v_val_rowids, priority);
  API_END();
}

int MXKVStorePullRowSparseEx(KVStoreHandle handle,
                             uint32_t num,
                             const char** keys,
                             NDArrayHandle* vals,
                             const NDArrayHandle* row_ids,
                             int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_val_rowids[i] = std::make_pair(static_cast<NDArray*>(vals[i]),
                                     *static_cast<NDArray*>(row_ids[i]));
  }
  static_cast<KVStore*>(handle)->PullRowSparse(v_keys, v_val_rowids, priority);
  API_END();
}

void MXKVStoreSetUpdaterImpl(KVStoreHandle handle,
                             MXKVStoreUpdater updater,
                             void* updater_handle) {
  MXKVStoreUpdater * updater_temp = updater;
  void* updater_handle_temp = updater_handle;
  std::function<void(int, const NDArray&, NDArray*)> updt
  = [updater_temp, updater_handle_temp](int key, const NDArray& recv, NDArray* local) {
    NDArray* recv_copy = new NDArray();
    *recv_copy = recv;
    NDArray* local_copy = new NDArray();
    *local_copy = *local;
    updater_temp(key, recv_copy, local_copy, updater_handle_temp);
  };
  static_cast<KVStore*>(handle)->set_updater(updt);
}

int MXKVStoreSetUpdater(KVStoreHandle handle,
                        MXKVStoreUpdater updater,
                        void* updater_handle) {
  API_BEGIN();
  MXKVStoreSetUpdaterImpl(handle, updater, updater_handle);
  API_END();
}

int MXKVStoreSetUpdaterEx(KVStoreHandle handle,
                          MXKVStoreUpdater updater,
                          MXKVStoreStrUpdater str_updater,
                          void* updater_handle) {
  API_BEGIN();
  // set updater with int keys
  MXKVStoreSetUpdaterImpl(handle, updater, updater_handle);
  // set updater with string keys
  MXKVStoreStrUpdater * updater_temp = str_updater;
  void* updater_handle_temp = updater_handle;
  std::function<void(const std::string&, const NDArray&, NDArray*)> updt
  = [updater_temp, updater_handle_temp]
    (const std::string& key, const NDArray& recv, NDArray* local) {
    NDArray* recv_copy = new NDArray();
    *recv_copy = recv;
    NDArray* local_copy = new NDArray();
    *local_copy = *local;
    updater_temp(key.c_str(), recv_copy, local_copy, updater_handle_temp);
  };
  static_cast<KVStore*>(handle)->set_updater(updt);
  API_END();
}

int MXKVStoreGetRank(KVStoreHandle handle, int *rank) {
  API_BEGIN();
  *rank = static_cast<KVStore*>(handle)->get_rank();
  API_END();
}

int MXKVStoreGetGroupSize(KVStoreHandle handle, int *size) {
  API_BEGIN();
  *size = static_cast<KVStore*>(handle)->get_group_size();
  API_END();
}

int MXKVStoreBarrier(KVStoreHandle handle) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->Barrier();
  API_END();
}

int MXKVStoreSetBarrierBeforeExit(KVStoreHandle handle,
                                  const int barrier_before_exit) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->set_barrier_before_exit(barrier_before_exit);
  API_END();
}

int MXInitPSEnv(uint32_t num_vars,
                const char **keys,
                const char **vals) {
  API_BEGIN();
  std::unordered_map<std::string, std::string> kwargs;
  for (uint32_t i = 0; i < num_vars; ++i) {
    kwargs[std::string(keys[i])] = std::string(vals[i]);
  }
  KVStore::InitPSEnv(kwargs);
  API_END();
}

int MXKVStoreIsWorkerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsWorkerNode();
  API_END();
}

int MXKVStoreIsServerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsServerNode();
  API_END();
}

int MXKVStoreIsSchedulerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsSchedulerNode();
  API_END();
}

int MXKVStoreRunServer(KVStoreHandle handle,
                       MXKVStoreServerController controller,
                       void *controller_handle) {
  API_BEGIN();
  MXKVStoreServerController *controller_temp = controller;
  void *controller_handle_temp = controller_handle;
  auto ctrl = [controller_temp, controller_handle_temp](int head, const std::string& body) {
      controller_temp(head, body.c_str(), controller_handle_temp);
  };
  static_cast<KVStore*>(handle)->RunServer(ctrl);
  API_END();
}

int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
                                   int cmd_id,
                                   const char* cmd_body) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->SendCommandToServers(
      cmd_id, std::string(cmd_body));
  API_END();
}

int MXKVStoreGetType(KVStoreHandle handle,
                     const char** type) {
  API_BEGIN();
  *CHECK_NOTNULL(type) = static_cast<KVStore*>(handle)->type().c_str();
  API_END();
}

int MXKVStoreGetNumDeadNode(KVStoreHandle handle,
                            const int node_id,
                            int *number,
                            const int timeout_sec) {
  API_BEGIN();
  *number = static_cast<KVStore*>(handle)->get_num_dead_node(node_id, timeout_sec);
  API_END();
}

struct MXRecordIOContext {
  dmlc::RecordIOWriter *writer;
  dmlc::RecordIOReader *reader;
  dmlc::Stream *stream;
  std::string *read_buff;
};

int MXRecordIOWriterCreate(const char *uri,
                           RecordIOHandle *out) {
  API_BEGIN();
  dmlc::Stream *stream = dmlc::Stream::Create(uri, "w");
  MXRecordIOContext *context = new MXRecordIOContext;
  context->writer = new dmlc::RecordIOWriter(stream);
  context->reader = NULL;
  context->stream = stream;
  context->read_buff = NULL;
  *out = reinterpret_cast<RecordIOHandle>(context);
  API_END();
}

int MXRecordIOWriterFree(RecordIOHandle handle) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  delete context->writer;
  delete context->stream;
  delete context;
  API_END();
}

int MXRecordIOWriterWriteRecord(RecordIOHandle handle,
                                const char *buf, size_t size) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->writer->WriteRecord(reinterpret_cast<const void*>(buf), size);
  API_END();
}

int MXRecordIOWriterTell(RecordIOHandle handle, size_t *pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  *pos = context->writer->Tell();
  API_END();
}

int MXRecordIOReaderCreate(const char *uri,
                           RecordIOHandle *out) {
  API_BEGIN();
  dmlc::Stream *stream = dmlc::Stream::Create(uri, "r");
  MXRecordIOContext *context = new MXRecordIOContext;
  context->reader = new dmlc::RecordIOReader(stream);
  context->writer = NULL;
  context->stream = stream;
  context->read_buff = new std::string();
  *out = reinterpret_cast<RecordIOHandle>(context);
  API_END();
}

int MXRecordIOReaderFree(RecordIOHandle handle) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  delete context->reader;
  delete context->stream;
  delete context->read_buff;
  delete context;
  API_END();
}

int MXRecordIOReaderReadRecord(RecordIOHandle handle,
                              char const **buf, size_t *size) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  if (context->reader->NextRecord(context->read_buff)) {
    *buf = context->read_buff->c_str();
    *size = context->read_buff->size();
  } else {
    *buf = NULL;
    *size = 0;
  }
  API_END();
}

int MXRecordIOReaderSeek(RecordIOHandle handle, size_t pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->reader->Seek(pos);
  API_END();
}

int MXRecordIOReaderTell(RecordIOHandle handle, size_t *pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  *pos = context->reader->Tell();
  API_END();
}

int MXRtcCreate(char* name, uint32_t num_input, uint32_t num_output,
                char** input_names, char** output_names,
                NDArrayHandle* inputs, NDArrayHandle* outputs,
                char* kernel, RtcHandle *out) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXRtcPush(RtcHandle handle, uint32_t num_input, uint32_t num_output,
              NDArrayHandle* inputs, NDArrayHandle* outputs,
              uint32_t gridDimX,
              uint32_t gridDimY,
              uint32_t gridDimZ,
              uint32_t blockDimX,
              uint32_t blockDimY,
              uint32_t blockDimZ) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXRtcFree(RtcHandle handle) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator) {
  API_BEGIN();
  mxnet::op::custom::CustomOperator::Get()->Register(op_type, creator);
  API_END();
}


int MXRtcCudaModuleCreate(const char* source, int num_options,
                          const char** options, int num_exports,
                          const char** exports, CudaModuleHandle *out) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  std::vector<std::string> str_opts;
  for (int i = 0; i < num_options; ++i) str_opts.emplace_back(options[i]);
  std::vector<std::string> str_exports;
  for (int i = 0; i < num_exports; ++i) str_exports.emplace_back(exports[i]);
  *out = new rtc::CudaModule(source, str_opts, str_exports);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaModuleFree(CudaModuleHandle handle) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  delete reinterpret_cast<rtc::CudaModule*>(handle);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelCreate(CudaModuleHandle handle, const char* name, int num_args,
                          int* is_ndarray, int* is_const, int* arg_types,
                          CudaKernelHandle *out) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  auto module = reinterpret_cast<rtc::CudaModule*>(handle);
  std::vector<rtc::CudaModule::ArgType> signature;
  for (int i = 0; i < num_args; ++i) {
    signature.push_back(rtc::CudaModule::ArgType{
        static_cast<bool>(is_ndarray[i]), static_cast<bool>(is_const[i]),
        static_cast<mshadow::TypeFlag>(arg_types[i])});
  }
  auto kernel = module->GetKernel(name, signature);
  *out = new std::shared_ptr<rtc::CudaModule::Kernel>(kernel);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelFree(CudaKernelHandle handle) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  delete reinterpret_cast<std::shared_ptr<rtc::CudaModule::Kernel>*>(handle);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelCall(CudaKernelHandle handle, int dev_id, void** args,
                        uint32_t grid_dim_x, uint32_t grid_dim_y,
                        uint32_t grid_dim_z, uint32_t block_dim_x,
                        uint32_t block_dim_y, uint32_t block_dim_z,
                        uint32_t shared_mem) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  auto kernel = reinterpret_cast<std::shared_ptr<rtc::CudaModule::Kernel>*>(handle);
  const auto& signature = (*kernel)->signature();
  std::vector<dmlc::any> any_args;
  for (size_t i = 0; i < signature.size(); ++i) {
    if (signature[i].is_ndarray) {
      any_args.emplace_back(*static_cast<NDArray*>(args[i]));
    } else {
      MSHADOW_TYPE_SWITCH(signature[i].dtype, DType, {
        any_args.emplace_back(*static_cast<DType*>(args[i]));
      });
    }
  }
  (*kernel)->Launch(Context::GPU(dev_id), any_args, grid_dim_x, grid_dim_y,
                    grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXNDArrayGetSharedMemHandle(NDArrayHandle handle, int* shared_pid, int* shared_id) {
  API_BEGIN();
  NDArray* arr = reinterpret_cast<NDArray*>(handle);
  Storage::Handle shandle;
  if (arr->ctx().dev_type == Context::kCPUShared) {
    arr->WaitToRead();
    shandle = arr->storage_handle();
    Storage::Get()->SharedIncrementRefCount(shandle);
  } else {
    NDArray new_arr(arr->shape(), Context::CPUShared(0), false, arr->dtype());
    CopyFromTo(*arr, new_arr);
    new_arr.WaitToRead();
    shandle = new_arr.storage_handle();
    Storage::Get()->SharedIncrementRefCount(shandle);
  }
  *shared_pid = shandle.shared_pid;
  *shared_id = shandle.shared_id;
  API_END();
}

int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, const uint32_t *shape,
                                 uint32_t ndim, int dtype, NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(shared_pid, shared_id, mxnet::TShape(shape, shape + ndim), dtype);
  API_END();
}

int MXNDArrayCreateFromSharedMemEx(int shared_pid, int shared_id, const int *shape,
                                   int ndim, int dtype, NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(shared_pid, shared_id, mxnet::TShape(shape, shape + ndim), dtype);
  API_END();
}

typedef Engine::VarHandle VarHandle;
typedef Engine::CallbackOnComplete CallbackOnComplete;

void AssertValidNumberVars(int num_const_vars, int num_mutable_vars) {
  CHECK_GE(num_const_vars, 0) << "Non-negative number of const vars expected.";
  CHECK_GE(num_mutable_vars, 0) << "Non-negative number of mutable vars expected.";
}

int MXEnginePushAsync(EngineAsyncFunc async_func, void* func_param,
                      EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                      EngineVarHandle const_vars_handle, int num_const_vars,
                      EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                      EngineFnPropertyHandle prop_handle, int priority,
                      const char* opr_name, bool wait) {
  API_BEGIN();

  auto exec_ctx = *static_cast<const Context*>(ctx_handle);
  auto const_vars = static_cast<VarHandle*>(const_vars_handle);
  auto mutable_vars = static_cast<VarHandle*>(mutable_vars_handle);
  auto prop = FnProperty::kNormal;
  if (prop_handle) {
    prop = *static_cast<const FnProperty*>(prop_handle);
  }

  Engine::AsyncFn exec_fn;
  if (deleter == nullptr) {
    exec_fn = [async_func, func_param](RunContext rctx,
                                       CallbackOnComplete on_complete) {
      async_func(&rctx, &on_complete, func_param);
    };
  } else {
    // Wrap func_param in a shared_ptr with deleter such that deleter
    // will be called when the lambda goes out of scope.
    std::shared_ptr<void> shared_func_param(func_param, deleter);
    exec_fn = [async_func, shared_func_param](RunContext rctx,
                                              CallbackOnComplete on_complete) {
      async_func(&rctx, &on_complete, shared_func_param.get());
    };
  }

  AssertValidNumberVars(num_const_vars, num_mutable_vars);
  std::vector<VarHandle> const_var_vec(const_vars, const_vars + num_const_vars);
  std::vector<VarHandle> mutable_var_vec(mutable_vars, mutable_vars + num_mutable_vars);
  Engine::Get()->PushAsync(exec_fn, exec_ctx, const_var_vec, mutable_var_vec,
                           prop, priority, opr_name, wait);

  API_END();
}

int MXEnginePushSync(EngineSyncFunc sync_func, void* func_param,
                     EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                     EngineVarHandle const_vars_handle, int num_const_vars,
                     EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                     EngineFnPropertyHandle prop_handle, int priority,
                     const char* opr_name) {
  API_BEGIN();

  auto exec_ctx = *static_cast<const Context*>(ctx_handle);
  auto const_vars = static_cast<VarHandle*>(const_vars_handle);
  auto mutable_vars = static_cast<VarHandle*>(mutable_vars_handle);
  auto prop = FnProperty::kNormal;
  if (prop_handle) {
    prop = *static_cast<const FnProperty*>(prop_handle);
  }

  Engine::SyncFn exec_fn;
  if (deleter == nullptr) {
    exec_fn = [sync_func, func_param](RunContext rctx) {
      sync_func(&rctx, func_param);
    };
  } else {
    // Wrap func_param in a shared_ptr with deleter such that deleter
    // will be called when the lambda goes out of scope.
    std::shared_ptr<void> shared_func_param(func_param, deleter);
    exec_fn = [sync_func, shared_func_param](RunContext rctx) {
      sync_func(&rctx, shared_func_param.get());
    };
  }

  AssertValidNumberVars(num_const_vars, num_mutable_vars);
  std::vector<VarHandle> const_var_vec(const_vars, const_vars + num_const_vars);
  std::vector<VarHandle> mutable_var_vec(mutable_vars, mutable_vars + num_mutable_vars);
  Engine::Get()->PushSync(exec_fn, exec_ctx, const_var_vec, mutable_var_vec,
                          prop, priority, opr_name);

  API_END();
}

int MXEnginePushAsyncND(EngineAsyncFunc async_func, void* func_param,
                        EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                        NDArrayHandle* const_nds_handle, int num_const_nds,
                        NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                        EngineFnPropertyHandle prop_handle, int priority,
                        const char* opr_name, bool wait) {
  API_BEGIN();
  NDArray** const_nds = reinterpret_cast<NDArray**>(const_nds_handle);
  NDArray** mutable_nds = reinterpret_cast<NDArray**>(mutable_nds_handle);
  std::vector<VarHandle> const_var_vec(num_const_nds);
  for (int i = 0; i < num_const_nds; ++i) const_var_vec[i] = const_nds[i]->var();
  std::vector<VarHandle> mutable_var_vec(num_mutable_nds);
  for (int i = 0; i < num_mutable_nds; ++i) mutable_var_vec[i] = mutable_nds[i]->var();
  return MXEnginePushAsync(async_func, func_param, deleter, ctx_handle,
                           const_var_vec.data(), num_const_nds,
                           mutable_var_vec.data(), num_mutable_nds,
                           prop_handle, priority, opr_name, wait);
  API_END();
}

int MXEnginePushSyncND(EngineSyncFunc sync_func, void* func_param,
                       EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                       NDArrayHandle* const_nds_handle, int num_const_nds,
                       NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                       EngineFnPropertyHandle prop_handle, int priority,
                       const char* opr_name) {
  API_BEGIN();
  NDArray** const_nds = reinterpret_cast<NDArray**>(const_nds_handle);
  NDArray** mutable_nds = reinterpret_cast<NDArray**>(mutable_nds_handle);
  std::vector<VarHandle> const_var_vec(num_const_nds);
  for (int i = 0; i < num_const_nds; ++i) const_var_vec[i] = const_nds[i]->var();
  std::vector<VarHandle> mutable_var_vec(num_mutable_nds);
  for (int i = 0; i < num_mutable_nds; ++i) mutable_var_vec[i] = mutable_nds[i]->var();
  return MXEnginePushSync(sync_func, func_param, deleter, ctx_handle,
                          const_var_vec.data(), num_const_nds,
                          mutable_var_vec.data(), num_mutable_nds,
                          prop_handle, priority, opr_name);
  API_END();
}

int MXStorageEmptyCache(int dev_type, int dev_id) {
  API_BEGIN();
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  Storage::Get()->ReleaseAll(ctx);
  API_END();
}

int MXShallowCopyNDArray(NDArrayHandle src_handle, NDArrayHandle* out) {
  NDArray* ret = nullptr;
  API_BEGIN();
  NDArray* src_array = static_cast<NDArray*>(src_handle);
  ret = new NDArray(*src_array);
  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}

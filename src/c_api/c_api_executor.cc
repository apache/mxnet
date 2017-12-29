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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_executor.cc
 * \brief C API of mxnet
 */
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/executor.h>
#include "./c_api_common.h"

int MXExecutorPrint(ExecutorHandle handle, const char **out_str) {
  Executor *exec = static_cast<Executor*>(handle);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  exec->Print(os);
  ret->ret_str = os.str();
  *out_str = (ret->ret_str).c_str();
  API_END();
}

int MXExecutorFree(ExecutorHandle handle) {
  API_BEGIN();
  delete static_cast<Executor*>(handle);
  API_END();
}

int MXExecutorForward(ExecutorHandle handle, int is_train) {
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  exec->Forward(is_train != 0);
  API_END();
}

int MXExecutorBackward(ExecutorHandle handle,
                       mx_uint len,
                       NDArrayHandle *head_grads) {
  return MXExecutorBackwardEx(handle, len, head_grads, true);
}

int MXExecutorBackwardEx(ExecutorHandle handle,
                         mx_uint len,
                         NDArrayHandle *head_grads,
                         int is_train) {
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> ndarrays;
  NDArray **args_ptr = reinterpret_cast<NDArray**>(head_grads);
  for (mx_uint i = 0; i < len; ++i) {
    ndarrays.push_back(*args_ptr[i]);
  }
  exec->Backward(ndarrays, is_train);
  API_END();
}

int MXExecutorOutputs(ExecutorHandle handle,
                      mx_uint *out_size,
                      NDArrayHandle **out) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> heads = exec->outputs();
  ret->ret_handles.resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = heads[i];
    ret->ret_handles[i] = ptr;
  }
  *out_size = heads.size();
  *out = dmlc::BeginPtr(ret->ret_handles);
  API_END();
}

int MXExecutorBind(SymbolHandle symbol_handle,
                   int dev_type,
                   int dev_id,
                   mx_uint len,
                   NDArrayHandle *in_args,
                   NDArrayHandle *arg_grad_store,
                   mx_uint *grad_req_type,
                   mx_uint aux_states_len,
                   NDArrayHandle *aux_states,
                   ExecutorHandle *out) {
  return MXExecutorBindX(symbol_handle,
                         dev_type, dev_id,
                         0, nullptr, nullptr, nullptr,
                         len, in_args, arg_grad_store, grad_req_type,
                         aux_states_len, aux_states, out);
}

int MXExecutorBindX(SymbolHandle symbol_handle,
                    int dev_type,
                    int dev_id,
                    mx_uint num_map_keys,
                    const char** map_keys,
                    const int* map_dev_types,
                    const int* map_dev_ids,
                    mx_uint len,
                    NDArrayHandle *in_args,
                    NDArrayHandle *arg_grad_store,
                    mx_uint *grad_req_type,
                    mx_uint aux_states_len,
                    NDArrayHandle *aux_states,
                    ExecutorHandle *out) {
  return MXExecutorBindEX(symbol_handle,
                          dev_type, dev_id,
                          num_map_keys, map_keys, map_dev_types, map_dev_ids,
                          len, in_args, arg_grad_store, grad_req_type,
                          aux_states_len, aux_states,
                          NULL, out);
}

int MXExecutorBindEX(SymbolHandle symbol_handle,
                     int dev_type,
                     int dev_id,
                     mx_uint num_map_keys,
                     const char** map_keys,
                     const int* map_dev_types,
                     const int* map_dev_ids,
                     mx_uint len,
                     NDArrayHandle *in_args,
                     NDArrayHandle *arg_grad_store,
                     mx_uint *grad_req_type,
                     mx_uint aux_states_len,
                     NDArrayHandle *aux_states,
                     ExecutorHandle shared_exec,
                     ExecutorHandle *out) {
  Executor* exec = nullptr;

  API_BEGIN();
  nnvm::Symbol *symb = static_cast<nnvm::Symbol*>(symbol_handle);
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  std::map<std::string, Context> ctx_map;
  for (mx_uint i = 0; i < num_map_keys; ++i) {
    ctx_map[std::string(map_keys[i])] = Context::Create(
        static_cast<Context::DeviceType>(map_dev_types[i]), map_dev_ids[i]);
  }
  NDArray **in_args_ptr = reinterpret_cast<NDArray**>(in_args);
  NDArray **arg_grad_ptr = reinterpret_cast<NDArray**>(arg_grad_store);
  NDArray **aux_states_ptr = reinterpret_cast<NDArray**>(aux_states);
  std::vector<NDArray> in_args_vec;
  std::vector<NDArray> arg_grad_vec;
  std::vector<OpReqType> grad_req_vec;
  std::vector<NDArray> aux_states_vec;
  for (mx_uint i = 0; i < len; ++i) {
    in_args_vec.push_back(*(in_args_ptr[i]));
    if (arg_grad_ptr[i] == nullptr) {
      arg_grad_vec.push_back(NDArray());
      grad_req_vec.push_back(kNullOp);
    } else {
      arg_grad_vec.push_back(*(arg_grad_ptr[i]));
      grad_req_vec.push_back(static_cast<OpReqType>(grad_req_type[i]));
    }
  }
  for (mx_uint i = 0; i < aux_states_len; ++i) {
    aux_states_vec.push_back(*(aux_states_ptr[i]));
  }
  *out = Executor::Bind(*symb, ctx, ctx_map, in_args_vec,
                        arg_grad_vec, grad_req_vec, aux_states_vec,
                        reinterpret_cast<Executor*>(shared_exec));
  API_END_HANDLE_ERROR(delete exec);
}

/*!
 * \brief
 * \param symbol_handle symbol handle
 * \param dev_type default device type
 * \param dev_id default device id
 * \param num_g2c_keys number of group2ctx keys
 * \param g2c_keys key list of group2ctx
 * \param g2c_dev_types device type list of group2ctx
 * \param g2c_dev_ids id list of group2ctx
 * \param provided_grad_req_list_len grad_req length provided by users in front-end
 * \param provided_grad_req_names grad_req names provided by users in front-end
 * \param provided_grad_req_types req types provided by users in front-end
 * \param num_provided_arg_shapes number of user provided in_arg and aux_state shapes
 * \param provided_arg_shape_names name list of provided shapes
 * \param provided_arg_shape_data provided shape data
 * \param provided_arg_shape_idx provided shape data index
 * \param num_provided_arg_dtypes number of user provided in_arg and axu_state dtypes
 * \param provided_arg_dtype_names argument name list of provided dtypes
 * \param provided_arg_dtypes data of provided dtypes
 * \param num_provided_arg_stypes number of user provided in_arg and axu_state storage types
 * \param provided_arg_stype_names argument name list of provided storage types
 * \param provided_arg_stypes data of provided storage types
 * \param num_shared_arg_names number of parameter names passed from _bind_ith_exec
 * \param shared_arg_name_list parameter name list passed from _bind_ith_exec
 * \param shared_buffer_len number of shared data arrays passed from _bind_ith_exec
 * \param shared_buffer_name_list shared data array names passed from _bind_ith_exec
 * \param shared_buffer_handle_list shared data array handles passed from _bind_ith_exec
 * \param updated_shared_buffer_name_list updated shared data array names after binding
 * \param updated_shared_buffer_handle_list updated shared data arrays after binding
 * \param num_in_args number of input arguments of this sym
 * \param in_args list_arguments associated with the current executor
 * \param arg_grads list of gradients of in_args associated with the current executor
 * \param num_aux_states number of aux states of this sym
 * \param aux_states list_auxiliary_states associated with the current executor
 * \param shared_exec_handle shared excutor handle passed from _bind_ith_exec
 * \param out the handle of the executor to be created
 */
int MXExecutorSimpleBind(SymbolHandle symbol_handle,
                         int dev_type,
                         int dev_id,
                         const mx_uint num_g2c_keys,
                         const char** g2c_keys,
                         const int* g2c_dev_types,
                         const int* g2c_dev_ids,
                         const mx_uint provided_grad_req_list_len,
                         const char** provided_grad_req_names,
                         const char** provided_grad_req_types,
                         const mx_uint num_provided_arg_shapes,
                         const char** provided_arg_shape_names,
                         const mx_uint* provided_arg_shape_data,
                         const mx_uint* provided_arg_shape_idx,
                         const mx_uint num_provided_arg_dtypes,
                         const char** provided_arg_dtype_names,
                         const int* provided_arg_dtypes,
                         const mx_uint num_provided_arg_stypes,
                         const char** provided_arg_stype_names,
                         const int* provided_arg_stypes,
                         const mx_uint num_shared_arg_names,
                         const char** shared_arg_name_list,
                         int* shared_buffer_len,
                         const char** shared_buffer_name_list,
                         NDArrayHandle* shared_buffer_handle_list,
                         const char*** updated_shared_buffer_name_list,
                         NDArrayHandle** updated_shared_buffer_handle_list,
                         mx_uint* num_in_args,
                         NDArrayHandle** in_args,
                         NDArrayHandle** arg_grads,
                         mx_uint* num_aux_states,
                         NDArrayHandle** aux_states,
                         ExecutorHandle shared_exec_handle,
                         ExecutorHandle* out) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol*>(symbol_handle);

  // get in_arg names
  std::vector<std::string> in_arg_names = sym->ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  std::vector<std::string> aux_state_names = sym->ListInputNames(nnvm::Symbol::kAuxiliaryStates);

  // attr_dict for setting up type_dict and arg/aux ctx
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> attr_dict;
  if (nullptr == provided_arg_dtypes || nullptr != g2c_keys || nullptr == provided_arg_stypes) {
    std::vector<std::tuple<std::string, std::string, std::string>> attrs =
      sym->ListAttrsRecursive();
    attr_dict.reserve(attrs.size());
    for (const auto& tp : attrs) {
      attr_dict[std::get<0>(tp)][std::get<1>(tp)] = std::get<2>(tp);
    }
  }

  // setup arg_dtype_map
  std::unordered_map<std::string, int> arg_dtype_map;
  if (nullptr == provided_arg_dtypes) {  // use attr_dict
    for (const auto& arg_name : in_arg_names) {
      const auto it = attr_dict.find(arg_name);
      if (it == attr_dict.end() || !it->second.count("__dtype__")) {
        arg_dtype_map[arg_name] = mshadow::kFloat32;
      }
    }
  } else {  // use user input type_dict
    // create dtype map for in_args and aux_states
    arg_dtype_map.reserve(num_provided_arg_dtypes);
    for (mx_uint i = 0; i < num_provided_arg_dtypes; ++i) {
      arg_dtype_map[provided_arg_dtype_names[i]] = provided_arg_dtypes[i];
    }
  }

  // setup arg_stype_map
  std::unordered_map<std::string, int> arg_stype_map;
  if (nullptr == provided_arg_stypes) {  // use attr_dict
    for (const auto& arg_name : in_arg_names) {
      const auto it = attr_dict.find(arg_name);
      if (it == attr_dict.end() || !it->second.count("__storage_type__")) {
        arg_stype_map[arg_name] = kDefaultStorage;
      }
    }
  } else {  // use user input type_dict
    // create stype map for in_args and aux_states
    arg_stype_map.reserve(num_provided_arg_stypes);
    for (mx_uint i = 0; i < num_provided_arg_stypes; ++i) {
      arg_stype_map[provided_arg_stype_names[i]] = provided_arg_stypes[i];
    }
  }

  // create default ctx
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  // create ctx map
  std::map<std::string, Context> ctx_map;
  std::vector<Context> in_arg_ctx_vec(in_arg_names.size(), ctx);
  std::vector<Context> aux_state_ctx_vec(aux_state_names.size(), ctx);
  if (nullptr != g2c_keys) {  // use user input group2ctx dict
    for (mx_uint i = 0; i < num_g2c_keys; ++i) {
      ctx_map[g2c_keys[i]] = Context::Create(
          static_cast<Context::DeviceType>(g2c_dev_types[i]), g2c_dev_ids[i]);
    }

    // initialize in_arg_ctx_vec using group2ctx if there are any
    for (size_t i = 0; i < in_arg_ctx_vec.size(); ++i) {
      const auto it1 = attr_dict.find(in_arg_names[i]);
      if (it1 != attr_dict.end()) {
        const auto it2 = it1->second.find("__ctx_group__");
        if (it2 != it1->second.end()) {
          const auto it3 = ctx_map.find(it2->second);
          if (it3 != ctx_map.end()) {
            in_arg_ctx_vec[i] = it3->second;
          }
        }
      }
    }

    // initialize aux_state_ctx_vec using group2ctx if there are any
    for (size_t i = 0; i < aux_state_ctx_vec.size(); ++i) {
      const auto it1 = attr_dict.find(aux_state_names[i]);
      if (it1 != attr_dict.end()) {
        const auto it2 = it1->second.find("__ctx_group__");
        if (it2 != it1->second.end()) {
          const auto it3 = ctx_map.find(it2->second);
          if (it3 != ctx_map.end()) {
            aux_state_ctx_vec[i] = it3->second;
          }
        }
      }
    }
  }

  // create provided_grad_req_map
  const std::map<std::string, OpReqType> req_map =
    {{"null", kNullOp}, {"write", kWriteTo}, {"add", kAddTo}};
  std::unordered_map<std::string, std::string> provided_grad_req_map;
  std::string grad_req_type;
  if (0 == provided_grad_req_list_len
      && nullptr == provided_grad_req_names
      && nullptr != provided_grad_req_types) {  // string, grad_req='write'
    CHECK_EQ(req_map.count(provided_grad_req_types[0]), 1U)
      << "grad_req=" << provided_grad_req_types[0] << " is not a valid input in simple_bind; "
      "only \'null\', \'write\', and \'add\' are supported";
    grad_req_type = "string";
  } else if (provided_grad_req_list_len > 0
      && nullptr == provided_grad_req_names
      && nullptr != provided_grad_req_types) {  // list, grad_req=['null', 'write']
    grad_req_type = "list";
    CHECK_EQ(provided_grad_req_list_len, in_arg_names.size())
      << "The length of grad_req list does not match the number of input arguments in simple_bind, "
      "expected " << in_arg_names.size() << ", provided " << provided_grad_req_list_len;
  } else if (provided_grad_req_list_len > 0
      && nullptr != provided_grad_req_names
      && nullptr != provided_grad_req_types) {  // dict, grad_req=['lhs': 'null', 'rhs': 'write']
    grad_req_type = "dict";
    provided_grad_req_map.reserve(provided_grad_req_list_len);
    for (mx_uint i = 0; i < provided_grad_req_list_len; ++i) {
      CHECK_EQ(req_map.count(provided_grad_req_types[i]), 1U)
        << "grad_req=" << provided_grad_req_types[i] << " is not a valid input in simple_bind; "
        "only \'null\', \'write\', and \'add\' are supported";
      provided_grad_req_map[provided_grad_req_names[i]] = provided_grad_req_types[i];
    }
  } else {  // grad_req is None
    grad_req_type = "none";
  }

  // initialize arg_grad_ctx_vec and grad_req_type_vec
  std::vector<Context> arg_grad_ctx_vec(in_arg_names.size(), ctx);
  std::vector<OpReqType> grad_req_type_vec(in_arg_names.size(), kNullOp);
  if ("none" != grad_req_type) {
    for (size_t i = 0; i < in_arg_names.size(); ++i) {
      OpReqType cur_req = kNullOp;
      if ("string" == grad_req_type) {
        cur_req = req_map.at(provided_grad_req_types[0]);
      } else if ("list" == grad_req_type) {
        CHECK_EQ(req_map.count(provided_grad_req_types[i]), 1U)
          << "grad_req=" << provided_grad_req_types[i] << " is not a valid input in simple_bind; "
          "only \'null\', \'write\', and \'add\' are supported";
        cur_req = req_map.at(provided_grad_req_types[i]);
      } else if ("dict" == grad_req_type) {
        const auto it = provided_grad_req_map.find(in_arg_names[i]);
        if (it != provided_grad_req_map.end()) {
          cur_req = req_map.at(it->second);
        }
      }
      if (kNullOp != cur_req) {
        arg_grad_ctx_vec[i] = in_arg_ctx_vec[i];
        grad_req_type_vec[i] = static_cast<OpReqType>(cur_req);
      }
    }
  }

  // create shape map for in_args and aux_states
  std::unordered_map<std::string, TShape> arg_shape_map(num_provided_arg_shapes);
  for (mx_uint i = 0; i < num_provided_arg_shapes; ++i) {
    auto p = arg_shape_map.emplace(provided_arg_shape_names[i],
        TShape(provided_arg_shape_data+provided_arg_shape_idx[i],
          provided_arg_shape_data+provided_arg_shape_idx[i+1]));
    CHECK(p.second) << "Duplicate shapes are provided for argument "
      << provided_arg_shape_names[i] << " in simple_bind";
  }

  // create para name set for sharing data array memory
  std::unordered_set<std::string> shared_arg_name_set(num_shared_arg_names);
  for (mx_uint i = 0; i < num_shared_arg_names; ++i) {
    shared_arg_name_set.insert(shared_arg_name_list[i]);
  }

  // create shared_buffer_map
  std::unordered_map<std::string, NDArray> shared_buffer_map;
  bool use_shared_buffer = (*shared_buffer_len >= 0);
  if (*shared_buffer_len > 0) {
    // create shared_buffer_map
    shared_buffer_map.reserve(*shared_buffer_len);
    NDArray** shared_buffer_ptrs =
      reinterpret_cast<NDArray**>(shared_buffer_handle_list);
    for (int i = 0; i < *shared_buffer_len; ++i) {
      shared_buffer_map[shared_buffer_name_list[i]] = *(shared_buffer_ptrs[i]);
    }
  }

  // create temporary place holders for the initialized NDArrays
  // to be passed back to front end
  std::vector<NDArray> in_arg_vec;
  std::vector<NDArray> arg_grad_vec;
  std::vector<NDArray> aux_state_vec;

  *out = Executor::SimpleBind(*sym, ctx, ctx_map, in_arg_ctx_vec, arg_grad_ctx_vec,
                              aux_state_ctx_vec, arg_shape_map, arg_dtype_map, arg_stype_map,
                              grad_req_type_vec, shared_arg_name_set, &in_arg_vec,
                              &arg_grad_vec, &aux_state_vec,
                              use_shared_buffer ? &shared_buffer_map : nullptr,
                              reinterpret_cast<Executor*>(shared_exec_handle));

  // copy ndarray ptrs to ret->handles so that front end
  // can access them
  ret->ret_handles.clear();
  ret->ret_handles.reserve(in_arg_vec.size()+arg_grad_vec.size()+aux_state_vec.size()
                           +shared_buffer_map.size());
  size_t nd_idx = 0;
  for (const auto& nd : in_arg_vec) {
    if (nd.is_none()) {
      LOG(FATAL) << "Input argument NDArray cannot be un-allocated";
    }
    ret->ret_handles.push_back(new NDArray(nd));
  }
  if (in_arg_vec.size() > 0) {
    *num_in_args = in_arg_vec.size();
    *in_args = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  for (const auto& nd : arg_grad_vec) {
    if (nd.is_none()) {
      ret->ret_handles.push_back(nullptr);
    } else {
      ret->ret_handles.push_back(new NDArray(nd));
    }
  }
  if (arg_grad_vec.size() > 0) {
    *arg_grads = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  for (const auto& nd : aux_state_vec) {
    if (nd.is_none()) {
      LOG(FATAL) << "Auxiliary argument NDArray cannot be un-allocated";
    }
    ret->ret_handles.push_back(new NDArray(nd));
  }
  if (aux_state_vec.size() > 0) {
    *num_aux_states = aux_state_vec.size();
    *aux_states = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  if (use_shared_buffer) {
    ret->ret_vec_str.clear();
    ret->ret_vec_str.reserve(shared_buffer_map.size());
    ret->ret_vec_charp.clear();
    ret->ret_vec_charp.reserve(shared_buffer_map.size());
    for (const auto& kv : shared_buffer_map) {
      if (kv.second.is_none()) {
        LOG(FATAL) << "Shared data NDArray cannot be un-allocated";
      }
      ret->ret_handles.push_back(new NDArray(kv.second));
      ret->ret_vec_str.emplace_back(kv.first);
      ret->ret_vec_charp.push_back(ret->ret_vec_str.back().c_str());
    }
    *shared_buffer_len = shared_buffer_map.size();
    *updated_shared_buffer_handle_list = &(ret->ret_handles[nd_idx]);
    *updated_shared_buffer_name_list = &(ret->ret_vec_charp[0]);
  }

  API_END();
}

int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                 ExecutorMonitorCallback callback,
                                 void* callback_handle) {
  API_BEGIN();
  ExecutorMonitorCallback callback_temp = callback;
  void* callback_handle_temp = callback_handle;
  std::function<void(const char*, void*)> clbk
  = [callback_temp, callback_handle_temp](const char *name, void* handle) {
    callback_temp(name, handle, callback_handle_temp);
  };
  Executor *exec = static_cast<Executor*>(handle);
  exec->SetMonitorCallback(clbk);
  API_END();
}

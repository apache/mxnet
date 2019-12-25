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
 * \file c_api_symbolic.cc
 * \brief C API of mxnet
 */
#include "mxnet/base.h"
#include "mxnet/c_api.h"
#include "mxnet/imperative.h"
#include "nnvm/c_api.h"
#include "nnvm/pass.h"
#include "nnvm/pass_functions.h"
#include "nnvm/symbolic.h"
#include "./c_api_common.h"
#include "../common/exec_utils.h"
#include "../operator/operator_common.h"
#include "../executor/exec_pass.h"
#include "../operator/subgraph/subgraph_property.h"

namespace mxnet {
namespace op {
void RegisterLegacyOpProp();
void RegisterLegacyNDFunc();
}
const std::vector<std::string> kHiddenKeys = {
  "ctx_group", "lr_mult", "wd_mult", "force_mirroring", "mirror_stage"
};
const std::vector<std::string> kReplacedHiddenKeys = {
  "__ctx_group__", "__lr_mult__", "__wd_mult__", "__force_mirroring__", "__mirror_stage__"
};
const char *kNamespaceSeparator = "$";


DMLC_JSON_ENABLE_ANY(int, int);

// convert nnvm symbol to a nnvm graph.
nnvm::Graph Symbol2Graph(const nnvm::Symbol &s) {
  nnvm::Graph g;
  g.outputs = s.outputs;
  g.attrs["mxnet_version"] = std::make_shared<nnvm::any>(static_cast<int>(MXNET_VERSION));
  if (Imperative::Get()->is_np_shape()) {
    g.attrs["is_np_shape"] = std::make_shared<nnvm::any>(
        static_cast<int>(Imperative::Get()->is_np_shape()));
  }
  return g;
}

std::vector<uint32_t> ReadOnlyArgIndices(const nnvm::IndexedGraph& idx) {
  std::vector<uint32_t> ret;
  auto& arg_nodes = idx.input_nodes();
  for (uint32_t i = 0; i < arg_nodes.size(); ++i) {
    if (idx.mutable_input_nodes().count(arg_nodes[i]) == 0) {
      ret.push_back(i);
    }
  }
  return ret;
}

}  // namespace mxnet

// symbolic configuration generation API.
// Redirect to NNVM's C API
int MXListAllOpNames(nn_uint *out_size,
                     const char ***out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListAllOpNames(out_size, out_array);
}

int MXSymbolListAtomicSymbolCreators(uint32_t *out_size,
                                     AtomicSymbolCreator **out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListUniqueOps(out_size, out_array);
}

int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                const char **name,
                                const char **description,
                                uint32_t *num_args,
                                const char ***arg_names,
                                const char ***arg_type_infos,
                                const char ***arg_descriptions,
                                const char **key_var_num_args,
                                const char **return_type) {
  static auto& map_key_var_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  const Op* op = static_cast<Op*>(creator);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_str.resize(0);

  if (map_key_var_args.count(op) != 0) {
    *key_var_num_args = map_key_var_args[op].c_str();
  } else {
    *key_var_num_args = ret->ret_str.c_str();
  }
  return NNGetOpInfo(
      creator, name, description,
      num_args, arg_names, arg_type_infos,
      arg_descriptions, return_type);
}

int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                               uint32_t num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  const nnvm::Op* op = static_cast<const nnvm::Op*>(creator);
  std::unordered_map<std::string, std::string> kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    bool flag = false;
    for (const auto &k : kHiddenKeys) {
      std::string tmp(keys[i]);
      size_t pos = tmp.rfind(k);
      if (pos == 0) {
        kwargs.insert({"__" + tmp + "__", std::string(vals[i])});
        flag = true;
        break;
      } else if (pos != std::string::npos && pos == tmp.length() - k.length()) {
        std::ostringstream os;
        os << "setting variable attributes with " << keys[i] << " is deprecated. "
           << "please instead use\nw = Variable(" << k << "=" << vals[i] << ")\n"
           << "sym = YourSymbolName(" << tmp.substr(0, pos-1) << "=w)";
        throw dmlc::Error(os.str());
      }
    }
    if (!flag)
      kwargs.insert({std::string(keys[i]), std::string(vals[i])});
  }
  *s = nnvm::Symbol::CreateFunctor(op, std::move(kwargs));
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}

int MXSymbolCreateVariable(const char *name, SymbolHandle *out) {
  return NNSymbolCreateVariable(name, out);
}

int MXSymbolCreateGroup(uint32_t num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  return NNSymbolCreateGroup(num_symbols, symbols, out);
}

int MXSymbolGetOutput(SymbolHandle symbol,
                      uint32_t index,
                      SymbolHandle *out) {
  return NNSymbolGetOutput(symbol, index, out);
}

int MXSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  *s = static_cast<nnvm::Symbol*>(symbol)->GetInternals();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolGetChildren(SymbolHandle symbol,
                        SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  *s = static_cast<nnvm::Symbol*>(symbol)->GetChildren();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolFree(SymbolHandle symbol) {
  return NNSymbolFree(symbol);
}

int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  return NNSymbolCopy(symbol, out);
}

int MXSymbolPrint(SymbolHandle symbol, const char **out_str) {
  return NNSymbolPrint(symbol, out_str);
}

int MXSymbolGetName(SymbolHandle symbol,
                    const char** out,
                    int* success) {
  return NNSymbolGetAttr(symbol, "name", out, success);
}

int MXSymbolGetAttr(SymbolHandle symbol,
                    const char* key,
                    const char** out,
                    int* success) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  if (s->GetAttr(key, &(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
    if (std::find(kHiddenKeys.begin(), kHiddenKeys.end(), key) != kHiddenKeys.end()) {
      std::string skey = "__" + std::string(key) + "__";
      if (s->GetAttr(skey, &(ret->ret_str))) {
        *out = (ret->ret_str).c_str();
        *success = 1;
      }
    }
  }
  API_END();
}

int MXSymbolSetAttr(SymbolHandle symbol,
                    const char* key,
                    const char* value) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > kwargs;
  std::string skey(key), sval(value);
  for (const auto &k : kHiddenKeys) {
    size_t pos = skey.rfind(k);
    if (pos == 0 && k.length() == skey.length()) {
      skey = "__" + skey + "__";
      break;
    } else if (pos != std::string::npos && pos + k.length() == skey.length()) {
      std::ostringstream os;
      os << "setting variable attributes with " << key << " is deprecated. "
         << "please instead use\nw = Variable(" << k << "=" << value << ")\n"
         << "sym = YourSymbolName(" << skey.substr(0, pos-1) << "=w)";
      throw dmlc::Error(os.str());
    }
  }
  kwargs.emplace_back(std::make_pair(std::move(skey), std::move(sval)));
  s->SetAttrs(kwargs);
  API_END();
}

int MXSymbolListAttr(SymbolHandle symbol,
                     uint32_t *out_size,
                     const char*** out) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  std::vector<std::tuple<std::string, std::string, std::string> > attr =
      s->ListAttrsRecursive();

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.clear();
  for (const auto& tp : attr) {
    attr_list.emplace_back(std::get<0>(tp) + kNamespaceSeparator + std::get<1>(tp));
    attr_list.emplace_back(std::get<2>(tp));
    if (find(kReplacedHiddenKeys.begin(), kReplacedHiddenKeys.end(), std::get<1>(tp))
          != kReplacedHiddenKeys.end()) {
      attr_list.push_back(std::get<0>(tp) + kNamespaceSeparator +
                          std::get<1>(tp).substr(2, std::get<1>(tp).length() - 4));
      attr_list.push_back(std::get<2>(tp));
    }
  }
  *out_size = attr_list.size()/2;
  ret->ret_vec_charp.clear();
  for (const auto& attr : attr_list) {
    ret->ret_vec_charp.push_back(attr.c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListAttrShallow(SymbolHandle symbol,
                            uint32_t *out_size,
                            const char*** out) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  std::unordered_map<std::string, std::string> attr =
      s->ListAttrs(static_cast<nnvm::Symbol::ListAttrOption>(1));  // NOLINT(*)

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.clear();
  for (const auto& kv : attr) {
    attr_list.push_back(kv.first);
    attr_list.push_back(kv.second);
    if (find(kReplacedHiddenKeys.begin(), kReplacedHiddenKeys.end(), kv.first)
          != kReplacedHiddenKeys.end()) {
      attr_list.push_back(kv.first.substr(2, kv.first.length() - 4));
      attr_list.push_back(kv.second);
    }
  }
  *out_size = attr_list.size()/2;
  ret->ret_vec_charp.clear();
  for (auto &attr : attr_list) {
    ret->ret_vec_charp.push_back(attr.c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListOutputs(SymbolHandle symbol,
                        uint32_t *out_size,
                        const char ***out_str_array) {
  return NNSymbolListOutputNames(symbol, out_size, out_str_array);
}

int MXSymbolGetNumOutputs(SymbolHandle symbol,
                           uint32_t *output_count) {
  return NNSymbolGetNumOutputs(symbol, output_count);
}

int MXSymbolCompose(SymbolHandle sym,
                    const char *name,
                    uint32_t num_args,
                    const char** keys,
                    SymbolHandle* args) {
  return NNSymbolCompose(sym, name, num_args, keys, args);
}

// adapter functions that re-implements the functions.
int MXSymbolListArguments(SymbolHandle symbol,
                          uint32_t *out_size,
                          const char ***out_str_array) {
  return NNSymbolListInputNames(symbol, 1, out_size, out_str_array);
}

int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                uint32_t *out_size,
                                const char ***out_str_array) {
  return NNSymbolListInputNames(symbol, 2, out_size, out_str_array);
}

int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                const char **out) {
  API_BEGIN();
  Op *e = static_cast<Op *>(creator);
  *out = e->name.c_str();
  API_END();
}

namespace mxnet {

extern std::vector<nnvm::Symbol *> GetInputSymbols(const nnvm::Symbol &sym);
extern bool CutGraphInputs(const std::vector<nnvm::NodeEntry *> &input_entries,
                           bool skip_var, std::vector<nnvm::NodeEntry> *orig_entries);

}

int MXSymbolGetInputSymbols(SymbolHandle sym, SymbolHandle **input_arr, int *input_size) {
  API_BEGIN();
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  std::vector<nnvm::Symbol *> input_syms = mxnet::GetInputSymbols(*s);
  *input_size = input_syms.size();

  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_handles.clear();
  ret->ret_handles.reserve(*input_size);
  for (int i = 0; i < *input_size; ++i) ret->ret_handles.push_back(input_syms[i]);
  *input_arr = reinterpret_cast<SymbolHandle*>(dmlc::BeginPtr(ret->ret_handles));
  API_END_HANDLE_ERROR();
}

int MXSymbolCutSubgraph(SymbolHandle sym, SymbolHandle **input_symbols,
                        int *input_size) {
  // Given a graph, we want to fetch the nodes that have been marked as part of
  // a subgraph.
  API_BEGIN();
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  const std::string subg_attr = "__subgraph_name__";
  auto out_node = s->outputs[0].node;
  auto it = out_node->attrs.dict.find(subg_attr);
  if (it != out_node->attrs.dict.end()) {
    const std::string &subg_name = it->second;
    std::vector<nnvm::NodeEntry *> input_entries;
    DFSVisit(s->outputs, [&subg_attr, &subg_name, &input_entries]
             (nnvm::NodePtr n) {
      // If the node itself isn't in the subgraph, we ignore it.
      auto it = n->attrs.dict.find(subg_attr);
      if (it == n->attrs.dict.end() || it->second != subg_name)
        return;

      // We search for nodes whose node entries aren't in the subgraph.
      for (size_t j = 0; j < n->inputs.size(); j++) {
        auto in_node = n->inputs[j].node;
        auto it = in_node->attrs.dict.find(subg_attr);
        if (it == in_node->attrs.dict.end() || it->second != subg_name)
          input_entries.push_back(&n->inputs[j]);
      }
    });

    std::vector<nnvm::NodeEntry> orig_entries;
    CutGraphInputs(input_entries, false, &orig_entries);
    std::vector<nnvm::Symbol *> input_syms(orig_entries.size());
    for (size_t i = 0; i < input_syms.size(); i++) {
      input_syms[i] = new nnvm::Symbol();
      input_syms[i]->outputs.push_back(orig_entries[i]);
    }
    *input_size = input_syms.size();

    MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*input_size);
    for (int i = 0; i < *input_size; ++i) ret->ret_handles.push_back(input_syms[i]);
    *input_symbols = reinterpret_cast<SymbolHandle*>(dmlc::BeginPtr(ret->ret_handles));
  } else {
    *input_size = 0;
  }

  API_END_HANDLE_ERROR();
}


/*!
 * \brief Convert shape attr in graph nodes to comply with NumPy semantics for
 * legacy models (before 1.6.0) if global flag is_np_shape has been turned on,
 * i.e., use -1 to indicate unknown number of dimensions and unknown dimension sizes.
 */
void ConvertShapeAttrToNumPyCompatible(nnvm::Graph* g) {
  if (Imperative::Get()->is_np_shape()
    && (!g->HasAttr("is_np_shape") || !g->GetAttr<int>("is_np_shape"))) {
    DFSVisit(g->outputs, [](nnvm::NodePtr n) {
      if (n->is_variable()) {
        auto it = n->attrs.dict.find("__shape__");
        if (it != n->attrs.dict.end()) {
          mxnet::TShape shape;
          std::istringstream is(it->second);
          is >> shape;
          common::ConvertToNumpyShape(&shape);
          std::ostringstream os;
          os << shape;
          it->second = os.str();
        }
      }
    });
  }
}

int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  dmlc::istream is(fi.get());
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(
    std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()));
  g = nnvm::ApplyPass(g, "LoadLegacyJSON");
  ConvertShapeAttrToNumPyCompatible(&g);
  s->outputs = g.outputs;
  *out = s;
  is.set_stream(nullptr);
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(std::string(json));
  g = nnvm::ApplyPass(g, "LoadLegacyJSON");
  ConvertShapeAttrToNumPyCompatible(&g);
  s->outputs = g.outputs;
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolRemoveAmpCast(SymbolHandle sym_handle, SymbolHandle* ret_sym_handle) {
  nnvm::Symbol* s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *source = static_cast<nnvm::Symbol*>(sym_handle);
  *s = source->Copy();
  s->outputs = nnvm::ApplyPass(Symbol2Graph(*s), "RemoveAmpCast").outputs;
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  dmlc::ostream os(fo.get());
  os << nnvm::pass::SaveJSON(Symbol2Graph(*s));
  // reset file pointer, force flush
  os.set_stream(nullptr);
  API_END();
}

int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  ret->ret_str = nnvm::pass::SaveJSON(Symbol2Graph(*s));
  *out_json = ret->ret_str.c_str();
  API_END();
}

namespace mxnet {

template<typename AttrType>
void MatchArguments(
    const nnvm::IndexedGraph& idx,
    const std::unordered_map<std::string, AttrType>& known_arg_attrs,
    std::vector<AttrType>* arg_attrs,
    const char* source) {
  auto& arg_nodes = idx.input_nodes();
  CHECK_EQ(arg_attrs->size(), arg_nodes.size());
  size_t nmatched = 0;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    const std::string& name = idx[arg_nodes[i]].source->attrs.name;
    auto it = known_arg_attrs.find(name);
    if (it != known_arg_attrs.end()) {
      arg_attrs->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_attrs.size()) {
    std::unordered_set<std::string> keys;
    std::ostringstream head, msg;
    msg << "\nCandidate arguments:\n";
    for (size_t i = 0; i < arg_nodes.size(); ++i) {
      std::string arg_name = idx[arg_nodes[i]].source->attrs.name;
      keys.insert(arg_name);
      msg << "\t[" << i << ']' << arg_name << '\n';
    }
    for (const auto& kv : known_arg_attrs) {
      const std::string& key = kv.first;
      if (keys.count(key) == 0) {
        LOG(FATAL) << source
                   << "Keyword argument name " << key << " not found."
                   << msg.str();
      }
    }
  }
}

}  // namespace mxnet

int MXSymbolInferShape(SymbolHandle sym,
                       uint32_t num_args,
                       const char** keys,
                       const uint32_t *arg_ind_ptr,
                       const uint32_t *arg_shape_data,
                       uint32_t *in_shape_size,
                       const uint32_t **in_shape_ndim,
                       const uint32_t ***in_shape_data,
                       uint32_t *out_shape_size,
                       const uint32_t **out_shape_ndim,
                       const uint32_t ***out_shape_data,
                       uint32_t *aux_shape_size,
                       const uint32_t **aux_shape_ndim,
                       const uint32_t ***aux_shape_data,
                       int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  mxnet::ShapeVector arg_shapes(g.indexed_graph().input_nodes().size(), mxnet::TShape());
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (uint32_t i = 0; i < num_args; ++i) {
      arg_shapes[read_only_args[i]] = mxnet::ShapeTypeCast(
          arg_shape_data + arg_ind_ptr[i], arg_shape_data + arg_ind_ptr[i+1]);
    }
  } else {
    std::unordered_map<std::string, mxnet::TShape> kwargs;
    for (uint32_t i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = mxnet::ShapeTypeCast(
          arg_shape_data + arg_ind_ptr[i], arg_shape_data + arg_ind_ptr[i+1]);
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_shapes, "InferShape");
  }

  try {
    g = mxnet::exec::InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  } catch (const mxnet::op::InferShapeError &err) {
    throw dmlc::Error(err.msg);
  }

  // if use legacy shape definition, need to convert numpy shape to legacy shape
  mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  if (!Imperative::Get()->is_np_shape()) {
    common::ConvertToLegacyShape(&shapes);
  }

  // copy back
  CopyAttr(g.indexed_graph(), shapes,
           &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));

  // copy data back
  MXAPIThreadLocalEntry<>::SetupShapeArrayReturnWithBuffer(ret->arg_shapes,
      &(ret->arg_shape_ndim), &(ret->arg_shape_data), &(ret->arg_shape_buffer));
  MXAPIThreadLocalEntry<>::SetupShapeArrayReturnWithBuffer(ret->out_shapes,
      &(ret->out_shape_ndim), &(ret->out_shape_data), &(ret->out_shape_buffer));
  MXAPIThreadLocalEntry<>::SetupShapeArrayReturnWithBuffer(ret->aux_shapes,
      &(ret->aux_shape_ndim), &(ret->aux_shape_data), &(ret->aux_shape_buffer));
  *in_shape_size = static_cast<uint32_t>(ret->arg_shapes.size());
  *in_shape_ndim = dmlc::BeginPtr(ret->arg_shape_ndim);
  *in_shape_data = dmlc::BeginPtr(ret->arg_shape_data);
  *out_shape_size = static_cast<uint32_t>(ret->out_shapes.size());
  *out_shape_ndim = dmlc::BeginPtr(ret->out_shape_ndim);
  *out_shape_data = dmlc::BeginPtr(ret->out_shape_data);
  *aux_shape_size = static_cast<uint32_t>(ret->aux_shapes.size());
  *aux_shape_ndim = dmlc::BeginPtr(ret->aux_shape_ndim);
  *aux_shape_data = dmlc::BeginPtr(ret->aux_shape_data);
  // mark complete
  *complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
  API_END();
}

template<typename dtype, typename stype, typename itype>
inline void SymbolInferShape(const char** keys,
                             uint32_t num_args,
                             const dtype* arg_shape_data,
                             const itype* arg_ind_ptr,
                             const int** in_shape_ndim,
                             const dtype*** in_shape_data,
                             const int** out_shape_ndim,
                             const dtype*** out_shape_data,
                             const int** aux_shape_ndim,
                             const dtype*** aux_shape_data,
                             nnvm::Symbol* s,
                             MXAPIThreadLocalEntry<dtype>* ret,
                             stype* in_shape_size,
                             stype* out_shape_size,
                             stype* aux_shape_size,
                             int* complete) {
  nnvm::Graph g = Symbol2Graph(*s);
  mxnet::ShapeVector arg_shapes(g.indexed_graph().input_nodes().size(), mxnet::TShape());
  if (keys == nullptr && num_args != 0) {
    std::vector < uint32_t > read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (uint32_t i = 0; i < num_args; ++i) {
      arg_shapes[read_only_args[i]] = mxnet::ShapeTypeCast(arg_shape_data + arg_ind_ptr[i],
                                                           arg_shape_data + arg_ind_ptr[i + 1]);
    }
  } else {
    std::unordered_map<std::string, mxnet::TShape> kwargs;
    for (uint32_t i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = mxnet::ShapeTypeCast(arg_shape_data + arg_ind_ptr[i],
                                             arg_shape_data + arg_ind_ptr[i + 1]);
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_shapes, "InferShape");
  }
  try {
    g = mxnet::exec::InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  } catch (const mxnet::op::InferShapeError& err) {
    throw dmlc::Error(err.msg);
  }
  // if use legacy shape definition, need to convert numpy shape to legacy shape
  mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  if (!Imperative::Get()->is_np_shape()) {
    common::ConvertToLegacyShape(&shapes);
  }
  // copy back
  CopyAttr(g.indexed_graph(), shapes, &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));
  // copy data back
  MXAPIThreadLocalEntry<dtype>::SetupShapeArrayReturnWithBufferEx(ret->arg_shapes,
                                                                  &(ret->arg_shape_ndim_ex),
                                                                  &(ret->arg_shape_data_ex),
                                                                  &(ret->arg_shape_buffer_ex));
  MXAPIThreadLocalEntry<dtype>::SetupShapeArrayReturnWithBufferEx(ret->out_shapes,
                                                                  &(ret->out_shape_ndim_ex),
                                                                  &(ret->out_shape_data_ex),
                                                                  &(ret->out_shape_buffer_ex));
  MXAPIThreadLocalEntry<dtype>::SetupShapeArrayReturnWithBufferEx(ret->aux_shapes,
                                                                  &(ret->aux_shape_ndim_ex),
                                                                  &(ret->aux_shape_data_ex),
                                                                  &(ret->aux_shape_buffer_ex));
  *in_shape_size = static_cast<stype>(ret->arg_shapes.size());
  *in_shape_ndim = dmlc::BeginPtr(ret->arg_shape_ndim_ex);
  *in_shape_data = dmlc::BeginPtr(ret->arg_shape_data_ex);
  *out_shape_size = static_cast<stype>(ret->out_shapes.size());
  *out_shape_ndim = dmlc::BeginPtr(ret->out_shape_ndim_ex);
  *out_shape_data = dmlc::BeginPtr(ret->out_shape_data_ex);
  *aux_shape_size = static_cast<stype>(ret->aux_shapes.size());
  *aux_shape_ndim = dmlc::BeginPtr(ret->aux_shape_ndim_ex);
  *aux_shape_data = dmlc::BeginPtr(ret->aux_shape_data_ex);
  // mark complete
  *complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
}

int MXSymbolInferShapeEx(SymbolHandle sym,
                         uint32_t num_args,
                         const char** keys,
                         const uint32_t *arg_ind_ptr,
                         const int *arg_shape_data,
                         uint32_t *in_shape_size,
                         const int **in_shape_ndim,
                         const int ***in_shape_data,
                         uint32_t *out_shape_size,
                         const int **out_shape_ndim,
                         const int ***out_shape_data,
                         uint32_t *aux_shape_size,
                         const int **aux_shape_ndim,
                         const int ***aux_shape_data,
                         int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  SymbolInferShape<int, uint32_t, uint32_t>(keys,
                                              num_args,
                                              arg_shape_data,
                                              arg_ind_ptr,
                                              in_shape_ndim,
                                              in_shape_data,
                                              out_shape_ndim,
                                              out_shape_data,
                                              aux_shape_ndim,
                                              aux_shape_data,
                                              s,
                                              ret,
                                              in_shape_size,
                                              out_shape_size,
                                              aux_shape_size,
                                              complete);
  API_END();
}

int MXSymbolInferShapeEx64(SymbolHandle sym,
                           uint32_t num_args,
                           const char** keys,
                           const int64_t *arg_ind_ptr,
                           const int64_t *arg_shape_data,
                           size_t *in_shape_size,
                           const int **in_shape_ndim,
                           const int64_t ***in_shape_data,
                           size_t *out_shape_size,
                           const int **out_shape_ndim,
                           const int64_t ***out_shape_data,
                           size_t *aux_shape_size,
                           const int **aux_shape_ndim,
                           const int64_t ***aux_shape_data,
                           int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry<int64_t> *ret = MXAPIThreadLocalStore<int64_t>::Get();
  API_BEGIN();
  SymbolInferShape<int64_t, size_t, int64_t>(keys,
                                                 num_args,
                                                 arg_shape_data,
                                                 arg_ind_ptr,
                                                 in_shape_ndim,
                                                 in_shape_data,
                                                 out_shape_ndim,
                                                 out_shape_data,
                                                 aux_shape_ndim,
                                                 aux_shape_data,
                                                 s,
                                                 ret,
                                                 in_shape_size,
                                                 out_shape_size,
                                                 aux_shape_size,
                                                 complete);
  API_END();
}

int MXSymbolInferShapePartial(SymbolHandle sym,
                              uint32_t num_args,
                              const char** keys,
                              const uint32_t *arg_ind_ptr,
                              const uint32_t *arg_shape_data,
                              uint32_t *in_shape_size,
                              const uint32_t **in_shape_ndim,
                              const uint32_t ***in_shape_data,
                              uint32_t *out_shape_size,
                              const uint32_t **out_shape_ndim,
                              const uint32_t ***out_shape_data,
                              uint32_t *aux_shape_size,
                              const uint32_t **aux_shape_ndim,
                              const uint32_t ***aux_shape_data,
                              int *complete) {
  int succ = 0;
  *complete = 1;
  return MXSymbolInferShape(sym, num_args, keys,
                            arg_ind_ptr, arg_shape_data,
                            in_shape_size, in_shape_ndim, in_shape_data,
                            out_shape_size, out_shape_ndim, out_shape_data,
                            aux_shape_size, aux_shape_ndim, aux_shape_data,
                            &succ);
}

int MXSymbolInferShapePartialEx(SymbolHandle sym,
                                uint32_t num_args,
                                const char** keys,
                                const uint32_t *arg_ind_ptr,
                                const int *arg_shape_data,
                                uint32_t *in_shape_size,
                                const int **in_shape_ndim,
                                const int ***in_shape_data,
                                uint32_t *out_shape_size,
                                const int **out_shape_ndim,
                                const int ***out_shape_data,
                                uint32_t *aux_shape_size,
                                const int **aux_shape_ndim,
                                const int ***aux_shape_data,
                                int *complete) {
  int succ = 0;
  *complete = 1;
  return MXSymbolInferShapeEx(sym, num_args, keys,
                              arg_ind_ptr, arg_shape_data,
                              in_shape_size, in_shape_ndim, in_shape_data,
                              out_shape_size, out_shape_ndim, out_shape_data,
                              aux_shape_size, aux_shape_ndim, aux_shape_data,
                              &succ);
}

int MXSymbolInferShapePartialEx64(SymbolHandle sym,
                                  uint32_t num_args,
                                  const char** keys,
                                  const int64_t *arg_ind_ptr,
                                  const int64_t *arg_shape_data,
                                  size_t *in_shape_size,
                                  const int **in_shape_ndim,
                                  const int64_t ***in_shape_data,
                                  size_t *out_shape_size,
                                  const int **out_shape_ndim,
                                  const int64_t ***out_shape_data,
                                  size_t *aux_shape_size,
                                  const int **aux_shape_ndim,
                                  const int64_t ***aux_shape_data,
                                  int *complete) {
  int succ = 0;
  *complete = 1;
  return MXSymbolInferShapeEx64(sym, num_args, keys,
                                arg_ind_ptr, arg_shape_data,
                                in_shape_size, in_shape_ndim, in_shape_data,
                                out_shape_size, out_shape_ndim, out_shape_data,
                                aux_shape_size, aux_shape_ndim, aux_shape_data,
                                &succ);
}

int MXSymbolInferType(SymbolHandle sym,
                      uint32_t num_args,
                      const char** keys,
                      const int *arg_type_data,
                      uint32_t *in_type_size,
                      const int **in_type_data,
                      uint32_t *out_type_size,
                      const int **out_type_data,
                      uint32_t *aux_type_size,
                      const int **aux_type_data,
                      int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  nnvm::DTypeVector arg_types(g.indexed_graph().input_nodes().size(), -1);
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (uint32_t i = 0; i < num_args; ++i) {
      arg_types[read_only_args[i]] = arg_type_data[i];
    }
  } else {
    std::unordered_map<std::string, int> kwargs;
    for (uint32_t i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = arg_type_data[i];
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_types, "InferType");
  }

  g = mxnet::exec::InferType(std::move(g), std::move(arg_types), "__dtype__");
  // copy back
  CopyAttr(g.indexed_graph(), g.GetAttr<nnvm::DTypeVector>("dtype"),
           &(ret->arg_types), &(ret->out_types), &(ret->aux_types));

  *in_type_size = static_cast<uint32_t>(ret->arg_types.size());
  *in_type_data = dmlc::BeginPtr(ret->arg_types);
  *out_type_size = static_cast<uint32_t>(ret->out_types.size());
  *out_type_data = dmlc::BeginPtr(ret->out_types);
  *aux_type_size = static_cast<uint32_t>(ret->aux_types.size());
  *aux_type_data = dmlc::BeginPtr(ret->aux_types);
  *complete = (g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0);
  API_END();
}

int MXSymbolInferTypePartial(SymbolHandle sym,
                             uint32_t num_args,
                             const char** keys,
                             const int *arg_type_data,
                             uint32_t *in_type_size,
                             const int **in_type_data,
                             uint32_t *out_type_size,
                             const int **out_type_data,
                             uint32_t *aux_type_size,
                             const int **aux_type_data,
                             int *complete) {
  int succ = 0;
  *complete = 1;
  return MXSymbolInferType(sym, num_args, keys,
                            arg_type_data,
                            in_type_size, in_type_data,
                            out_type_size, out_type_data,
                            aux_type_size, aux_type_data,
                            &succ);
}

int MXSymbolGrad(SymbolHandle sym, uint32_t num_wrt, const char** wrt, SymbolHandle* out) {
  API_BEGIN();
  LOG(FATAL) << "not implemented";
  API_END();
}

int MXQuantizeSymbol(SymbolHandle sym_handle,
                     SymbolHandle *ret_sym_handle,
                     const int* dev_type,
                     const uint32_t num_excluded_sym_names,
                     const char **excluded_sym_names,
                     const uint32_t num_excluded_op_names,
                     const char **excluded_op_names,
                     const uint32_t num_offline,
                     const char **offline_params,
                     const char *quantized_dtype,
                     const bool calib_quantize,
                     const char *quantize_mode,
                     mx_uint* out_num_calib_names,
                     const char ***out_calib_names) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol*>(sym_handle);
  nnvm::Graph g = Symbol2Graph(*sym);
  int target_dev = *dev_type;
  std::unordered_set<std::string> excluded_node_names;
  for (size_t i = 0; i < num_excluded_sym_names; ++i) {
    excluded_node_names.emplace(excluded_sym_names[i]);
  }
  std::unordered_set<std::string> excluded_op;
  for (size_t i = 0; i < num_excluded_op_names; ++i) {
    excluded_op.emplace(excluded_op_names[i]);
  }
  std::unordered_set<std::string> offline;
  for (size_t i = 0; i < num_offline; ++i) {
    offline.emplace(offline_params[i]);
  }
  std::string quantized_type(quantized_dtype);
  std::string quantized_mode(quantize_mode);
  g.attrs["excluded_nodes"] = std::make_shared<nnvm::any>(std::move(excluded_node_names));
  g.attrs["excluded_ops"] = std::make_shared<nnvm::any>(std::move(excluded_op));
  g.attrs["offline_params"] = std::make_shared<nnvm::any>(std::move(offline));
  g.attrs["quantized_dtype"] = std::make_shared<nnvm::any>(std::move(quantized_type));
  g.attrs["target_ctx"] = std::make_shared<nnvm::any>(target_dev);
  g.attrs["quantize_mode"] = std::make_shared<nnvm::any>(std::move(quantized_mode));
  g = ApplyPass(std::move(g), "QuantizeGraph");
  const auto& calib_nodes = g.GetAttr<std::vector<std::string>>("calib_nodes");
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_vec_str = std::move(calib_nodes);
  *out_num_calib_names = ret->ret_vec_str.size();
  ret->ret_vec_charp.clear();
  ret->ret_vec_charp.reserve(ret->ret_vec_str.size());
  for (const auto &str : ret->ret_vec_str) {
    ret->ret_vec_charp.push_back(str.c_str());
  }
  *out_calib_names = dmlc::BeginPtr(ret->ret_vec_charp);
  s->outputs = g.outputs;
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

// helper function to add mapping of node_name -> dtype map
// for the given indexed graph and inferred_dtypes
static void _SetInputDTypes(
    const nnvm::IndexedGraph& idx,
    const nnvm::DTypeVector& inferred_dtypes,
    std::unordered_map<std::string, int>* node_name_dtype_map,
    std::unordered_map<std::string, int>* node_without_dtype_map) {
  const std::string dtype_keyword = "__dtype__";
  for (uint32_t nid : idx.input_nodes()) {
    const auto& node = idx[nid].source;
    const auto& node_with_dtype = node->attrs.dict.find(dtype_keyword);
    // input nodes classified into nodes_with_dtype, nodes_without_dtype
    // This classification required because if param_names not provided
    // we want to update dtypes of only those nodes which have dtypes set
    // inferred_dtypes are obtained for the nodes, if unknown
    // dtype is set to fp32
    if (node_with_dtype != node->attrs.dict.end()) {
      if (inferred_dtypes[idx.entry_id(nid, 0)] == -1) {
        (*node_name_dtype_map)[node->attrs.name] = 0;
      } else {
        (*node_name_dtype_map)[node->attrs.name] =
            inferred_dtypes[idx.entry_id(nid, 0)];
      }
    } else {
      if (inferred_dtypes[idx.entry_id(nid, 0)] == -1) {
        (*node_without_dtype_map)[node->attrs.name] = 0;
      } else {
        (*node_without_dtype_map)[node->attrs.name] =
            inferred_dtypes[idx.entry_id(nid, 0)];
      }
    }
  }
}

// helper function update the node dtype attrs for a vector of nodeptrs
// given the node name to dtype information and the names of model_params
// if model_params is provided the function will dtype of only model params.
// if model_params is empty, the function will dtype of all nodes which had
// a prior dtype set.
// args is a const_reference vector of NodePtrs. NodePtrs are immutable but
// the Nodes they are pointing will be mutated in this function
static void _UpdateSymDTypeAttrs(
    const std::unordered_map<std::string, int>& node_name_dtype_map,
    const std::unordered_map<std::string, int>& node_without_dtype_map,
    const std::unordered_set<std::string>& model_params,
    const std::vector<nnvm::NodePtr>& args) {
  const std::string dtype_keyword = "__dtype__";

  // Update args to have the right dtype attrs
  if (model_params.size() > 0) {
    // if model params provided, set dtype only for model params
    for (size_t i = 0; i < args.size(); ++i) {
      const std::string& node_name = args[i]->attrs.name;
      auto it_model_params = model_params.find(node_name);
      auto it_with_dtype = node_name_dtype_map.find(node_name);
      auto it_without_dtype = node_without_dtype_map.find(node_name);
      if (it_model_params != model_params.end()) {
        // need to update __dtype__ attribute if already set, else set it
        if (it_with_dtype != node_name_dtype_map.end()) {
          args[i]->attrs.dict[dtype_keyword] =
              std::to_string(it_with_dtype->second);
        } else {
          CHECK(it_without_dtype != node_without_dtype_map.end())
              << "make sure all nodes without dtype have properly been added "
                 "in node_without_dtype_map";
          args[i]->attrs.dict[dtype_keyword] =
              std::to_string(it_without_dtype->second);
        }
      }
    }
  } else {
    // if model params not provided, update __dtype__ for all inputs,
    // which already had it set, don't touch the rest
    for (size_t i = 0; i < args.size(); ++i) {
      auto it = node_name_dtype_map.find(args[i]->attrs.name);
      if (it != node_name_dtype_map.end()) {
        if (args[i]->attrs.dict.find(dtype_keyword) !=
            args[i]->attrs.dict.end()) {
          args[i]->attrs.dict[dtype_keyword] = std::to_string(it->second);
        }
      }
    }
  }
}

int MXReducePrecisionSymbol(SymbolHandle sym_handle,
                            SymbolHandle *ret_sym_handle,
                            uint32_t num_args,
                            const int *arg_type_data,
                            uint32_t num_ind_ptr,
                            const int* ind_ptr,
                            const int* target_dtype,
                            const int cast_optional_params,
                            const uint32_t num_target_dtype_op_names,
                            const uint32_t num_fp32_op_names,
                            const uint32_t num_widest_dtype_op_names,
                            const uint32_t num_conditional_fp32_op_names,
                            const uint32_t num_excluded_symbols,
                            const uint32_t num_model_params,
                            const char **target_dtype_op_names,
                            const char **fp32_op_names,
                            const char **widest_dtype_op_names,
                            const char **conditional_fp32_op_names,
                            const char **excluded_symbols,
                            const char **param_names,
                            const char **param_vals,
                            const char **model_param_names,
                            const char **arg_names) {
  nnvm::Symbol *result_sym = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol *>(sym_handle);
  nnvm::Graph g = Symbol2Graph(*sym);
  std::unordered_set<std::string> target_dtype_ops;
  std::unordered_set<std::string> fp32_ops;
  std::unordered_set<std::string> widest_dtype_ops;
  std::unordered_set<std::string> excluded_syms;
  std::unordered_set<std::string> model_params;

  // conditional_fp32_ops contains the mapping of op_name -> (map of param_name -> param_values)
  // which need to be conditionally selected to be casted to FP32
  std::unordered_map<std::string,
                     std::unordered_map<std::string,
                                        std::vector<std::string>>> conditional_fp32_ops;
  int target_dt = *target_dtype;

  for (size_t i = 0; i < num_target_dtype_op_names; ++i) {
    target_dtype_ops.emplace(target_dtype_op_names[i]);
  }
  for (size_t i = 0; i < num_fp32_op_names; ++i) {
    fp32_ops.emplace(fp32_op_names[i]);
  }
  for (size_t i = 0; i < num_widest_dtype_op_names; ++i) {
    widest_dtype_ops.emplace(widest_dtype_op_names[i]);
  }
  for (size_t i = 0; i < num_excluded_symbols; ++i) {
    excluded_syms.emplace(excluded_symbols[i]);
  }
  for (size_t i = 0; i < num_model_params; ++i) {
    model_params.emplace(model_param_names[i]);
  }

  for (size_t i = 0; i < num_ind_ptr - 1; ++i) {
    for (int j = ind_ptr[i]; j < ind_ptr[i + 1]; ++j) {
      conditional_fp32_ops[conditional_fp32_op_names[i]][param_names[i]]
          .emplace_back(std::string(param_vals[j]));
    }
  }

  std::unordered_map<std::string, int> kwargs;
  std::unordered_map<std::string, int> node_name_dtype_map, node_without_dtype_map;
  nnvm::DTypeVector arg_types(g.indexed_graph().input_nodes().size(), -1);
  for (uint32_t i = 0; i < num_args; ++i) {
    kwargs[arg_names[i]] = arg_type_data[i];
    node_name_dtype_map[arg_names[i]] = arg_type_data[i];
  }
  mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_types, "InferType");

  g.attrs["target_dtype_ops"] =
      std::make_shared<nnvm::any>(std::move(target_dtype_ops));
  g.attrs["fp32_ops"] = std::make_shared<nnvm::any>(std::move(fp32_ops));
  g.attrs["widest_dtype_ops"] =
      std::make_shared<nnvm::any>(std::move(widest_dtype_ops));
  g.attrs["conditional_fp32_ops"] =
      std::make_shared<nnvm::any>(std::move(conditional_fp32_ops));
  g.attrs["excluded_syms"] =
      std::make_shared<nnvm::any>(std::move(excluded_syms));
  g.attrs["target_dtype"] = std::make_shared<nnvm::any>(target_dt);

  g = ApplyPass(std::move(g), "ReducePrecision");
  // Need to run type inference since it is possible that inferred
  // type of some inputs has changed
  g = mxnet::exec::InferType(std::move(g), std::move(arg_types), "");
  const nnvm::DTypeVector &inferred_dtypes =
      g.GetAttr<nnvm::DTypeVector>("dtype");

  g.attrs["inferred_dtypes"] = std::make_shared<dmlc::any>(std::move(inferred_dtypes));
  g.attrs["target_dtype"] = std::make_shared<nnvm::any>(target_dt);

  if (cast_optional_params) {
    g = ApplyPass(std::move(g), "AMPInferUnknown");
    const nnvm::DTypeVector &inferred_dtype_result =
        g.GetAttr<nnvm::DTypeVector>("inferred_dtype_result");
    const nnvm::IndexedGraph &idx = g.indexed_graph();
    // set node name -> input dtype mapping using infer dtype
    _SetInputDTypes(idx, inferred_dtype_result, &node_name_dtype_map, &node_without_dtype_map);
  } else {
    const nnvm::IndexedGraph &idx = g.indexed_graph();
    // set node name -> input dtype mapping using infer dtype
    _SetInputDTypes(idx, inferred_dtypes, &node_name_dtype_map, &node_without_dtype_map);
  }


  result_sym->outputs = g.outputs;
  *ret_sym_handle = result_sym;
  nnvm::Symbol *ret_sym = static_cast<nnvm::Symbol *>(*ret_sym_handle);
  const std::vector<nnvm::NodePtr>& args = ret_sym->ListInputs(nnvm::Symbol::kAll);

  // update symbol dtype attrs using the node name -> dtype mapping, if dtype is already set
  // in the symbol, else set dtype for the model_params
  _UpdateSymDTypeAttrs(node_name_dtype_map, node_without_dtype_map, model_params, args);

  API_END_HANDLE_ERROR(delete result_sym);
}

int MXSetCalibTableToQuantizedSymbol(SymbolHandle qsym_handle,
                                     const uint32_t num_layers,
                                     const char** layer_names,
                                     const float* min_ranges,
                                     const float* max_ranges,
                                     SymbolHandle* ret_qsym_handle) {
  nnvm::Symbol* s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(qsym_handle);
  nnvm::Graph g = Symbol2Graph(*sym);
  std::unordered_map<std::string, std::pair<float, float>> calib_table;
  for (size_t i = 0; i < num_layers; ++i) {
    calib_table.emplace(layer_names[i], std::make_pair(min_ranges[i], max_ranges[i]));
  }
  g.attrs["calib_table"] = std::make_shared<nnvm::any>(std::move(calib_table));
  g = ApplyPass(std::move(g), "SetCalibTableToQuantizedGraph");
  s->outputs = g.outputs;
  *ret_qsym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXGenBackendSubgraph(SymbolHandle sym_handle, const char *backend_name,
                         SymbolHandle *ret_sym_handle) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol *>(sym_handle);
  *s = sym->Copy();
  auto backend = mxnet::op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(backend_name);
  const auto& subgraph_prop_list = backend->GetSubgraphProperties();
  for (auto property : subgraph_prop_list) {
    if (property->HasAttr("disable") && property->GetAttr<bool>("disable") == true) {
      auto full_name = property->HasAttr("property_name")
                           ? property->GetAttr<std::string>("property_name")
                           : std::string();
      LOG(INFO) << "subgraph property " << full_name << " from backend " << backend_name
                << " is disabled.";
      continue;
    }
    nnvm::Graph g = Symbol2Graph(*s);
    property->SetAttr("graph", g);
    g.attrs["subgraph_property"] = std::make_shared<nnvm::any>(property);
    g = ApplyPass(std::move(g), "BuildSubgraph");
    property->RemoveAttr("graph");
    g.attrs.erase("subgraph_property");
    s->outputs = g.outputs;
  }
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXGenAtomicSymbolFromSymbol(SymbolHandle sym_handle, SymbolHandle *ret_sym_handle) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *source = static_cast<nnvm::Symbol *>(sym_handle);
  CHECK_GE(source->outputs.size(), 1) << "Input symbol does not have outputs.";
  const auto &node = source->outputs[0].node;
  for (const auto &other_node : source->outputs) {
    if (node.get() != other_node.node.get()) {
      LOG(FATAL)
        << "Generating atomic symbol from other symbol only works for nongrouped symbol.";
    }
  }
  const auto *op = node->op();
  const auto attrs = source->ListAttrs(nnvm::Symbol::ListAttrOption::kShallow);
  *s = nnvm::Symbol::CreateFunctor(op, attrs);
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXShallowCopySymbol(SymbolHandle src, SymbolHandle* out) {
  nnvm::Symbol* out_sym = new nnvm::Symbol;
  API_BEGIN();
  nnvm::Symbol* src_sym = static_cast<nnvm::Symbol*>(src);
  *out_sym = *src_sym;
  *out = out_sym;
  API_END_HANDLE_ERROR(delete out_sym);
}

int MXOptimizeForBackend(SymbolHandle sym_handle,
                         const char* backend_name,
                         const int dev_type,
                         SymbolHandle* ret_sym_handle,
                         const mx_uint len,
                         NDArrayHandle* in_args_handle,
                         const mx_uint num_options,
                         const char** keys,
                         const char** vals) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol *>(sym_handle);
  *s = sym->Copy();
  nnvm::Graph g = Symbol2Graph(*s);
  if (len) {
    NDArray **in_args_ptr = reinterpret_cast<NDArray**>(in_args_handle);
    Context default_ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), 0);
    mxnet::ShapeVector arg_shapes(len);
    nnvm::DTypeVector arg_dtypes(len);
    StorageTypeVector arg_stypes(len);
    for (mx_uint i = 0; i < len; i++) {
      const auto &in_arg = *(in_args_ptr[i]);
      arg_shapes[i] = in_arg.shape();
      arg_dtypes[i] = in_arg.dtype();
      arg_stypes[i] = in_arg.storage_type();
    }
    const auto& indexed_graph = g.indexed_graph();
    const auto num_forward_inputs = indexed_graph.input_nodes().size();
    g.attrs["context"] = std::make_shared<nnvm::any>(
        exec::ContextVector(indexed_graph.num_nodes(), default_ctx));
    // infer shapes
    g = exec::InferShape(std::move(g), std::move(arg_shapes), "__shape__");
    // infer dtypes
    g = exec::InferType(std::move(g), std::move(arg_dtypes), "__dtype__");
    if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) {
      common::HandleInferTypeError(num_forward_inputs, indexed_graph,
                                   g.GetAttr<nnvm::DTypeVector>("dtype"));
    }
    // infer stypes
    g = exec::InferStorageType(std::move(g), std::move(arg_stypes), "__storage_type__");
    if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) {
      common::HandleInferStorageTypeError(num_forward_inputs, indexed_graph,
                                          g.GetAttr<StorageTypeVector>("storage_type"));
    }
  }
  std::vector<std::pair<std::string, std::string>> options_map;
  for (mx_uint i = 0; i < num_options; ++i) {
    options_map.emplace_back(keys[i], vals[i]);
  }
  const auto backend = mxnet::op::SubgraphBackendRegistry::Get()->GetSubgraphBackend(backend_name);
  const auto& subgraph_prop_list = backend->GetSubgraphProperties();
  for (auto property : subgraph_prop_list) {
    property->PrePartition(g, options_map);
    g.attrs["subgraph_property"] = std::make_shared<nnvm::any>(property);
    g = ApplyPass(std::move(g), "BuildSubgraph");
    g.attrs.erase("subgraph_property");
    property->PostPartition(g);
  }
  s->outputs = g.outputs;
  *ret_sym_handle = s;
  API_END_HANDLE_ERROR(delete s);
}

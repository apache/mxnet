/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_symbolic.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>
#include <nnvm/pass_functions.h>
#include "./c_api_common.h"
#include "../operator/operator_common.h"

namespace mxnet {
namespace op {
void RegisterLegacyOpProp();
void RegisterLegacyNDFunc();
void FixLegacyGraphBatchNorm(nnvm::Symbol* s);
}

// convert nnvm symbol to a nnvm graph.
nnvm::Graph Symbol2Graph(const nnvm::Symbol &s) {
  nnvm::Graph g;
  g.outputs = s.outputs;
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

int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListUniqueOps(out_size, out_array);
}

int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                const char **name,
                                const char **description,
                                mx_uint *num_args,
                                const char ***arg_names,
                                const char ***arg_type_infos,
                                const char ***arg_descriptions,
                                const char **key_var_num_args,
                                const char **return_type) {
  static auto& map_key_var_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  const Op* op = static_cast<Op*>(creator);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
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
                               mx_uint num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  return NNSymbolCreateAtomicSymbol(creator, num_param, keys, vals, out);
}

int MXSymbolCreateVariable(const char *name, SymbolHandle *out) {
  return NNSymbolCreateVariable(name, out);
}

int MXSymbolCreateGroup(mx_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  return NNSymbolCreateGroup(num_symbols, symbols, out);
}

int MXSymbolGetOutput(SymbolHandle symbol,
                      mx_uint index,
                      SymbolHandle *out) {
  return NNSymbolGetOutput(symbol, index, out);
}

int MXSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  return NNSymbolGetInternals(symbol, out);
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
  return NNSymbolGetAttr(symbol, key, out, success);
}

int MXSymbolSetAttr(SymbolHandle symbol,
                    const char* key,
                    const char* value) {
  return NNSymbolSetAttrs(symbol, 1, &key, &value);
}

int MXSymbolListAttr(SymbolHandle symbol,
                     mx_uint *out_size,
                     const char*** out) {
  return NNSymbolListAttrs(symbol, 0, out_size, out);
}

int MXSymbolListAttrShallow(SymbolHandle symbol,
                            mx_uint *out_size,
                            const char*** out) {
  return NNSymbolListAttrs(symbol, 1, out_size, out);
}

int MXSymbolListOutputs(SymbolHandle symbol,
                        mx_uint *out_size,
                        const char ***out_str_array) {
  return NNSymbolListOutputNames(symbol, out_size, out_str_array);
}

int MXSymbolCompose(SymbolHandle sym,
                    const char *name,
                    mx_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  return NNSymbolCompose(sym, name, num_args, keys, args);
}

// adapter functions that re-implements the functions.
int MXSymbolListArguments(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  return NNSymbolListInputNames(symbol, 1, out_size, out_str_array);
}

int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                mx_uint *out_size,
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

int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  dmlc::istream is(fi.get());
  s->outputs = nnvm::pass::LoadJSON(
     std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>())).outputs;
  op::FixLegacyGraphBatchNorm(s);
  *out = s;
  is.set_stream(nullptr);
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  s->outputs = nnvm::pass::LoadJSON(json).outputs;
  op::FixLegacyGraphBatchNorm(s);
  *out = s;
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
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
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
                       mx_uint num_args,
                       const char** keys,
                       const mx_uint *arg_ind_ptr,
                       const mx_uint *arg_shape_data,
                       mx_uint *in_shape_size,
                       const mx_uint **in_shape_ndim,
                       const mx_uint ***in_shape_data,
                       mx_uint *out_shape_size,
                       const mx_uint **out_shape_ndim,
                       const mx_uint ***out_shape_data,
                       mx_uint *aux_shape_size,
                       const mx_uint **aux_shape_ndim,
                       const mx_uint ***aux_shape_data,
                       int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  nnvm::ShapeVector arg_shapes(g.indexed_graph().input_nodes().size(), TShape());
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (mx_uint i = 0; i < num_args; ++i) {
      arg_shapes[read_only_args[i]] = TShape(arg_shape_data + arg_ind_ptr[i],
                                             arg_shape_data + arg_ind_ptr[i+1]);
    }
  } else {
    std::unordered_map<std::string, TShape> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = TShape(arg_shape_data + arg_ind_ptr[i],
                               arg_shape_data + arg_ind_ptr[i+1]);
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_shapes, "InferShape");
  }

  try {
    g = nnvm::pass::InferShape(std::move(g), arg_shapes, "__shape__");
  } catch (const mxnet::op::InferShapeError &err) {
    throw dmlc::Error(err.msg);
  }

  // copy back
  CopyAttr(g.indexed_graph(), g.GetAttr<nnvm::ShapeVector>("shape"),
           &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));

  // copy data back
  MXAPIThreadLocalEntry::SetupShapeArrayReturn(
      ret->arg_shapes, &(ret->arg_shape_ndim), &(ret->arg_shape_data));
  MXAPIThreadLocalEntry::SetupShapeArrayReturn(
      ret->out_shapes, &(ret->out_shape_ndim), &(ret->out_shape_data));
  MXAPIThreadLocalEntry::SetupShapeArrayReturn(
      ret->aux_shapes, &(ret->aux_shape_ndim), &(ret->aux_shape_data));
  *in_shape_size = static_cast<mx_uint>(ret->arg_shapes.size());
  *in_shape_ndim = dmlc::BeginPtr(ret->arg_shape_ndim);
  *in_shape_data = dmlc::BeginPtr(ret->arg_shape_data);
  *out_shape_size = static_cast<mx_uint>(ret->out_shapes.size());
  *out_shape_ndim = dmlc::BeginPtr(ret->out_shape_ndim);
  *out_shape_data = dmlc::BeginPtr(ret->out_shape_data);
  *aux_shape_size = static_cast<mx_uint>(ret->aux_shapes.size());
  *aux_shape_ndim = dmlc::BeginPtr(ret->aux_shape_ndim);
  *aux_shape_data = dmlc::BeginPtr(ret->aux_shape_data);
  // mark complete
  *complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
  API_END();
}

int MXSymbolInferShapePartial(SymbolHandle sym,
                              mx_uint num_args,
                              const char** keys,
                              const mx_uint *arg_ind_ptr,
                              const mx_uint *arg_shape_data,
                              mx_uint *in_shape_size,
                              const mx_uint **in_shape_ndim,
                              const mx_uint ***in_shape_data,
                              mx_uint *out_shape_size,
                              const mx_uint **out_shape_ndim,
                              const mx_uint ***out_shape_data,
                              mx_uint *aux_shape_size,
                              const mx_uint **aux_shape_ndim,
                              const mx_uint ***aux_shape_data,
                              int *complete) {
  int succ;
  *complete = 1;
  return MXSymbolInferShape(sym, num_args, keys,
                            arg_ind_ptr, arg_shape_data,
                            in_shape_size, in_shape_ndim, in_shape_data,
                            out_shape_size, out_shape_ndim, out_shape_data,
                            aux_shape_size, aux_shape_ndim, aux_shape_data,
                            &succ);
}

int MXSymbolInferType(SymbolHandle sym,
                      mx_uint num_args,
                      const char** keys,
                      const int *arg_type_data,
                      mx_uint *in_type_size,
                      const int **in_type_data,
                      mx_uint *out_type_size,
                      const int **out_type_data,
                      mx_uint *aux_type_size,
                      const int **aux_type_data,
                      int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  nnvm::DTypeVector arg_types(g.indexed_graph().input_nodes().size(), -1);
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (mx_uint i = 0; i < num_args; ++i) {
      arg_types[read_only_args[i]] = arg_type_data[i];
    }
  } else {
    std::unordered_map<std::string, int> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = arg_type_data[i];
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_types, "InferType");
  }

  g = nnvm::pass::InferType(std::move(g), arg_types);
  // copy back
  CopyAttr(g.indexed_graph(), g.GetAttr<nnvm::DTypeVector>("dtype"),
           &(ret->arg_types), &(ret->out_types), &(ret->aux_types));

  *in_type_size = static_cast<mx_uint>(ret->arg_types.size());
  *in_type_data = dmlc::BeginPtr(ret->arg_types);
  *out_type_size = static_cast<mx_uint>(ret->out_types.size());
  *out_type_data = dmlc::BeginPtr(ret->out_types);
  *aux_type_size = static_cast<mx_uint>(ret->aux_types.size());
  *aux_type_data = dmlc::BeginPtr(ret->aux_types);
  *complete = (g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0);
  API_END();
}

int MXSymbolGrad(SymbolHandle sym, mx_uint num_wrt, const char** wrt, SymbolHandle* out) {
  API_BEGIN();
  LOG(FATAL) << "not implemented";
  API_END();
}

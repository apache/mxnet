#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>
#include <nnvm/pass_functions.h>
#include <mxnet/symbolic.h>
#include "./c_api_common.h"

// symbolic configuration generation API.
// Redirect to NNVM's C API
int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  return NNSymbolListAtomicSymbolCreators(out_size, out_array);
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
  key_var_num_args = nullptr;
  return NNSymbolGetAtomicSymbolInfo(
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
  return NNSymbolListAttrs(symbol, 1, out_size, out);
}

int MXSymbolListAttrShallow(SymbolHandle symbol,
                            mx_uint *out_size,
                            const char*** out) {
  return NNSymbolListAttrs(symbol, 0, out_size, out);
}

int MXSymbolListArguments(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  return NNSymbolListArguments(symbol, out_size, out_str_array);
}

int MXSymbolListOutputs(SymbolHandle symbol,
                        mx_uint *out_size,
                        const char ***out_str_array) {
  return NNSymbolListOutputs(symbol, out_size, out_str_array);
}

int MXSymbolCompose(SymbolHandle sym,
                    const char *name,
                    mx_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  return NNSymbolCompose(sym, name, num_args, keys, args);
}

// adapter functions that re-implements the functions.
int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                mx_uint *out_size,
                                const char ***out_str_array) {
  // TODO(tqchen)
  API_BEGIN();
  LOG(FATAL) << "not implemented";
  API_END();
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
      std::string(std::istreambuf_iterator<char>(is), {})).outputs;
  is.set_stream(nullptr);
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  s->outputs = nnvm::pass::LoadJSON(json).outputs;
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  API_BEGIN();
  nnvm::Graph g;
  g.outputs = s->outputs;
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  dmlc::ostream os(fo.get());
  os << nnvm::pass::SaveJSON(g);
  // reset file pointer, force flush
  os.set_stream(nullptr);
  API_END();
}

int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Graph g;
  g.outputs = s->outputs;
  ret->ret_str = nnvm::pass::SaveJSON(g);
  *out_json = ret->ret_str.c_str();
  API_END();
}


int MXSymbolGrad(SymbolHandle sym, mx_uint num_wrt, const char** wrt, SymbolHandle* out) {
  API_BEGIN();
  Symbol* s = static_cast<Symbol*>(sym);
  std::vector<std::string> wrts(num_wrt);
  for (mx_uint i = 0; i < num_wrt; ++i) {
    wrts[i] = wrt[i];
  }
  Symbol* ret = new Symbol();
  *ret = s->Grad(wrts);
  *out = ret;
  API_END();
}

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
  Symbol *s = static_cast<Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  bool succ;
  API_BEGIN();
  if (keys == nullptr && num_args != 0) {
    ret->arg_shapes.clear();
    for (mx_uint i = 0; i < num_args; ++i) {
      ret->arg_shapes.push_back(TShape(arg_shape_data + arg_ind_ptr[i],
                                       arg_shape_data + arg_ind_ptr[i+1]));
    }
    succ = s->InferShape(&(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));
  } else {
    std::unordered_map<std::string, TShape> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = TShape(arg_shape_data + arg_ind_ptr[i],
                               arg_shape_data + arg_ind_ptr[i+1]);
    }
    succ = s->InferShape(kwargs, &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));
  }
  if (succ) {
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
    *complete = 1;
  } else {
    *complete = 0;
  }
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
  Symbol *s = static_cast<Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  bool succ;
  API_BEGIN();
  if (keys == nullptr && num_args != 0) {
    ret->arg_shapes.clear();
    for (mx_uint i = 0; i < num_args; ++i) {
      ret->arg_shapes.push_back(TShape(arg_shape_data + arg_ind_ptr[i],
                                       arg_shape_data + arg_ind_ptr[i+1]));
    }
    succ = s->InferShape(&(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes), true);
  } else {
    std::unordered_map<std::string, TShape> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = TShape(arg_shape_data + arg_ind_ptr[i],
                               arg_shape_data + arg_ind_ptr[i+1]);
    }
    succ = s->InferShape(kwargs, &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes), true);
  }
  if (succ) {
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
    *complete = 1;
  } else {
    *complete = 0;
  }
  API_END();
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
  Symbol *s = static_cast<Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  bool succ;
  API_BEGIN();
  if (keys == nullptr && num_args != 0) {
    ret->arg_types.clear();
    for (mx_uint i = 0; i < num_args; ++i) {
      ret->arg_types.push_back(arg_type_data[i]);
    }
    succ = s->InferType(&(ret->arg_types), &(ret->out_types), &(ret->aux_types));
  } else {
    std::unordered_map<std::string, int> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = arg_type_data[i];
    }
    succ = s->InferType(kwargs, &(ret->arg_types), &(ret->out_types), &(ret->aux_types));
  }
  if (succ) {
    *in_type_size = static_cast<mx_uint>(ret->arg_types.size());
    *in_type_data = dmlc::BeginPtr(ret->arg_types);
    *out_type_size = static_cast<mx_uint>(ret->out_types.size());
    *out_type_data = dmlc::BeginPtr(ret->out_types);
    *aux_type_size = static_cast<mx_uint>(ret->aux_types.size());
    *aux_type_data = dmlc::BeginPtr(ret->aux_types);
    *complete = 1;
  } else {
    *complete = 0;
  }
  API_END();
}

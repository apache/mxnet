/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api.cc
 * \brief C API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/recordio.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/symbolic.h>
#include <mxnet/operator.h>
#include <mxnet/optimizer.h>
#include <mxnet/io.h>
#include <mxnet/c_api.h>
#include <mxnet/kvstore.h>
#include <mxnet/mxrtc.h>
#include <vector>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <functional>
#include <utility>
#include "./c_api_error.h"
#include "../common/thread_local.h"
#include "../operator/custom-inl.h"

using namespace mxnet;

/*! \brief entry to to easily hold returning information */
struct MXAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning handles */
  std::vector<void *> ret_handles;
  /*! \brief result holder for returning shapes */
  std::vector<TShape> arg_shapes, out_shapes, aux_shapes;
  /*! \brief result holder for returning type flags */
  std::vector<int> arg_types, out_types, aux_types;
  /*! \brief result holder for returning shape dimensions */
  std::vector<mx_uint> arg_shape_ndim, out_shape_ndim, aux_shape_ndim;
  /*! \brief result holder for returning shape pointer */
  std::vector<const mx_uint*> arg_shape_data, out_shape_data, aux_shape_data;
  // helper function to setup return value of shape array
  inline static void SetupShapeArrayReturn(
      const std::vector<TShape> &shapes,
      std::vector<mx_uint> *ndim,
      std::vector<const mx_uint*> *data) {
    ndim->resize(shapes.size());
    data->resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      ndim->at(i) = shapes[i].ndim();
      data->at(i) = shapes[i].data();
    }
  }
};

// define the threadlocal store.
typedef mxnet::common::ThreadLocalStore<MXAPIThreadLocalEntry> MXAPIThreadLocalStore;

// Internal function to get the information
// from function registry
// Used to implement MXSymbolGetAtomicSymbolInfo and MXFuncGetInfo
template<typename FunRegType>
inline int MXAPIGetFunctionRegInfo(const FunRegType *e,
                                   const char **name,
                                   const char **description,
                                   mx_uint *num_args,
                                   const char ***arg_names,
                                   const char ***arg_type_infos,
                                   const char ***arg_descriptions,
                                   const char **return_type) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  API_BEGIN();
  *name = e->name.c_str();
  *description = e->description.c_str();
  *num_args = static_cast<mx_uint>(e->arguments.size());
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
int MXRandomSeed(int seed) {
  API_BEGIN();
  mxnet::RandomSeed(seed);
  API_END();
}

int MXNotifyShutdown() {
  API_BEGIN();
  Engine::Get()->NotifyShutdown();
  API_END();
}

int MXNDArrayCreateNone(NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray();
  API_END();
}

int MXNDArrayCreate(const mx_uint *shape,
                    mx_uint ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(
      TShape(shape, shape + ndim),
      Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
      delay_alloc != 0);
  API_END();
}

int MXNDArrayCreateEx(const mx_uint *shape,
                    mx_uint ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    int dtype,
                    NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(
      TShape(shape, shape + ndim),
      Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
      delay_alloc != 0,
      dtype);
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
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
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
                  mx_uint num_args,
                  NDArrayHandle* args,
                  const char** keys) {
  API_BEGIN();
  std::vector<NDArray> data(num_args);
  std::vector<std::string> names;
  for (mx_uint i = 0; i < num_args; ++i) {
    data[i] = *static_cast<NDArray*>(args[i]);
  }
  if (keys != nullptr) {
    names.resize(num_args);
    for (mx_uint i = 0; i < num_args; ++i) {
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
                  mx_uint *out_size,
                  NDArrayHandle** out_arr,
                  mx_uint *out_name_size,
                  const char*** out_names) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
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
  *out_size = static_cast<mx_uint>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<mx_uint>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXNDArrayFree(NDArrayHandle handle) {
  API_BEGIN();
  delete static_cast<NDArray*>(handle);
  API_END();
}

int MXNDArraySlice(NDArrayHandle handle,
                   mx_uint slice_begin,
                   mx_uint slice_end,
                   NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->Slice(
      slice_begin, slice_end);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt(NDArrayHandle handle,
                mx_uint idx,
                NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->At(idx);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

MXNET_DLL int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *dims,
                               NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  TShape new_shape(dims, dims+ndim);
  *ptr = static_cast<NDArray*>(handle)->Reshape(new_shape);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayGetShape(NDArrayHandle handle,
                      mx_uint *out_dim,
                      const mx_uint **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const TShape &s = arr->shape();
    *out_dim = s.ndim();
    *out_pdata = s.data();
  } else {
    *out_dim = 0;
  }
  API_END();
}

int MXNDArrayGetData(NDArrayHandle handle,
                     mx_float **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    CHECK(arr->ctx().dev_mask() == cpu::kDevMask)
        << "MXNDArrayGetData can only be called for NDArray on CPU";
    const TBlob &b = arr->data();
    CHECK(b.CheckContiguous());
    *out_pdata = b.FlatTo2D<cpu, mx_float>().dptr_;
  } else {
    *out_pdata = nullptr;
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

int MXListFunctions(mx_uint *out_size,
                    FunctionHandle **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<NDArrayFunctionReg>::List();
  *out_size = static_cast<mx_uint>(vec.size());
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
                  mx_uint *num_args,
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
                   mx_uint *num_use_vars,
                   mx_uint *num_scalars,
                   mx_uint *num_mutate_vars,
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
                 mx_float *scalar_args,
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
                 mx_float *scalar_args,
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
// Part 3: symbolic configuration generation
//--------------------------------------------

int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<OperatorPropertyReg>::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (AtomicSymbolCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                const char **out) {
  API_BEGIN();
  OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
  *out = e->name.c_str();
  API_END();
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
  OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
  *key_var_num_args = e->key_var_num_args.c_str();
  return MXAPIGetFunctionRegInfo(e, name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 return_type);
}

int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                               mx_uint num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  Symbol *s = new Symbol();
  OperatorProperty *op = nullptr;

  API_BEGIN();
  OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
  op = e->body();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (mx_uint i = 0; i < num_param; ++i) {
    kwargs.push_back({std::string(keys[i]), std::string(vals[i])});
  }
  op->Init(kwargs);
  *s = Symbol::Create(op);
  *out = s;
  API_END_HANDLE_ERROR(delete s; delete op);
}

int MXSymbolCreateVariable(const char *name, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = Symbol::CreateVariable(name);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateGroup(mx_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  Symbol *s = new Symbol();
  Symbol **sym_arr = (Symbol**)symbols; // NOLINT(*)
  API_BEGIN();
  std::vector<Symbol> syms;
  for (mx_uint i = 0; i < num_symbols; ++i) {
    syms.push_back(*sym_arr[i]);
  }
  *s = Symbol::CreateGroup(syms);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolGetOutput(SymbolHandle symbol,
                      mx_uint index,
                      SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = (*static_cast<Symbol*>(symbol))[index];
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<Symbol*>(symbol)->GetInternals();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  dmlc::istream is(fi.get());
  dmlc::JSONReader reader(&is);
  s->Load(&reader);
  // reset file pointer
  is.set_stream(nullptr);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  std::string buf(json);
  std::istringstream is(buf);
  dmlc::JSONReader reader(&is);
  s->Load(&reader);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname) {
  Symbol *s = static_cast<Symbol*>(symbol);
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  dmlc::ostream os(fo.get());
  dmlc::JSONWriter writer(&os);
  s->Save(&writer);
  // reset file pointer, force flush
  os.set_stream(nullptr);
  API_END();
}

int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  s->Save(&writer);
  ret->ret_str = os.str();
  *out_json = ret->ret_str.c_str();
  API_END();
}

int MXSymbolFree(SymbolHandle symbol) {
  API_BEGIN();
  delete static_cast<Symbol*>(symbol);
  API_END();
}

int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<const Symbol*>(symbol)->Copy();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolPrint(SymbolHandle symbol, const char **out_str) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  s->Print(os);
  ret->ret_str = os.str();
  *out_str = (ret->ret_str).c_str();
  API_END();
}

int MXSymbolGetName(SymbolHandle symbol,
                    const char** out,
                    int* success) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  if (s->GetName(&(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
  }
  API_END();
}

int MXSymbolGetAttr(SymbolHandle symbol,
                    const char* key,
                    const char** out,
                    int* success) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  if (s->GetAttr(key, &(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
  }
  API_END();
}

int MXSymbolSetAttr(SymbolHandle symbol,
                    const char* key,
                    const char* value) {
  Symbol *s = static_cast<Symbol*>(symbol);
  API_BEGIN();
  s->SetAttr(key, value);
  API_END();
}

int _MXSymbolListAttrImpl(SymbolHandle symbol,
                          bool shalow,
                          mx_uint *out_size,
                          const char*** out) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::map<std::string, std::string> attr =
      std::move(shalow ? s->ListAttrShallow() : s->ListAttr());
  std::vector<std::string> attrList;
  *out_size = 0;
  for (auto it : attr) {
    attrList.push_back(it.first);
    attrList.push_back(it.second);
    (*out_size)++;
  }

  ret->ret_vec_str = std::move(attrList);
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListAttr(SymbolHandle symbol,
                     mx_uint *out_size,
                     const char*** out) {
  return _MXSymbolListAttrImpl(symbol, false, out_size, out);
}

int MXSymbolListAttrShallow(SymbolHandle symbol,
                            mx_uint *out_size,
                            const char*** out) {
  return _MXSymbolListAttrImpl(symbol, true, out_size, out);
}

int MXSymbolListArguments(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(s->ListArguments());
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<mx_uint>(ret->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListOutputs(SymbolHandle symbol,
                        mx_uint *out_size,
                        const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(s->ListOutputs());
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<mx_uint>(ret->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                mx_uint *out_size,
                                const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(s->ListAuxiliaryStates());
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<mx_uint>(ret->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolCompose(SymbolHandle sym,
                    const char *name,
                    mx_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  API_BEGIN();
  std::string s_name;
  if (name != nullptr) s_name = name;

  Symbol* s = static_cast<Symbol*>(sym);
  if (keys == nullptr && num_args != 0) {
    std::vector<Symbol> pos_args;
    for (mx_uint i = 0; i < num_args; ++i) {
      pos_args.push_back(*((Symbol*)args[i]));  //  NOLINT(*)
    }
    s->Compose(pos_args, s_name);
  } else {
    std::unordered_map<std::string, Symbol> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = *((Symbol*)args[i]);  //  NOLINT(*)
    }
    s->Compose(kwargs, s_name);
  }
  API_END();
}

int MXSymbolGrad(SymbolHandle sym, mx_uint num_wrt, const char** wrt, SymbolHandle* out) {
  API_BEGIN();
  Symbol* s = static_cast<Symbol*>(sym);
  std::vector<std::string> wrts(num_wrt);
  for (mx_uint i = 0; i < num_wrt; ++i) {
    wrts[i] = wrt[i];
  }
  Symbol* ret = new Symbol;
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
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> ndarrays;
  NDArray **args_ptr = reinterpret_cast<NDArray**>(head_grads);
  for (mx_uint i = 0; i < len; ++i) {
    ndarrays.push_back(*args_ptr[i]);
  }
  exec->Backward(ndarrays);
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
  API_BEGIN();
  Symbol *symb = static_cast<Symbol*>(symbol_handle);
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

//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
int MXListDataIters(mx_uint *out_size,
                    DataIterCreator **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<DataIteratorReg>::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (DataIterCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXDataIterGetIterInfo(DataIterCreator creator,
                          const char **name,
                          const char **description,
                          mx_uint *num_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions) {
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  return MXAPIGetFunctionRegInfo(e, name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 NULL);
}

int MXDataIterCreateIter(DataIterCreator creator,
                         mx_uint num_param,
                         const char **keys,
                         const char **vals,
                         DataIterHandle *out) {
  IIterator<DataBatch> *iter = nullptr;
  API_BEGIN();
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  iter = e->body();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (mx_uint i = 0; i < num_param; ++i) {
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
  TShape shape = db.data[1].shape();
  if (shape[1] == 1) {
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

int MXKVStoreFree(KVStoreHandle handle) {
  API_BEGIN();
  delete static_cast<KVStore*>(handle);
  API_END();
}

int MXKVStoreInit(KVStoreHandle handle,
                  mx_uint num,
                  const int* keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (mx_uint i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
  API_END();
}

int MXKVStorePush(KVStoreHandle handle,
                  mx_uint num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (mx_uint i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePull(KVStoreHandle handle,
                  mx_uint num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (mx_uint i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority);
  API_END();
}

int MXKVStoreSetUpdater(KVStoreHandle handle,
                        MXKVStoreUpdater updater,
                        void* updater_handle) {
  API_BEGIN();
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

int MXInitPSEnv(mx_uint num_vars,
                const char **keys,
                const char **vals) {
  API_BEGIN();
  std::unordered_map<std::string, std::string> kwargs;
  for (mx_uint i = 0; i < num_vars; ++i) {
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
  API_END();
}

int MXRecordIOWriterWriteRecord(RecordIOHandle *handle,
                                const char *buf, size_t size) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->writer->WriteRecord(reinterpret_cast<const void*>(buf), size);
  API_END();
}

int MXRecordIOWriterTell(RecordIOHandle *handle, size_t *pos) {
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

int MXRecordIOReaderFree(RecordIOHandle *handle) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  delete context->reader;
  delete context->stream;
  delete context->read_buff;
  API_END();
}

int MXRecordIOReaderReadRecord(RecordIOHandle *handle,
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

int MXRecordIOReaderSeek(RecordIOHandle *handle, size_t pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->reader->Seek(pos);
  API_END();
}

int MXRtcCreate(char* name, mx_uint num_input, mx_uint num_output,
                char** input_names, char** output_names,
                NDArrayHandle* inputs, NDArrayHandle* outputs,
                char* kernel, RtcHandle *out) {
  API_BEGIN();
#if ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  std::vector<std::pair<std::string, NDArray> > input, output;
  for (mx_uint i = 0; i < num_input; ++i) {
    input.push_back(std::pair<std::string, NDArray>(input_names[i],
                                                    *reinterpret_cast<NDArray*>(inputs[i])));
  }
  for (mx_uint i = 0; i < num_output; ++i) {
    output.push_back(std::pair<std::string, NDArray>(output_names[i],
                                                     *reinterpret_cast<NDArray*>(outputs[i])));
  }
  MXRtc *rtc = new MXRtc(name, input, output, kernel);
  *out = reinterpret_cast<RtcHandle>(rtc);
#else
  LOG(FATAL) << "Need to compile with USE_CUDA=1 and USE_NVRTC=1 for MXRtc.";
#endif  // ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  API_END();
}

int MXRtcPush(RtcHandle handle, mx_uint num_input, mx_uint num_output,
              NDArrayHandle* inputs, NDArrayHandle* outputs,
              mx_uint gridDimX,
              mx_uint gridDimY,
              mx_uint gridDimZ,
              mx_uint blockDimX,
              mx_uint blockDimY,
              mx_uint blockDimZ) {
  API_BEGIN();
#if ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  std::vector<NDArray> input, output;
  for (mx_uint i = 0; i < num_input; ++i) {
    input.push_back(*reinterpret_cast<NDArray*>(inputs[i]));
  }
  for (mx_uint i = 0; i < num_output; ++i) {
    output.push_back(*reinterpret_cast<NDArray*>(outputs[i]));
  }
  reinterpret_cast<MXRtc*>(handle)->push(input, output,
                                         gridDimX,
                                         gridDimY,
                                         gridDimZ,
                                         blockDimX,
                                         blockDimY,
                                         blockDimZ);
#else
  LOG(FATAL) << "Need to compile with USE_CUDA=1 and USE_NVRTC=1 for MXRtc.";
#endif  // ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  API_END();
}

int MXRtcFree(RtcHandle handle) {
  API_BEGIN();
#if ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  delete reinterpret_cast<MXRtc*>(handle);
#else
  LOG(FATAL) << "Need to compile with USE_CUDA=1 and USE_NVRTC=1 for MXRtc.";
#endif  // ((MXNET_USE_CUDA) && (MXNET_USE_NVRTC))
  API_END();
}

int MXOptimizerFindCreator(const char *key,
                           OptimizerCreator *out) {
  API_BEGIN();
  *out = (OptimizerCreator*)dmlc::Registry<OptimizerReg>::Find(key);  // NOLINT(*)
  API_END();
}

int MXOptimizerCreateOptimizer(OptimizerCreator creator,
                               mx_uint num_param,
                               const char **keys,
                               const char **vals,
                               OptimizerHandle *out) {
  API_BEGIN();
  OptimizerReg *e = static_cast<OptimizerReg *>(creator);
  Optimizer* opt = e->body();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (mx_uint i = 0; i < num_param; ++i) {
    kwargs.push_back({std::string(keys[i]), std::string(vals[i])});
  }
  opt->Init(kwargs);
  *out = opt;
  API_END();
}

int MXOptimizerFree(OptimizerHandle handle) {
  API_BEGIN();
  Optimizer *opt = static_cast<Optimizer*>(handle);
  delete opt;
  API_END();
}

int MXOptimizerUpdate(OptimizerHandle handle,
                      int index,
                      NDArrayHandle weight,
                      NDArrayHandle grad,
                      mx_float lr,
                      mx_float wd) {
  API_BEGIN();
  Optimizer *opt = static_cast<Optimizer*>(handle);
  opt->Update(index,
              static_cast<NDArray*>(weight),
              static_cast<NDArray*>(grad),
              lr, wd);
  API_END();
}

int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator) {
  API_BEGIN();
  mxnet::op::CustomOpProp::Register(op_type, creator);
  API_END();
}

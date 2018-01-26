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
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/recordio.h>
#include <dmlc/omp.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/io.h>
#include <mxnet/c_api.h>
#include <mxnet/kvstore.h>
#include <mxnet/rtc.h>
#include <mxnet/storage.h>
#include <vector>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <functional>
#include <utility>
#include "./c_api_common.h"
#include "../operator/custom/custom-inl.h"
#include "../engine/profiler.h"

using namespace mxnet;

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

int MXSetProfilerConfig(int mode, const char* filename) {
  // mode, kOnlySymbolic: 0, kAllOperator: 1
  API_BEGIN();
#if MXNET_USE_PROFILER
  engine::Profiler::Get()->SetConfig(engine::Profiler::ProfilerMode(mode), std::string(filename));
#else
  LOG(FATAL) << "Need to compile with USE_PROFILER=1 for MXNet Profiler";
#endif
  API_END();
}

int MXDumpProfile() {
  API_BEGIN();
#if MXNET_USE_PROFILER
  engine::Profiler *profiler = engine::Profiler::Get();
  CHECK(profiler->IsEnableOutput())
    << "Profiler haven't been run. Config and start profiler first";
  engine::Profiler::Get()->DumpProfile();
#else
  LOG(FATAL) << "Need to compile with USE_PROFILER=1 for MXNet Profiler";
#endif
  API_END()
}

int MXSetProfilerState(int state) {
  // state, kNotRunning: 0, kRunning: 1
  API_BEGIN();
#if MXNET_USE_PROFILER
  engine::Profiler::Get()->SetState(engine::Profiler::ProfilerState(state));
#else
  LOG(FATAL) << "Need to compile with USE_PROFILER=1 for MXNet Profiler";
#endif
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

int MXGetVersion(int *out) {
  API_BEGIN();
  *out = static_cast<int>(MXNET_VERSION);
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

int MXNDArrayCreateSparseEx(int storage_type,
                    const mx_uint *shape,
                    mx_uint ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    int dtype,
                    mx_uint num_aux,
                    int *aux_type,
                    mx_uint *aux_ndims,
                    const mx_uint *aux_shape,
                    NDArrayHandle *out) {
  API_BEGIN();
  std::vector<int> aux_types;
  std::vector<TShape> aux_shapes;
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
      TShape(shape, shape + ndim),
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
  *ptr = static_cast<NDArray*>(handle)->SliceWithRecord(
      slice_begin, slice_end);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt(NDArrayHandle handle,
                mx_uint idx,
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
  TShape new_shape(dims, dims+ndim);
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
                      mx_uint *out_dim,
                      const mx_uint **out_pdata) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const TShape &s = arr->shape();
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

int MXNDArrayGetData(NDArrayHandle handle,
                     void **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_pdata = arr->data().dptr_;
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

int MXNDArrayGetAuxType(NDArrayHandle handle,
                        mx_uint i,
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
                           mx_uint i,
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

int MXKVStoreSetGradientCompression(KVStoreHandle handle, mx_uint num_params,
                                    const char** keys, const char** vals) {
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > params;
  for (mx_uint i = 0; i < num_params; ++i) {
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

int MXKVStoreInitEx(KVStoreHandle handle,
                  mx_uint num,
                  const char** keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
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

int MXKVStorePushEx(KVStoreHandle handle,
                  mx_uint num,
                  const char** keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
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

int MXKVStorePullEx(KVStoreHandle handle,
                  mx_uint num,
                  const char** keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (mx_uint i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePullRowSparse(KVStoreHandle handle,
                           mx_uint num,
                           const int* keys,
                           NDArrayHandle* vals,
                           const NDArrayHandle* row_ids,
                           int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (mx_uint i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_val_rowids[i] = std::make_pair(static_cast<NDArray*>(vals[i]),
                                     *static_cast<NDArray*>(row_ids[i]));
  }
  static_cast<KVStore*>(handle)->PullRowSparse(v_keys, v_val_rowids, priority);
  API_END();
}

int MXKVStorePullRowSparseEx(KVStoreHandle handle,
                             mx_uint num,
                             const char** keys,
                             NDArrayHandle* vals,
                             const NDArrayHandle* row_ids,
                             int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (mx_uint i = 0; i < num; ++i) {
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

int MXRtcCreate(char* name, mx_uint num_input, mx_uint num_output,
                char** input_names, char** output_names,
                NDArrayHandle* inputs, NDArrayHandle* outputs,
                char* kernel, RtcHandle *out) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
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
                        mx_uint grid_dim_x, mx_uint grid_dim_y,
                        mx_uint grid_dim_z, mx_uint block_dim_x,
                        mx_uint block_dim_y, mx_uint block_dim_z,
                        mx_uint shared_mem) {
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

int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, const mx_uint *shape,
                                 mx_uint ndim, int dtype, NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(shared_pid, shared_id, TShape(shape, shape + ndim), dtype);
  API_END();
}

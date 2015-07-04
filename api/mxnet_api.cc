/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxnet_api.cc
 * \brief C API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/narray.h>
#include <mxnet/api_registry.h>
#include "./mxnet_api.h"

// NOTE: all functions return 0 upon success
// consider add try/catch block for user error
// handling in the future
using namespace mxnet;

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN(); and finishes with API_END(); */
#define API_END() } catch(dmlc::Error &e) { return MXHandleException(e); } return 0;

/*!
 * \brief a helper function for error handling
 *  will set the last error to be str_set when it is not NULL
 * \param str_set the error to set
 * \return a pointer message to last error
 */
const char *MXSetGetLastError_(const char *str_set) {
  // use last_error to record last error
  static thread_local std::string last_error;
  if (str_set != NULL) {
    last_error = str_set;
  }
  return last_error.c_str();
}

/*! \brief return str message of the last error */
const char *MXGetLastError() {
  return MXSetGetLastError_(NULL);
}

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int MXHandleException(const dmlc::Error &e) {
  MXSetGetLastError_(e.what());
  return -1;
}

// NOTE: return value is added in API_END
int MXNArrayCreateNone(NArrayHandle *out) {
  API_BEGIN();
  *out = new NArray();
  API_END();
}

int MXNArrayCreateShareMem(mx_float *data,
                           mx_uint *shape,
                           mx_uint ndim,
                           NArrayHandle *out) {
  API_BEGIN();
  *out = new NArray(TBlob(data, TShape(shape, shape + ndim),
                          cpu::kDevMask), 0);
  API_END();
}

int MXNArrayCreate(const mx_uint *shape,
                   mx_uint ndim,
                   int dev_mask,
                   int dev_id,
                   int delay_alloc,
                   NArrayHandle *out) {
  API_BEGIN();
  *out = new NArray(TShape(shape, shape + ndim),
                    Context(dev_mask, dev_id),
                    delay_alloc != 0);
  API_END();
}

int MXNArrayWait(NArrayHandle handle) {
  API_BEGIN();
  static_cast<NArray*>(handle)->Wait();
  API_END();
}

int MXNArrayWaitAll() {
  API_BEGIN();
  DAGEngine::Get()->WaitForAll();
  API_END();
}

int MXNArrayFree(NArrayHandle handle) {
  API_BEGIN();
  delete static_cast<NArray*>(handle);
  API_END();
}

int MXNArrayGetShape(NArrayHandle handle,
                     mx_uint *out_dim,
                     const mx_uint **out_pdata) {
  API_BEGIN();
  NArray *arr = static_cast<NArray*>(handle);
  if (!arr->is_none()) {
    const TShape &s = arr->shape();
    *out_dim = s.ndim();
    *out_pdata = s.data();
  } else {
    *out_dim = 0;
  }
  API_END();
}

int MXNArrayGetData(NArrayHandle handle,
                    mx_float **out_pdata) {
  API_BEGIN();
  NArray *arr = static_cast<NArray*>(handle);
  if (!arr->is_none()) {
    CHECK(arr->ctx().dev_mask == cpu::kDevMask)
        << "MXNArrayGetData can only be called for NArray on CPU";
    const TBlob &b = arr->data();
    CHECK(b.CheckContiguous());
    *out_pdata = b.FlatTo2D<cpu, mx_float>().dptr_;
  } else {
    *out_pdata = nullptr;
  }
  API_END();
}

int MXNArrayGetContext(NArrayHandle handle,
                       int *out_dev_mask,
                       int *out_dev_id) {
  API_BEGIN();
  NArray *arr = static_cast<NArray*>(handle);
  if (!arr->is_none()) {
    const Context &ctx = arr->ctx();
    *out_dev_mask = ctx.dev_mask;
    *out_dev_id = ctx.dev_id;
  } else {
    *out_dev_mask = 0;
    *out_dev_id = 0;
  }
  API_END();
}

int MXListFunctions(mx_uint *out_size,
                    FunctionHandle **out_array) {
  API_BEGIN();
  auto &vec = FunctionRegistry::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (FunctionHandle*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXGetFunction(const char *name,
                  FunctionHandle *out) {
  API_BEGIN();
  *out = FunctionRegistry::Find(name);
  API_END();
}

int MXFuncGetName(FunctionHandle fun,
                  const char **out_name) {
  API_BEGIN();
  auto *f = static_cast<const FunctionRegistry::Entry *>(fun);
  *out_name = f->name.c_str();
  API_END();
}

int MXFuncDescribe(FunctionHandle fun,
                   mx_uint *num_use_vars,
                   mx_uint *num_scalars,
                   mx_uint *num_mutate_vars,
                   int *type_mask) {
  API_BEGIN();
  auto *f = static_cast<const FunctionRegistry::Entry *>(fun);
  *num_use_vars = f->num_use_vars;
  *num_scalars = f->num_scalars;
  *num_mutate_vars = f->num_mutate_vars;
  *type_mask = f->type_mask;
  API_END();
}

int MXFuncInvoke(FunctionHandle fun,
                 NArrayHandle *use_vars,
                 mx_float *scalar_args,
                 NArrayHandle *mutate_vars) {
  API_BEGIN();
  auto *f = static_cast<const FunctionRegistry::Entry *>(fun);
  (*f)((NArray**)(use_vars),  //  NOLINT(*)
       scalar_args,
       (NArray**)(mutate_vars));  // NOLINT(*)
  API_END();
}

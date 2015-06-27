#include <mxnet/base.h>
#include <mxnet/narray.h>
#include <mxnet/api_registry.h>
#include "./mxnet_api.h"

// NOTE: all functions return 0 upon success
// consider add try/catch block for user error
// handling in the future
using namespace mxnet;

// macro to guard beginning and end section of all functions
// every function starts with API_BEGIN(); and finishes with API_END();
#define API_BEGIN() try {
#define API_END() } catch(MXNetException &e) { return MXHandleException(e); } return 0;

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int MXHandleException(const MXNetException &e) {
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
                   NArrayHandle *out) {
  API_BEGIN();
  *out = new NArray(TShape(shape, shape + ndim),
                    Context(dev_mask, dev_id));
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
    // TODO: change to exception
    CHECK(arr->ctx().dev_mask != cpu::kDevMask);
    const TBlob &b = arr->data();
    CHECK(b.CheckContiguous());
    *out_pdata = b.FlatTo2D<cpu, mx_float>().dptr_;
  } else {
    *out_pdata = nullptr;
  }
  API_END();
}

int MXNArrayGetDevice(NArrayHandle handle,
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
  *out_array = (FunctionHandle*)(dmlc::BeginPtr(vec));
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

int MXFuncDescribeArgs(FunctionHandle fun,
                       mx_uint *num_use_vars,
                       mx_uint *num_scalars,
                       mx_uint *num_mutate_vars) {
  API_BEGIN();
  auto *f = static_cast<const FunctionRegistry::Entry *>(fun);
  *num_use_vars = f->num_use_vars;
  *num_scalars = f->num_scalars;
  *num_mutate_vars = f->num_mutate_vars;
  API_END();
}

int MXFuncInvoke(FunctionHandle fun,
                 NArrayHandle *use_vars,
                 mx_float *scalar_args,
                 NArrayHandle *mutate_vars) {
  API_BEGIN();
  auto *f = static_cast<const FunctionRegistry::Entry *>(fun);
  (*f)((NArray**)(use_vars),
       scalar_args,
       (NArray**)(mutate_vars));
  API_END();  
}

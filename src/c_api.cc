/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api.cc
 * \brief C API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/narray.h>
#include <mxnet/symbolic.h>
#include <mxnet/registry.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <vector>
#include <mutex>
#include <memory>

// macro hanlding for threadlocal variables
#ifdef __GNUC__
  #define MX_TREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
  #define  MX_TREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
  #define MX_TREAD_LOCAL __declspec(thread)
#endif

#ifndef MX_TREAD_LOCAL
#message("Warning: Threadlocal is not enabled");
#endif

using namespace mxnet;

/*! \brief entry to to easily hold returning information */
struct MXAPIThreadLocalEntry {
  /*! \brief holds last error message */
  std::string last_error;
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning shapes */
  std::vector<TShape> arg_shapes, out_shapes;
  /*! \brief result holder for returning shape dimensions */
  std::vector<mx_uint> arg_shape_ndim, out_shape_ndim;
  /*! \brief result holder for returning shape pointer */
  std::vector<const mx_uint*> arg_shape_data, out_shape_data;
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

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
class MXAPIThreadLocalStore {
 public:
  /*! \brief store return entry */
  typedef MXAPIThreadLocalEntry T;
  /*! \return get a thread local singleton */
  static T* Get() {
    static MX_TREAD_LOCAL T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
  }

 private:
  /*! \brief constructor */
  MXAPIThreadLocalStore() {}
  /*! \brief destructor */
  ~MXAPIThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static MXAPIThreadLocalStore *Singleton() {
    static MXAPIThreadLocalStore inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T *str) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
  }
  /*! \brief internal mutex */
  std::mutex mutex_;
  /*!\brief internal data */
  std::vector<T*> data_;
};

// NOTE: all functions return 0 upon success
// consider add try/catch block for user error
// handling in the future

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &e) { return MXHandleException(e); } return 0;
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &e) { Finalize; return MXHandleException(e); } return 0; // NOLINT(*)

/*! \brief return str message of the last error */
const char *MXGetLastError() {
  return MXAPIThreadLocalStore::Get()->last_error.c_str();
}

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int MXHandleException(const dmlc::Error &e) {
  MXAPIThreadLocalStore::Get()->last_error = e.what();
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
  auto &vec = Registry<NArrayFunctionEntry>::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (FunctionHandle*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXGetFunction(const char *name,
                  FunctionHandle *out) {
  API_BEGIN();
  *out = Registry<NArrayFunctionEntry>::Find(name);
  API_END();
}

int MXFuncGetName(FunctionHandle fun,
                  const char **out_name) {
  API_BEGIN();
  auto *f = static_cast<const NArrayFunctionEntry*>(fun);
  *out_name = f->name.c_str();
  API_END();
}

int MXFuncDescribe(FunctionHandle fun,
                   mx_uint *num_use_vars,
                   mx_uint *num_scalars,
                   mx_uint *num_mutate_vars,
                   int *type_mask) {
  API_BEGIN();
  auto *f = static_cast<const NArrayFunctionEntry*>(fun);
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
  auto *f = static_cast<const NArrayFunctionEntry*>(fun);
  (*f)((NArray**)(use_vars),  //  NOLINT(*)
       scalar_args,
       (NArray**)(mutate_vars));  //  NOLINT(*)
  API_END();
}

//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------

int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  API_BEGIN();
  auto &vec = Registry<OperatorPropertyEntry>::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (AtomicSymbolCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                const char **out) {
  API_BEGIN();
  OperatorPropertyEntry *e = static_cast<OperatorPropertyEntry *>(creator);
  *out = e->name.c_str();
  API_END();
}

int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                               int num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  Symbol *s = new Symbol();
  OperatorProperty *op = nullptr;

  API_BEGIN();
  OperatorPropertyEntry *e = static_cast<OperatorPropertyEntry *>(creator);
  op = (*e)();
  for (int i = 0; i < num_param; ++i) {
    op->SetParam(keys[i], vals[i]);
  }
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

int MXSymbolListReturns(SymbolHandle symbol,
                        mx_uint *out_size,
                        const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = std::move(s->ListReturns());
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
    succ = s->InferShape(&(ret->arg_shapes), &(ret->out_shapes));
  } else {
    std::unordered_map<std::string, TShape> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = TShape(arg_shape_data + arg_ind_ptr[i],
                               arg_shape_data + arg_ind_ptr[i+1]);
    }
    succ = s->InferShape(kwargs, &(ret->arg_shapes), &(ret->out_shapes));
  }
  if (succ) {
    MXAPIThreadLocalEntry::SetupShapeArrayReturn(
        ret->arg_shapes, &(ret->arg_shape_ndim), &(ret->arg_shape_data));
    MXAPIThreadLocalEntry::SetupShapeArrayReturn(
        ret->out_shapes, &(ret->out_shape_ndim), &(ret->out_shape_data));
    *in_shape_size = static_cast<mx_uint>(ret->arg_shapes.size());
    *in_shape_ndim = dmlc::BeginPtr(ret->arg_shape_ndim);
    *in_shape_data = dmlc::BeginPtr(ret->arg_shape_data);
    *out_shape_size = static_cast<mx_uint>(ret->out_shapes.size());
    *out_shape_ndim = dmlc::BeginPtr(ret->out_shape_ndim);
    *out_shape_data = dmlc::BeginPtr(ret->out_shape_data);
    *complete = 1;
  } else {
    *complete = 0;
  }
  API_END();
}

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
#include <mxnet/c_api.h>
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

/*! \brief symbol wrapper to easily hold returning information */
struct MXAPISymbolWrapper {
  /*! \brief the actual symbol */
  mxnet::Symbol sym;
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
};

/*!
 * \brief helper to store error message in threadlocal storage
 */
class MXAPIErrorMessageHelper {
 public:
  /*! \brief get a single instance out from */
  static MXAPIErrorMessageHelper *Get() {
    static MXAPIErrorMessageHelper inst;
    return &inst;
  }
  /*!
   * \brief a helper function for error handling
   *  will set the last error to be str_set when it is not NULL
   * \param str_set the error to set
   * \return a pointer message to last error
   */
  static const char *SetGetLastError(const char *str_set) {
    // use last_error to record last error
    static MX_TREAD_LOCAL std::string *last_error = NULL;
    if (last_error == NULL) {
      last_error = new std::string();
      Get()->RegisterDelete(last_error);
    }
    if (str_set != NULL) {
      *last_error = str_set;
    }
    return last_error->c_str();
  }

 private:
  /*! \brief constructor */
  MXAPIErrorMessageHelper() {}
  /*! \brief destructor */
  ~MXAPIErrorMessageHelper() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(std::string *str) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
  }
  /*! \brief internal mutex */
  std::mutex mutex_;
  /*!\brief internal data */
  std::vector<std::string*> data_;
};

// NOTE: all functions return 0 upon success
// consider add try/catch block for user error
// handling in the future
using namespace mxnet;

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
  return MXAPIErrorMessageHelper::SetGetLastError(NULL);
}

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
int MXHandleException(const dmlc::Error &e) {
  MXAPIErrorMessageHelper::SetGetLastError(e.what());
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
  auto &vec = Registry<AtomicSymbolEntry>::List();
  *out_size = static_cast<mx_uint>(vec.size());
  *out_array = (AtomicSymbolCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                const char **out) {
  API_BEGIN();
  AtomicSymbolEntry *e = static_cast<AtomicSymbolEntry *>(creator);
  *out = e->name.c_str();
  API_END();
}

int MXSymbolCreateFromAtomicSymbol(AtomicSymbolCreator creator,
                                   int num_param,
                                   const char **keys,
                                   const char **vals,
                                   SymbolHandle *out) {
  MXAPISymbolWrapper *s = new MXAPISymbolWrapper();
  AtomicSymbol *atomic_symbol = nullptr;

  API_BEGIN();
  AtomicSymbolEntry *e = static_cast<AtomicSymbolEntry *>(creator);
  atomic_symbol = (*e)();
  for (int i = 0; i < num_param; ++i) {
    atomic_symbol->SetParam(keys[i], vals[i]);
  }
  s->sym = Symbol::Create(atomic_symbol);
  *out = s;
  API_END_HANDLE_ERROR(delete s; delete atomic_symbol);
}

int MXSymbolCreateVariable(const char *name, SymbolHandle *out) {
  MXAPISymbolWrapper *s = new MXAPISymbolWrapper();
  API_BEGIN();
  s->sym = Symbol::CreateVariable(name);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateGroup(mx_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  MXAPISymbolWrapper *s = new MXAPISymbolWrapper();
  MXAPISymbolWrapper **sym_arr = (MXAPISymbolWrapper**)symbols; // NOLINT(*)
  API_BEGIN();
  std::vector<Symbol> syms;
  for (mx_uint i = 0; i < num_symbols; ++i) {
    syms.push_back(sym_arr[i]->sym);
  }
  s->sym = Symbol::CreateGroup(syms);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolFree(SymbolHandle symbol) {
  API_BEGIN();
  delete static_cast<MXAPISymbolWrapper*>(symbol);
  API_END();
}

int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  MXAPISymbolWrapper *s = new MXAPISymbolWrapper();

  API_BEGIN();
  s->sym = (static_cast<const MXAPISymbolWrapper*>(symbol)->sym).Copy();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolPrint(SymbolHandle symbol, const char **out_str) {
  MXAPISymbolWrapper *s = static_cast<MXAPISymbolWrapper*>(symbol);

  API_BEGIN();
  std::ostringstream os;
  (s->sym).Print(os);
  s->ret_str = os.str();
  *out_str = (s->ret_str).c_str();
  API_END();
}

int MXSymbolListArguments(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  MXAPISymbolWrapper *s = static_cast<MXAPISymbolWrapper*>(symbol);
  API_BEGIN();
  s->ret_vec_str = std::move((s->sym).ListArguments());
  s->ret_vec_charp.clear();
  for (size_t i = 0; i < s->ret_vec_str.size(); ++i) {
    s->ret_vec_charp.push_back(s->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<mx_uint>(s->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(s->ret_vec_charp);
  API_END();
}

int MXSymbolListReturns(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  MXAPISymbolWrapper *s = static_cast<MXAPISymbolWrapper*>(symbol);
  API_BEGIN();
  s->ret_vec_str = std::move((s->sym).ListReturns());
  s->ret_vec_charp.clear();
  for (size_t i = 0; i < s->ret_vec_str.size(); ++i) {
    s->ret_vec_charp.push_back(s->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<mx_uint>(s->ret_vec_charp.size());
  *out_str_array = dmlc::BeginPtr(s->ret_vec_charp);
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

  MXAPISymbolWrapper* s = static_cast<MXAPISymbolWrapper*>(sym);
  if (keys == nullptr && num_args != 0) {
    std::vector<Symbol> pos_args;
    for (mx_uint i = 0; i < num_args; ++i) {
      pos_args.push_back(((MXAPISymbolWrapper*)(args[i]))->sym);  //  NOLINT(*)
    }
    (s->sym).Compose(pos_args, s_name);
  } else {
    std::unordered_map<std::string, Symbol> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = ((MXAPISymbolWrapper*)(args[i]))->sym;  //  NOLINT(*)
    }
    (s->sym).Compose(kwargs, s_name);
  }
  API_END();
}

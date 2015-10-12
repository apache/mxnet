/*!
 *  Copyright (c) 2015 by Contributors
 * \file executor.h
 * \brief Rcpp Symbol of MXNet.
 */
#include <Rcpp.h>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./executor.h"
#include "./ndarray.h"
#include "./symbol.h"

namespace mxnet {
namespace R {

Executor::RObjectType Executor::UpdateArgArray(const Executor::RObjectType &exec,
                                               const Rcpp::List& array,
                                               bool match_name,
                                               bool skip_null) {
  RCHECK(Rcpp::is<Executor>(exec))
      << "Expected exec to be "<< Executor::TypeName();
  Executor::RObjectType ret = Executor::Move(exec);
  UpdateArray("arg.arrays", array, Executor::XPtr(ret)->arg_arrays_, match_name, skip_null);
  return ret;
}

Executor::RObjectType Executor::UpdateAuxArray(const Executor::RObjectType &exec,
                                               const Rcpp::List& array,
                                               bool match_name,
                                               bool skip_null) {
  RCHECK(Rcpp::is<Executor>(exec))
      << "Expected exec to be "<< Executor::TypeName();
  Executor::RObjectType ret = Executor::Move(exec);
  UpdateArray("aux.arrays", array, Executor::XPtr(ret)->aux_arrays_, match_name, skip_null);
  return ret;
}

void Executor::UpdateArray(const char* array_name,
                           const Rcpp::List& from,
                           Rcpp::List* to,
                           bool match_name,
                           bool skip_null) {
  if (!match_name) {
    RCHECK(from.size() == to->size())
        << "Update array list must contain names";
    for (size_t i = 0; i < from.size(); ++i) {
      if (to->at(i) != R_NilValue) {
        if (from[i] != R_NilValue) {
          NDArray::CopyFromTo(*NDArray::XPtr(from[i]), NDArray::XPtr(to->at(i)));
        } else {
          RCHECK(skip_null)
              << "Position " << i << " expected to be not NULL";
        }
      } else {
        RCHECK(from[i] == R_NilValue)
            << "Position " << i << " expected to be NULL";
      }
    }
  } else {
    RCHECK(HasName(from))
        << "match.name is set to TRUE, the input list must have names in all elements";
    std::vector<std::string> names = from.names();
    for (size_t i = 0; i < names.size(); ++i) {
      RCHECK(names[i].length() != 0)
          << "match.name is set to TRUE, the input list must have names in all elements";
      RCHECK(to->containsElementNamed(names[i].c_str()))
          << "cannot find key " << names[i] << " in the array " << array_name;
      int index = to->findName(names[i]);
      if (to->at(index) != R_NilValue) {
        if (from[i] != R_NilValue) {
          NDArray::CopyFromTo(*NDArray::XPtr(from[i]), NDArray::XPtr(to->at(index)));
        } else {
          RCHECK(skip_null)
              << "Element " << names[i] << " expected to be not NULL";
        }
      } else {
        RCHECK(from[i] == R_NilValue)
            << "Element " << names[i] << " expected to be NULL";
      }
    }
  }
}

Rcpp::List Executor::CloneArray(const Rcpp::List& src) {
  Rcpp::List ret(src.size());
  ret.names() = src.names();
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] != R_NilValue) {
      RCHECK(Rcpp::is<NDArray>(src[i]))
          << "Expected exec to be "<< Executor::TypeName();
      ret[i] = NDArray::XPtr(src[i])->Clone();
    } else {
      ret[i] = R_NilValue;
    }
  }
  return ret;
}

Executor::RObjectType Executor::Forward(const RObjectType &exec,
                                        bool is_train,
                                        const Rcpp::List& kwargs) {
  // TODO(tqchen, KK) support kwargs to copy in additional data
  RCHECK(Rcpp::is<Executor>(exec))
      << "Expect exec to be " << Executor::TypeName();
  Executor::RObjectType ret = Executor::Move(exec);
  MX_CALL(MXExecutorForward(Executor::XPtr(ret)->handle_, is_train));
  return ret;
}

Executor::RObjectType Executor::Backward(const RObjectType &exec,
                                         const Rcpp::List &output_grads) {
  RCHECK(Rcpp::is<Executor>(exec))
      << "Expect exec to be " << Executor::TypeName();
  RCHECK(Executor::XPtr(exec)->grad_arrays_ != nullptr)
      << "This executor has not been binded with req.grad";
  Executor::RObjectType ret = Executor::Move(exec);
  std::vector<NDArrayHandle> grad_handles
      = NDArray::GetHandles(output_grads, "output_grads", false);
  MX_CALL(MXExecutorBackward(Executor::XPtr(ret)->handle_,
                             static_cast<mx_uint>(grad_handles.size()),
                             dmlc::BeginPtr(grad_handles)));
  return ret;
}


inline Rcpp::List* CreateArrayList(const Rcpp::List& source_array,
                                   const std::string& key,
                                   const std::vector<std::string>& names,
                                   const Context::RObjectType& ctx,
                                   std::vector<NDArrayHandle>* handles) {
  Rcpp::List* ret = new Rcpp::List(source_array.size());
  try {
    ret->names() = names;
    handles->resize(source_array.size());
    for (size_t i = 0; i < source_array.size(); ++i) {
      RCHECK(Rcpp::is<NDArray>(source_array[i]))
          << "Expect input " << key << " to be list of " << NDArray::TypeName();
      ret->at(i) = NDArray::Empty(NDArray::XPtr(source_array[i])->shape(), ctx);
      handles->at(i) = NDArray::XPtr(ret->at(i))->handle();
      NDArray::CopyFromTo(*NDArray::XPtr(source_array[i]),
                          NDArray::XPtr(ret->at(i)));
    }
  } catch(const Rcpp::exception& ex) {
    delete ret;
    throw ex;
  }
  return ret;
}

inline Rcpp::List* CreateGradList(const Rcpp::List& source_array,
                                  const Rcpp::List& grad_reqs,
                                  const std::vector<std::string>& names,
                                  const Context::RObjectType& ctx,
                                  std::vector<NDArrayHandle> *handles,
                                  std::vector<mx_uint> *grad_req_type) {
  Rcpp::List* ret = new Rcpp::List(grad_reqs.size(), R_NilValue);
  try {
    ret->names() = names;
    handles->resize(grad_reqs.size(), nullptr);
    grad_req_type->resize(grad_reqs.size(), 0);

    for (size_t i = 0; i < grad_reqs.size(); ++i) {
      RCHECK(Rcpp::is<bool>(grad_reqs[i]))
          << "Expect input grad_reqs to be list of booleans";
      if (Rcpp::as<bool>(grad_reqs[i])) {
        ret->at(i) = NDArray::Empty(NDArray::XPtr(source_array[i])->shape(), ctx);
        handles->at(i) = NDArray::XPtr(ret->at(i))->handle();
        grad_req_type->at(i) = 1;
      }
    }
  } catch(const Rcpp::exception& ex) {
    delete ret;
    throw ex;
  }
  return ret;
}

inline Rcpp::List* CreateOutList(mx_uint out_size,
                                 NDArrayHandle *out_arr,
                                 const std::vector<std::string>& names) {
  Rcpp::List* ret = new Rcpp::List(out_size);
  try {
    ret->names() = names;
    for (size_t i = 0; i < out_size; ++i) {
      ret->at(i) = NDArray::RObject(out_arr[i]);
    }
  } catch(const Rcpp::exception& ex) {
    delete ret;
    throw ex;
  }
  return ret;
}

Executor::RObjectType Executor::Bind(const Symbol::RObjectType& symbol,
                                     const Context::RObjectType& context,
                                     const Rcpp::List& arg_arrays,
                                     const Rcpp::List& aux_arrays,
                                     const Rcpp::List& grad_reqs) {
  Executor* exec = new Executor();
  try {
    Symbol *sym = Symbol::XPtr(symbol);
    // handles
    std::vector<mx_uint> grad_req_type;
    std::vector<NDArrayHandle> arg_handles, grad_handles, aux_handles;
    // for failure handling
    exec->arg_arrays_ = CreateArrayList(
        arg_arrays, "arg_arrays",
        sym->ListArguments(),
        context, &arg_handles);
    exec->aux_arrays_ = CreateArrayList(
        aux_arrays, "aux_arrays",
        sym->ListAuxiliaryStates(),
        context, &aux_handles);
    exec->grad_arrays_ = CreateGradList(
        arg_arrays, grad_reqs,
        sym->ListArguments(),
        context, &grad_handles, &grad_req_type);
    Context ctx(context);
    MX_CALL(MXExecutorBind(
        sym->handle_,
        ctx.dev_type, ctx.dev_id,
        static_cast<mx_uint>(arg_handles.size()), dmlc::BeginPtr(arg_handles),
        dmlc::BeginPtr(grad_handles), dmlc::BeginPtr(grad_req_type),
        static_cast<mx_uint>(aux_handles.size()), dmlc::BeginPtr(aux_handles),
        &(exec->handle_)));
    mx_uint out_size;
    NDArrayHandle *out_arr;
    MX_CALL(MXExecutorOutputs(exec->handle_, &out_size, &out_arr));
    exec->out_arrays_ = CreateOutList(
        out_size, out_arr, sym->ListOuputs());
  } catch(const Rcpp::exception& ex) {
    delete exec;
    throw ex;
  }
  return Rcpp::internal::make_new_object(exec);
}
void Executor::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<Executor>("MXExecutor")
      .finalizer(&Executor::Finalizer)
      .property("ref.arg.arrays", &Executor::arg_arrays)
      .property("ref.grad.arrays", &Executor::grad_arrays)
      .property("arg.arrays", &Executor::GetArgArrays)
      .property("grad.arrays", &Executor::GetGradArrays)
      .property("aux.arrays", &Executor::GetAuxArrays)
      .property("outputs", &Executor::GetOuputArrays);

  // As we aim for updated = mx.exec.set.arguments(old, arg_array)
  // And old is moved and shouldno longer be used.
  // The reason why setter/getter is not used,
  // is because setter did not implement copy-on-write, so this can be un-natural to R-user
  // We use a update function to do
  // new_object = update(old_object, param) instead

  function("mx.exec.update.arg.arrays",
           &Executor::UpdateArgArray,
           List::create(_["exec"], _["array"],
                        _["match.name"] = false, _["skip.null"] = false),
           "Update arguments array of executor, this will move the original executor");
  function("mx.exec.update.aux.arrays",
           &Executor::UpdateAuxArray,
           List::create(_["exec"], _["array"],
                        _["match.name"] = false, _["skip.null"] = false),
           "Update auxilary states array of executor, this will move the original executor");

  function("mx.exec.forward",
           &Executor::Forward,
           List::create(_["exec"], _["is.train"] = false, _["kwargs"] = Rcpp::List()),
           "Peform a forward operation on exec, this will set the outputs.");
  function("mx.exec.backward",
           &Executor::Backward,
           List::create(_["exec"], _["output.grads"] = Rcpp::List()),
           "Peform a backward operation on exec, this will set the gradients requested.");
  function("mx.symbol.bind",
           &Executor::Bind,
           List::create(_["symbol"], _["ctx"],
                        _["arg.arrays"], _["aux.arrays"], _["grad.reqs"]),
           "Bind the symbol on argument arrays, generate gradient array according to grad_reqs");
}

}  // namespace R
}  // namespace mxnet

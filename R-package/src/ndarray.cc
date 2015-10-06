#include <Rcpp.h>
#include "./base.h"
#include "./ndarray.h"

namespace mxnet {
namespace R {  // NOLINT(*)

void NDArray::Save(const Rcpp::RObject &sxptr,
                   const std::string& filename) {
  // TODO(KK) add constant instead of integer
  if (TYPEOF(sxptr) == 19) {
    Rcpp::List data_lst(sxptr);
    std::vector<std::string> lst_names = data_lst.names();
    size_t num_args = data_lst.size();
    std::vector<NDArrayHandle> handles(num_args);
    std::vector<const char*> keys(num_args);

    for (int i = 0 ; i < data_lst.size(); ++i) {
      keys[i] = lst_names[i].c_str();
      handles[i] = NDArray::XPtr(data_lst[i])->handle_;
    }
    MX_CALL(MXNDArraySave(filename.c_str(), num_args,
                          dmlc::BeginPtr(handles),
                          dmlc::BeginPtr(keys)));
  } else if (TYPEOF(sxptr) == 22) {
    MX_CALL(MXNDArraySave(filename.c_str(), 1,
                          &(NDArray::XPtr(sxptr)->handle_), nullptr));
  } else {
    RLOG_FATAL << "only NDArray or list of NDArray" << std::endl;
  }
}

Rcpp::List NDArray::Load(const std::string& filename) {
  mx_uint out_size;
  NDArrayHandle* out_arr;
  mx_uint out_name_size;
  const char** out_names;
  MX_CALL(MXNDArrayLoad(filename.c_str(),
                        &out_size, &out_arr,
                        &out_name_size, &out_names));
  Rcpp::List out(out_size);
  for (mx_uint i = 0; i < out_size; ++i) {
    out[i] = NDArray::RObject(out_arr[i]);
  }
  if (out_name_size != 0) {
    std::vector<std::string> lst_names(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      lst_names[i] = out_names[i];
    }
    out.attr("names") = lst_names;
  }
  return out;
}

NDArray::RObjectType NDArray::Empty(
    const Rcpp::Dimension& rshape,
    const Context::RObjectType& rctx) {
  std::vector<mx_uint> shape(rshape.size());
  for (size_t i = 0; i < rshape.size(); ++i){
    shape[i] = static_cast<mx_uint>(rshape[i]);
  }
  Context ctx(rctx);
  NDArrayHandle handle;
  MX_CALL(MXNDArrayCreate(dmlc::BeginPtr(shape),
                          static_cast<mx_uint>(shape.size()),
                          ctx.dev_type, ctx.dev_id, false, &handle));
  return NDArray::RObject(handle);
}

NDArrayFunction::NDArrayFunction(FunctionHandle handle)
    : handle_(handle) {
  // initialize the docstring
  {
    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;
    MX_CALL(MXFuncGetInfo(handle, &name, &description, &num_args,
                          &arg_names, &arg_type_infos, &arg_descriptions));
    // set function name
    name_ = name;
    // dostring: generate python style for now, change to R style later
    std::ostringstream os;
    os << description << "\n\n"
       << "Parameters\n"
       << "----------\n";
    for (mx_uint i = 0; i < num_args; ++i) {
      os << "    " << arg_names[i] << " : "  << arg_type_infos[i] << "\n"
         << "        " << arg_descriptions[i] << "\n";
    }
    os << "Returns\n"
       << "-------\n"
       << "out : NDArray\n"
       << "    The output result of the function";
    // set the dostring
    this->docstring = os.str();
  }
  // initialize the function information
  {
    const int kNDArrayArgBeforeScalar = 1;
    const int kAcceptEmptyMutateTarget = 1 << 2;
    int type_mask;
    MX_CALL(MXFuncDescribe(
        handle, &num_use_vars_, &num_scalars_,
        &num_mutate_vars_, &type_mask));
    if ((type_mask & kNDArrayArgBeforeScalar) != 0) {
      begin_use_vars_ = 0;
      begin_scalars_ = num_use_vars_;
    } else {
      begin_scalars_ = num_scalars_;
      begin_scalars_ = 0;
    }
    num_args_ = num_use_vars_ + num_scalars_;
    accept_empty_out_ = ((type_mask & kAcceptEmptyMutateTarget) != 0);
  }
}

SEXP NDArrayFunction::operator() (SEXP* args) {
  BEGIN_RCPP;
  RCHECK(accept_empty_out_)
      << "not yet support mutate target";
  NDArrayHandle ohandle;
  MX_CALL(MXNDArrayCreateNone(&ohandle));
  std::vector<mx_float> scalars(num_scalars_);
  std::vector<NDArrayHandle> use_vars(num_use_vars_);

  for (mx_uint i = 0; i < num_scalars_; ++i) {
    // better to use Rcpp cast?
    scalars[i] = (REAL)(args[begin_scalars_ + i])[0];
  }
  for (mx_uint i = 0; i < num_use_vars_; ++i) {
    use_vars[i] = NDArray::XPtr(args[begin_use_vars_ + i])->handle_;
  }
  MX_CALL(MXFuncInvoke(handle_,
                       dmlc::BeginPtr(use_vars),
                       dmlc::BeginPtr(scalars),
                       &ohandle));
  return NDArray::RObject(ohandle);
  END_RCPP;
}

// register normal function.
void NDArray::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  function("mx.nd.load", &NDArray::Load);
  function("mx.nd.save", &NDArray::Save);
}

void NDArrayFunction::InitRcppModule() {
  Rcpp::Module* scope = ::getCurrentScope();
  RCHECK(scope != nullptr)
      << "Init Module need to be called inside scope";
  mx_uint out_size;
  FunctionHandle *arr;
  MX_CALL(MXListFunctions(&out_size, &arr));
  for (int i = 0; i < out_size; ++i) {
    NDArrayFunction *f = new NDArrayFunction(arr[i]);
    scope->Add(f->get_name(), f);
  }
}
}  // namespace Rcpp
}  // namespace mxnet

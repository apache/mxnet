/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief Rcpp NDArray of MXNet.
 */
#include <Rcpp.h>
#include "./base.h"
#include "./io.h"
#include "./ndarray.h"

namespace mxnet {
namespace R {

void MXDataIter::Reset() {
  MX_CALL(MXDataIterBeforeFirst(handle_));
}

bool MXDataIter::Next() {
  int ret;
  MX_CALL(MXDataIterNext(handle_, &ret));
  return ret != 0;
}

int MXDataIter::NumPad() const {
  int pad;
  MX_CALL(MXDataIterGetPadNum(handle_, &pad));
  return pad;
}

Rcpp::List MXDataIter::Value() const {
  NDArrayHandle data, label;
  MX_CALL(MXDataIterGetData(handle_, &data));
  MX_CALL(MXDataIterGetLabel(handle_, &label));
  return Rcpp::List::create(
      Rcpp::Named("data") = NDArray::RObject(data, false),
      Rcpp::Named("label") = NDArray::RObject(label, false));
}

DataIterCreateFunction::DataIterCreateFunction
(DataIterCreator handle)
    : handle_(handle) {
  const char* name;
  const char* description;
  mx_uint num_args;
  const char **arg_names;
  const char **arg_type_infos;
  const char **arg_descriptions;
  const char *key_var_num_args;

  MX_CALL(MXDataIterGetIterInfo(
      handle_, &name, &description, &num_args,
      &arg_names, &arg_type_infos, &arg_descriptions));

  if (name[0] == '_') {
    name_ = std::string("mx.varg.io.internal.") + (name + 1);
  } else {
    name_ = std::string("mx.varg.io.") + name;
  }
  std::ostringstream os;
  os << description << "\n\n"
     << "Parameters\n"
     << "----------\n"
     << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
     << "Returns\n"
     << "-------\n"
     << "out : data iter\n"
     << "    The resulting data iter.";
  this->docstring = os.str();
}

SEXP DataIterCreateFunction::operator() (SEXP* args) {
  BEGIN_RCPP;
  Rcpp::List kwargs(args[0]);
  std::vector<std::string> keys = SafeGetListNames(kwargs);
  std::vector<std::string> str_keys(keys.size());
  std::vector<std::string> str_vals(keys.size());
  for (size_t i = 0; i < kwargs.size(); ++i) {
    RCHECK(keys[i].length() != 0)
        << name_ << " only accept key=value style arguments";
    str_keys[i] = FormatParamKey(keys[i]);
    str_vals[i] = toPyString(keys[i], kwargs[i]);
  }
  DataIterHandle out;
  std::vector<const char*> c_str_keys = CKeys(str_keys);
  std::vector<const char*> c_str_vals = CKeys(str_vals);

  MX_CALL(MXDataIterCreateIter(
      handle_, static_cast<mx_uint>(str_keys.size()),
      dmlc::BeginPtr(c_str_keys),
      dmlc::BeginPtr(c_str_vals),
      &out));
  return MXDataIter::RObject(out);
  END_RCPP;
}

void DataIter::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<DataIter>("MXDataIter")
      .method("iter.next", &DataIter::Next)
      .method("reset", &DataIter::Reset)
      .method("value", &DataIter::Value)
      .method("num.pad", &DataIter::NumPad);

  class_<MXDataIter>("MXNativeDataIter")
      .derives<DataIter>("MXDataIter")
      .finalizer(&MXDataIter::Finalizer);

  class_<ArrayDataIter>("MXArrayDataIter")
      .derives<DataIter>("MXDataIter");
}

void DataIterCreateFunction::InitRcppModule() {
  Rcpp::Module* scope = ::getCurrentScope();
  RCHECK(scope != nullptr)
      << "Init Module need to be called inside scope";
  mx_uint out_size;
  DataIterCreator *arr;
  MX_CALL(MXListDataIters(&out_size, &arr));
  for (int i = 0; i < out_size; ++i) {
    DataIterCreateFunction *f = new DataIterCreateFunction(arr[i]);
    scope->Add(f->get_name(), f);
  }
}
}  // namespace R
}  // namespace mxnet

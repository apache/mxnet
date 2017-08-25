/*!
 *  Copyright (c) 2015 by Contributors
 * \file io.cc
 * \brief Rcpp IO module of mxnet.
 */
#include <Rcpp.h>
#include <cstring>
#include "./base.h"
#include "./io.h"
#include "./export.h"
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

ArrayDataIter::ArrayDataIter(const Rcpp::NumericVector& data,
                             const Rcpp::NumericVector& label,
                             const Rcpp::NumericVector& unif_rnds,
                             int batch_size,
                             bool shuffle) : counter_(0) {
  Rcpp::IntegerVector dshape = data.attr("dim");
  Rcpp::IntegerVector lshape = label.attr("dim");
  if (dshape[dshape.size() - 1] != lshape[lshape.size() - 1]) {
    if (dshape[0] == lshape[0]) {
      RLOG_FATAL << "Seems X, y was passed in a Row major way, "
                 << "MXNetR adopts a column major convention.\n"
                 << "Please pass in transpose of X instead";
    } else {
      RLOG_FATAL << "Data and label shape in-consistent";
    }
  }
  num_data = lshape[lshape.size() - 1];
  std::vector<size_t> order(num_data);
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }

  if (shuffle) {
    RCHECK(unif_rnds.size() == num_data);
    for (size_t i = order.size() - 1; i != 0; --i) {
      size_t idx = static_cast<size_t>(unif_rnds[i] * (i + 1));
      if (idx < i) {
        std::swap(order[i], order[idx]);
      }
    }
  }
  ArrayDataIter::Convert(data, order, batch_size, &data_);
  ArrayDataIter::Convert(label, order, batch_size, &label_);
  num_pad_ = (batch_size - (order.size() % batch_size)) % batch_size;
  RCHECK(label_.size() == data_.size())
      << "Datasize not consistent";
}

void ArrayDataIter::Convert(const Rcpp::NumericVector& src,
                            const std::vector<size_t>& order,
                            size_t batch_size,
                            std::vector<NDArray> *out) {
  Rcpp::RObject dim = src.attr("dim");
  Rcpp::Dimension rshape(dim);
  size_t ndim = rshape.size();
  std::vector<mx_float> temp(src.size()), batch;
  std::copy(src.begin(), src.end(), temp.begin());
  out->clear();
  out->reserve(rshape[ndim - 1] / batch_size + 1);
  size_t line_size = 1;
  for (size_t i = 0; i < rshape.size() - 1; ++i) {
    line_size *= rshape[i];
  }
  rshape[ndim - 1] = batch_size;
  batch.resize(batch_size * line_size, 0.0f);

  for (size_t begin = 0; begin < order.size(); begin += batch_size) {
    size_t end = std::min(order.size(), begin + batch_size);
    for (size_t i = begin; i < end; ++i) {
      std::memcpy(&batch[(i - begin) * line_size],
                  &temp[order[i] * line_size],
                  sizeof(mx_float) * line_size);
    }
    NDArray::RObjectType ret = NDArray::Empty(rshape, Context::CPU());
    MX_CALL(MXNDArraySyncCopyFromCPU(
        NDArray(ret)->handle,
        dmlc::BeginPtr(batch), batch.size()));
    out->push_back(NDArray(ret));
  }
}

Rcpp::List ArrayDataIter::Value() const {
  RCHECK(counter_ != 0 && counter_ <= num_data)
      << "Read Iter at end or before iter.next is called";
  return Rcpp::List::create(
      Rcpp::Named("data") = data_[counter_ - 1].RObject(),
      Rcpp::Named("label") = label_[counter_ - 1].RObject());
}

bool ArrayDataIter::Next() {
  if (counter_ < data_.size()) {
    ++counter_; return true;
  } else {
    return false;
  }
}

int ArrayDataIter::NumPad() const {
  if (counter_ == label_.size()) {
    return static_cast<int>(num_pad_);
  } else {
    return 0;
  }
}

Rcpp::RObject ArrayDataIter::Create(const Rcpp::NumericVector& data,
                                    const Rcpp::NumericVector& label,
                                    const Rcpp::NumericVector& unif_rnds,
                                    int batch_size,
                                    bool shuffle) {
  return Rcpp::internal::make_new_object(
      new ArrayDataIter(data, label, unif_rnds, batch_size, shuffle));
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
     << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
     << "@return iter The result mx.dataiter\n\n"
     << "@export\n";
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
      .derives<DataIter>("MXDataIter");

  class_<ArrayDataIter>("MXArrayDataIter")
      .derives<DataIter>("MXDataIter");

  function("mx.io.internal.arrayiter", &ArrayDataIter::Create);
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

/*!
 *  Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief Rcpp NDArray of MXNet.
 */
#include <Rcpp.h>
#include <string>
#include <vector>
#include "./base.h"
#include "./kvstore.h"
#include "./ndarray.h"

namespace mxnet {
namespace R {

void KVStore::Init(const std::vector<int>& keys, const Rcpp::List& weights) {
  RCHECK(keys.size() == weights.size())
      << "The length of keys should be same as length of weights";
  std::vector<NDArrayHandle> handles = NDArray::GetHandles(weights, "weights");
  MX_CALL(MXKVStoreInit(
      handle_, static_cast<mx_uint>(handles.size()),
      dmlc::BeginPtr(keys), dmlc::BeginPtr(handles)));
}

void KVStore::Push(const std::vector<int>& keys,
                   const Rcpp::List& weight_lists,
                   const std::vector<int>& priority) {
  RCHECK(keys.size() == priority.size() || priority.size() == 0)
      << "The length of keys should be same as length of priority";

  std::vector<std::vector<NDArrayHandle> > vec(weight_lists.size());
  for (size_t i = 0; i < weight_lists.size(); ++i) {
    RCHECK(Rcpp::is<Rcpp::List>(weight_lists[i]))
        << "Expect weight_lists to be list(list(ndarray))";
    Rcpp::List list = Rcpp::as<Rcpp::List>(weight_lists[i]);
    RCHECK(list.size() == keys.size())
        << "Expect length of keys to be same as each weight_list";
    vec[i] = NDArray::GetHandles(list, "weight_list");
  }
  // do push
  std::vector<int> group_keys(vec.size());
  std::vector<NDArrayHandle> vals(vec.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    for (size_t j = 0; j < vec.size(); ++j) {
      vals[j] = vec[j][i];
    }
    std::fill(group_keys.begin(), group_keys.end(), keys[i]);
    MX_CALL(MXKVStorePush(handle_,
                          static_cast<mx_uint>(vals.size()),
                          dmlc::BeginPtr(group_keys),
                          dmlc::BeginPtr(vals),
                          priority.size() == 0 ? 0 : priority[i]));
  }
}

void KVStore::Pull(const std::vector<int>& keys,
                   const Rcpp::List& out_lists,
                   const std::vector<int>& priority) {
  RCHECK(keys.size() == priority.size() || priority.size() == 0)
      << "The length of keys should be same as length of priority";
  std::vector<std::vector<NDArrayHandle> > vec(out_lists.size());
  for (size_t i = 0; i < out_lists.size(); ++i) {
    RCHECK(Rcpp::is<Rcpp::List>(out_lists[i]))
        << "Expect out_lists to be list(list(ndarray))";
    Rcpp::List src = Rcpp::as<Rcpp::List>(out_lists[i]);
    RCHECK(src.size() == keys.size())
        << "Expect length of keys to be same as each out_lists";
    vec[i] = NDArray::GetHandles(src, "out_list");
  }
  // do pull
  std::vector<int> group_keys(vec.size());
  std::vector<NDArrayHandle> vals(vec.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    for (size_t j = 0; j < vec.size(); ++j) {
      vals[j] = vec[j][i];
    }
    std::fill(group_keys.begin(), group_keys.end(), keys[i]);
    MX_CALL(MXKVStorePull(handle_, static_cast<mx_uint>(vals.size()),
                          dmlc::BeginPtr(group_keys),
                          dmlc::BeginPtr(vals),
                          priority.size() == 0 ? 0 : priority[i]));
  }
}

std::string KVStore::type() const {
  const char* stype;
  MX_CALL(MXKVStoreGetType(handle_, &stype));
  return std::string(stype);
}

bool KVStore::update_on_kvstore() const {
  std::string type = this->type();
  return type != "local_allreduce_cpu" && type != "local_allreduce_device";
}

extern "C" void KVUpdaterCallback(int key, NDArrayHandle recv, NDArrayHandle local, void* handle) {
  NDArray weight(local, true), grad(recv, true);
  static_cast<KVStore*>(handle)->Update(key, grad, &weight);
}

void KVStore::SetOptimizer(const Rcpp::List& optimizer) {
  std::vector<std::string> names = optimizer.names();
  RCHECK(names.size() == 2 &&
         names[0] == "create.state" &&
         names[1] == "update")
      << "Invalid optimizer";
  fcreate_state_ = optimizer[0];
  fupdate_ = optimizer[1];
  optimizer_set_ = true;
  MX_CALL(MXKVStoreSetUpdater(handle_,
                              KVUpdaterCallback,
                              this));
}

Rcpp::List KVStore::CreateState(int index, const NDArray& weight) const {
  RCHECK(optimizer_set_)
      << "Need to call set.optimizer for KVStore " << type();
  // Use R Internal API here
  Rcpp::Shield<SEXP> call(Rf_lang3(fcreate_state_, Rcpp::wrap(index), weight.RObject()));
  SEXP ret = Rcpp_eval(call);
  if (Rf_isNull(ret)) {
    return Rcpp::List::create();
  } else if (TYPEOF(ret) == EXTPTRSXP) {
    return Rcpp::List::create(Rcpp::Named("state") = ret);
  } else {
    return ret;
  }
}

void KVStore::Update(int index, const NDArray& grad, NDArray *weight) {
  RCHECK(optimizer_set_)
      << "Need to call set.optimizer for KVStore " << type();
  std::map<int, Rcpp::List>::iterator it = states_.find(index);
  Rcpp::List state_lst = this->CreateState(index, *weight);
  if (it == states_.end()) {
    if (state_lst.size() != 0) {
      states_.insert(std::make_pair(index, state_lst));
      it = states_.find(index);
    }
  }

  Rcpp::List rlist;
  if (state_lst.size() == 0) {
    Rcpp::Shield<SEXP> call(Rf_lang5(fupdate_, Rcpp::wrap(index),
                                     weight->RObject(), grad.RObject(),
                                     R_NilValue));
    rlist = Rcpp_eval(call);
  } else if (state_lst.size() == 1) {
    Rcpp::Shield<SEXP> call(Rf_lang5(fupdate_, Rcpp::wrap(index),
                                     weight->RObject(), grad.RObject(),
                                     it->second[0]));
    rlist = Rcpp_eval(call);
  } else {
    // Use R Internal API here
    Rcpp::Shield<SEXP> call(Rf_lang5(fupdate_, Rcpp::wrap(index),
                                     weight->RObject(), grad.RObject(),
                                     it->second));
    rlist = Rcpp_eval(call);
  }
  NDArray::CopyFromTo(NDArray::FromRObject(rlist["weight"]), weight);
}


Rcpp::RObject KVStore::Create(const char *type) {
  KVStoreHandle handle;
  MX_CALL(MXKVStoreCreate(type, &handle));
  return Rcpp::internal::make_new_object(new KVStore(handle));
}

void KVStore::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<KVStore>("MXKVStore")
      .method("init", &KVStore::Init)
      .method("push", &KVStore::Push)
      .method("pull", &KVStore::Pull)
      .method("set.optimizer", &KVStore::SetOptimizer)
      .property("type", &KVStore::type)
      .property("update.on.kvstore", &KVStore::update_on_kvstore);

  function("mx.kv.create", &KVStore::Create,
           List::create(_["type"] = "local"),
           "Create a new kvstore");
}
}  // namespace R
}  // namespace mxnet

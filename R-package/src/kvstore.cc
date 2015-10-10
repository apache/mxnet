/*!
 *  Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief Rcpp NDArray of MXNet.
 */
#include <Rcpp.h>
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

Rcpp::List KVStore::Pull(const std::vector<int>& keys,
                         const Rcpp::List& out_lists,
                         const std::vector<int>& priority) {
  RCHECK(keys.size() == priority.size() || priority.size() == 0)
      << "The length of keys should be same as length of priority";
  Rcpp::List moved_list(out_lists.size());
  std::vector<std::vector<NDArrayHandle> > vec(out_lists.size());
  for (size_t i = 0; i < out_lists.size(); ++i) {
    RCHECK(Rcpp::is<Rcpp::List>(out_lists[i]))
        << "Expect out_lists to be list(list(ndarray))";
    Rcpp::List src = Rcpp::as<Rcpp::List>(out_lists[i]);
    RCHECK(src.size() == keys.size())
        << "Expect length of keys to be same as each out_lists";
    vec[i] = NDArray::GetHandles(src, "out_list");
    Rcpp::List moved(src.size());
    for (size_t j = 0; j < src.size(); ++j) {
      moved[j] = NDArray::Move(src[j]);
    }
    moved_list[i] = moved;
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
  return moved_list;
}

Rcpp::RObject KVStore::Create(const char *type) {
  KVStoreHandle handle;
  MX_CALL(MXKVStoreCreate(type, &handle));
  return Rcpp::internal::make_new_object(new KVStore(handle));
}

void KVStore::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<KVStore>("MXKVStore")
      .finalizer(&KVStore::Finalizer)
      .method("init", &KVStore::Init)
      .method("push", &KVStore::Push)
      .method("pull", &KVStore::Pull);

  function("mx.kv.create", &KVStore::Create,
           List::create(_["type"] = "local"),
           "Create a new kvstore");
}
}  // namespace R
}  // namespace mxnet

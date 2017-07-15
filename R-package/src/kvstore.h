/*!
 *  Copyright (c) 2015 by Contributors
 * \file kvstore.h
 * \brief Rcpp Parameter Store interface of MXNet
 */
#ifndef MXNET_RCPP_KVSTORE_H_
#define MXNET_RCPP_KVSTORE_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include <vector>
#include <map>
#include "./base.h"

namespace mxnet {
namespace R {
/*!
 * \brief MXNet's Parameter store interface.
 */
class KVStore {
 public:
  /*!
   * \brief initialize all the weights
   * \param keys The keys of each weight.
   * \param weights the weights NDArray list.
   */
  void Init(const std::vector<int>& keys, const Rcpp::List& weights);
  /*!
   * \brief Push the weights to the KVStore.
   *
   *  This operation will do a aggregation first on weight_lists, the push things out.
   *
   *  sum_list[i] = sum(list[i] for list in weight_lists)
   *  Then push(keys[i], sum_list[i]) for each i.
   *
   * \param keys list of keys, corresponds to key of each location.
   * \param weight_lists List of Rcpp::List.
   * \param priority The priority of each key.
   */
  void Push(const std::vector<int>& keys,
            const Rcpp::List& weight_lists,
            const std::vector<int>& priority);
  /*!
   * \brief Pull the data back.
   *  This operation will MUTATE the content of out_lists.
   *
   * \param keys List of keys, corresponds to key of each location.
   * \param out_lists List of Rcpp::List
   * \param priority The priority of each key.
   * \return The result list of pull.
   */
  void Pull(const std::vector<int>& keys,
            const Rcpp::List& out_lists,
            const std::vector<int>& priority);
  /*! \return The type of KVStore */
  std::string type() const;
  /*! \brief Whether to perform update on KVStore */
  bool update_on_kvstore() const;
  /*! \brief Setup optimizer */
  void SetOptimizer(const Rcpp::List& optimizer);
  // update function
  void Update(int index, const NDArray& grad, NDArray *weight);
  /*!
   * \brief create a KVStore
   * \return the created KVStore
   */
  static Rcpp::RObject Create(const char *type);
  /*! \brief initialize the R cpp Module */
  static void InitRcppModule();
  // destructor
  ~KVStore() {
    MX_CALL(MXKVStoreFree(handle_));
  }

 private:
  explicit KVStore(KVStoreHandle handle)
      : handle_(handle), optimizer_set_(false) {}
  // the internal callback to kvstore. This might return NULL
  Rcpp::List CreateState(int index, const NDArray& weight) const;
  /*! \brief internal KVStore handle */
  KVStoreHandle handle_;
  /*! \brief Whether optimizer is setted*/
  bool optimizer_set_;
  /*! \brief The internal state */
  std::map<int, Rcpp::List> states_;
  /*! \brief Function to create state */
  Rcpp::RObject fcreate_state_;
  /*! \brief Function to perform update */
  Rcpp::RObject fupdate_;
};

}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_KVSTORE_H_

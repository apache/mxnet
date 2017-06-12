/*!
*  Copyright (c) 2016 by Contributors
* \file executor.h
* \brief executor definition
* \author Chuntao Hong, Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"

namespace mxnet {
namespace cpp {

class Optimizer;

/*!
* \brief Executor interface
*/
class Executor {
  friend class Monitor;
 public:
  Executor(const Symbol &symbol, Context context,
           const std::vector<NDArray> &arg_arrays,
           const std::vector<NDArray> &grad_arrays,
           const std::vector<OpReqType> &grad_reqs,
           const std::vector<NDArray> &aux_arrays,
           const std::map<std::string, Context> &group_to_ctx =
               std::map<std::string, Context>(),
           Executor *shared_exec = nullptr);
  explicit Executor(const ExecutorHandle &h) { handle_ = h; }
  /*!
  * \brief Perform a Forward operation of Operator
  *  After this operation, user can get the result by using function head.
  */
  void Forward(bool is_train) {
    MXExecutorForward(handle_, is_train ? 1 : 0);
    mx_uint out_size;
    NDArrayHandle *out_array;
    CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
    for (mx_uint i = 0; i < out_size; ++i) {
      outputs[i] = NDArray(out_array[i]);
    }
  }
  /*!
  * \brief Perform a Backward operation of the Operator.
  *  This must be called after Forward.
  *  After this operation, NDArrays specified by grad_in_args_store will be
  *updated accordingly.
  *  User is allowed to pass in an empty Array if the head node is
  *  loss function and head gradeitn is not needed.
  *
  * \param head_grads the gradient of head nodes to be backproped.
  */
  void Backward(const std::vector<NDArray> &head_grads =
                    std::vector<NDArray>()) {
    std::vector<NDArrayHandle> head_grads_;
    for (auto d : head_grads) {
      head_grads_.push_back(d.GetHandle());
    }
    if (head_grads_.size() > 0) {
      MXExecutorBackward(handle_, head_grads_.size(), head_grads_.data());
    } else {
      MXExecutorBackward(handle_, 0, nullptr);
    }
  }
  // TODO(zhangchen-qinyinghua)
  // To implement reshape function
  void Reshape();
  /*!
  * \brief update the arguments with given learning rate and optimizer
  * \return the SymbolHandle
  */
  std::string DebugStr();
  /*!
  * \brief update the arguments with given learning rate and optimizer
  * \param opt the pointer to the optimizer
  * \param lr learning rate
  * \param wd weight decay
  * \param arg_update_begin begin index of the arguments to be updated, it
  * starts after the input data by default
  * \param arg_update_end end index of the arguments to be updated, it ends
  * before the label data by default
  */
  void UpdateAll(Optimizer *opt, float lr, float wd, int arg_update_begin = 1,
                 int arg_update_end = -1);
  /*!
  * \brief destructor, free the handle
  */
  ~Executor() { MXExecutorFree(handle_); }
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<NDArray> aux_arrays;
  /*!
  * \brief arrays store the outputs of forward
  */
  std::vector<NDArray> outputs;
  std::map<std::string, NDArray> arg_dict() {
    return GetDict(symbol_.ListArguments(), arg_arrays);
  }
  std::map<std::string, NDArray> grad_dict() {
    return GetDict(symbol_.ListArguments(), grad_arrays);
  }
  std::map<std::string, NDArray> aux_dict() {
    return GetDict(symbol_.ListAuxiliaryStates(), aux_arrays);
  }

 private:
  Executor(const Executor &e);
  Executor &operator=(const Executor &e);
  ExecutorHandle handle_;
  Symbol symbol_;
  std::map<std::string, NDArray> GetDict(const std::vector<std::string> &names,
                                         const std::vector<NDArray> &arrays) {
    std::map<std::string, NDArray> ret;
    std::set<std::string> name_set;
    for (const auto &s : names) {
      CHECK(name_set.find(s) == name_set.end()) << "Duplicate names detected, "
                                                << s;
      name_set.insert(s);
    }
    CHECK_EQ(name_set.size(), arrays.size())
        << "names size not equal to arrays size";
    for (size_t i = 0; i < names.size(); ++i) {
      ret[names[i]] = arrays[i];
    }
    return ret;
  }
};
}  // namespace cpp
}  // namespace mxnet
#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_H_

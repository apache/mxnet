/*!
*  Copyright (c) 2016 by Contributors
* \file model.h
* \brief MXNET.cpp model module
* \author Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_MODEL_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_MODEL_H_

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

struct FeedForwardConfig {
  Symbol symbol;
  std::vector<Context> ctx = {Context::cpu()};
  int num_epoch = 0;
  int epoch_size = 0;
  std::string optimizer = "sgd";
  // TODO(zhangchen-qinyinghua) More implement
  // initializer=Uniform(0.01),
  // numpy_batch_size=128,
  // arg_params=None, aux_params=None,
  // allow_extra_params=False,
  // begin_epoch=0,
  // **kwargs):
  FeedForwardConfig(const FeedForwardConfig &other) {}
  FeedForwardConfig() {}
};
class FeedForward {
 public:
  explicit FeedForward(const FeedForwardConfig &conf) : conf_(conf) {}
  void Predict();
  void Score();
  void Fit();
  void Save();
  void Load();
  static FeedForward Create();

 private:
  void InitParams();
  void InitPredictor();
  void InitIter();
  void InitEvalIter();
  FeedForwardConfig conf_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_MODEL_H_


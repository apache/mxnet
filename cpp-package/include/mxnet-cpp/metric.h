/*!
*  Copyright (c) 2016 by Contributors
* \file base.h
* \brief metrics defined
* \author Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_METRIC_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_METRIC_H_

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "mxnet-cpp/ndarray.h"
#include "dmlc/logging.h"

namespace mxnet {
namespace cpp {

class EvalMetric {
 public:
  explicit EvalMetric(const std::string& name, int num = 0)
      : name(name), num(num) {}
  virtual void Update(NDArray labels, NDArray preds) = 0;
  void Reset() {
    num_inst = 0;
    sum_metric = 0.0f;
  }
  float Get() { return sum_metric / num_inst; }
  void GetNameValue();

 protected:
  std::string name;
  int num;
  float sum_metric = 0.0f;
  int num_inst = 0;

  static bool CheckLabelShapes(NDArray labels, NDArray preds,
                               Shape shape = Shape(0)) {
    // TODO(zhangchen-qinyinghua)
    // inplement this
    return true;
  }
};

class Accuracy : public EvalMetric {
 public:
  Accuracy() : EvalMetric("accuracy") {}

  void Update(NDArray labels, NDArray preds) {
    CHECK_EQ(labels.GetShape().size(), 1);
    mx_uint len = labels.GetShape()[0];
    std::vector<mx_float> pred_data(len);
    std::vector<mx_float> label_data(len);
    preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, len);
    labels.SyncCopyToCPU(&label_data, len);
    NDArray::WaitAll();
    for (mx_uint i = 0; i < len; ++i) {
      sum_metric += (pred_data[i] == label_data[i]) ? 1 : 0;
      num_inst += 1;
    }
  }
};

class LogLoss : public EvalMetric {
 public:
  LogLoss() : EvalMetric("logloss") {}

  void Update(NDArray labels, NDArray preds) {
    static const float epsilon = 1e-15;
    mx_uint len = labels.GetShape()[0];
    mx_uint m = preds.GetShape()[1];
    std::vector<mx_float> pred_data(len * m);
    std::vector<mx_float> label_data(len);
    preds.SyncCopyToCPU(&pred_data, pred_data.size());
    labels.SyncCopyToCPU(&label_data, len);
    NDArray::WaitAll();
    for (mx_uint i = 0; i < len; ++i) {
      sum_metric +=
          -std::log(std::max(pred_data[i * m + label_data[i]], epsilon));
      num_inst += 1;
    }
  }
};

}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_METRIC_H_


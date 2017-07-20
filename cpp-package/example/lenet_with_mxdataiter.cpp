/*!
 * Copyright (c) 2016 by Contributors
 */
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
#include "../include/mxnet-cpp/op.h"

using namespace std;
using namespace mxnet::cpp;

Symbol LenetSymbol() {
  /*
   * LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
   * "Gradient-based learning applied to document recognition."
   * Proceedings of the IEEE (1998)
   * */

  /*define the symbolic net*/
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");
  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
  Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

  Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
  Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::kTanh);
  Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
  Symbol tanh2 = Activation("tanh2", conv2, ActivationActType::kTanh);
  Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol flatten = Flatten("flatten", pool2);
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
  Symbol tanh3 = Activation("tanh3", fc1, ActivationActType::kTanh);
  Symbol fc2 = FullyConnected("fc2", tanh3, fc2_w, fc2_b, 10);

  Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

  return lenet;
}

int main(int argc, char const *argv[]) {
  /*setup basic configs*/
  int W = 28;
  int H = 28;
  int batch_size = 128;
  int max_epoch = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto lenet = LenetSymbol();
  std::map<string, NDArray> args_map;

  args_map["data"] = NDArray(Shape(batch_size, 1, W, H), Context::gpu());
  args_map["data_label"] = NDArray(Shape(batch_size), Context::gpu());
  lenet.InferArgsMap(Context::gpu(), &args_map, args_map);

  args_map["fc1_w"] = NDArray(Shape(500, 4 * 4 * 50), Context::gpu());
  NDArray::SampleGaussian(0, 1, &args_map["fc1_w"]);
  args_map["fc2_b"] = NDArray(Shape(10), Context::gpu());
  args_map["fc2_b"] = 0;

  auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./train-images-idx3-ubyte")
      .SetParam("label", "./train-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 1)
      .SetParam("flat", 0)
      .CreateDataIter();
  auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./t10k-images-idx3-ubyte")
      .SetParam("label", "./t10k-labels-idx1-ubyte")
      .CreateDataIter();

  Optimizer* opt = OptimizerRegistry::Find("ccsgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0)
     ->SetParam("clip_gradient", 10);

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(Context::gpu());
      args_map["data_label"] = data_batch.label.Copy(Context::gpu());
      NDArray::WaitAll();
      auto *exec = lenet.SimpleBind(Context::gpu(), args_map);
      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(opt, learning_rate, weight_decay);
      delete exec;
    }

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(Context::gpu());
      args_map["data_label"] = data_batch.label.Copy(Context::gpu());
      NDArray::WaitAll();
      auto *exec = lenet.SimpleBind(Context::gpu(), args_map);
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
      delete exec;
    }
    LG << "Accuracy: " << acu.Get();
  }
  MXNotifyShutdown();
  return 0;
}

/*!
 * Copyright (c) 2016 by Contributors
 */
#include <string>
#include <vector>
#include <map>

#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
#include "../include/mxnet-cpp/op.h"

using namespace mxnet::cpp;

Symbol ConvFactory(Symbol data, int num_filter,
                   Shape kernel,
                   Shape stride = Shape(1, 1),
                   Shape pad = Shape(0, 0),
                   const std::string & name = "",
                   const std::string & suffix = "") {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  return Activation("relu_" + name + suffix, conv, "relu");
}

Symbol InceptionFactory(Symbol data, int num_1x1, int num_3x3red,
                        int num_3x3, int num_d5x5red, int num_d5x5,
                        PoolingPoolType pool, int proj, const std::string & name) {
  Symbol c1x1 = ConvFactory(data, num_1x1, Shape(1, 1),
                            Shape(1, 1), Shape(0, 0), name + "_1x1");

  Symbol c3x3r = ConvFactory(data, num_3x3red, Shape(1, 1),
                             Shape(1, 1), Shape(0, 0), name + "_3x3", "_reduce");

  Symbol c3x3 = ConvFactory(c3x3r, num_3x3, Shape(3, 3),
                            Shape(1, 1), Shape(1, 1), name + "_3x3");

  Symbol cd5x5r = ConvFactory(data, num_d5x5red, Shape(1, 1),
                              Shape(1, 1), Shape(0, 0), name + "_5x5", "_reduce");

  Symbol cd5x5 = ConvFactory(cd5x5r, num_d5x5, Shape(5, 5),
                             Shape(1, 1), Shape(2, 2), name + "_5x5");

  Symbol pooling = Pooling(name + "_pool", data, Shape(3, 3), pool,
                           false, false, PoolingPoolingConvention::kValid,
                           Shape(1, 1), Shape(1, 1));

  Symbol cproj = ConvFactory(pooling, proj, Shape(1, 1),
                             Shape(1, 1), Shape(0, 0), name + "_proj");

  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd5x5);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol GoogleNetSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol conv2 = ConvFactory(pool1, 64, Shape(1, 1), Shape(1, 1),
                             Shape(0, 0), "conv2");
  Symbol conv3 = ConvFactory(conv2, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv3");
  Symbol pool3 = Pooling("pool3", conv3, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, PoolingPoolType::kMax, 32, "in3a");
  Symbol in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, PoolingPoolType::kMax, 64, "in3b");
  Symbol pool4 = Pooling("pool4", in3b, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, PoolingPoolType::kMax, 64, "in4a");
  Symbol in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, PoolingPoolType::kMax, 64, "in4b");
  Symbol in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, PoolingPoolType::kMax, 64, "in4c");
  Symbol in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, PoolingPoolType::kMax, 64, "in4d");
  Symbol in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, PoolingPoolType::kMax, 128, "in4e");
  Symbol pool5 = Pooling("pool5", in4e, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, PoolingPoolType::kMax, 128, "in5a");
  Symbol in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, PoolingPoolType::kMax, 128, "in5b");
  Symbol pool6 = Pooling("pool6", in5b, Shape(7, 7), PoolingPoolType::kAvg,
                         false, false, PoolingPoolingConvention::kValid, Shape(1, 1));

  Symbol flatten = Flatten("flatten", pool6);

  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, num_classes);

  return SoftmaxOutput("softmax", fc1, data_label);
}

int main(int argc, char const *argv[]) {
  int batch_size = 50;
  int max_epoch = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto googlenet = GoogleNetSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  args_map["data"] = NDArray(Shape(batch_size, 3, 256, 256), Context::gpu());
  args_map["data_label"] = NDArray(Shape(batch_size), Context::gpu());
  googlenet.InferArgsMap(Context::gpu(), &args_map, args_map);

  auto train_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./train.lst")
      .SetParam("path_imgrec", "./train.rec")
      .SetParam("data_shape", Shape(3, 256, 256))
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 1)
      .CreateDataIter();

  auto val_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./val.lst")
      .SetParam("path_imgrec", "./_val.rec")
      .SetParam("data_shape", Shape(3, 256, 256))
      .SetParam("batch_size", batch_size)
      .CreateDataIter();

  Optimizer* opt = OptimizerRegistry::Find("ccsgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10);

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      args_map["data"] = data_batch.data.Copy(Context::gpu());
      args_map["data_label"] = data_batch.label.Copy(Context::gpu());
      NDArray::WaitAll();
      auto *exec = googlenet.SimpleBind(Context::gpu(), args_map);
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
      auto *exec = googlenet.SimpleBind(Context::gpu(), args_map);
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

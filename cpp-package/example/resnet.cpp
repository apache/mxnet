/*!
 * Copyright (c) 2016 by Contributors
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
#include "../include/mxnet-cpp/op.h"

using namespace mxnet::cpp;

Symbol ConvolutionNoBias(const std::string& symbol_name,
                         Symbol data,
                         Symbol weight,
                         Shape kernel,
                         int num_filter,
                         Shape stride = Shape(1, 1),
                         Shape dilate = Shape(1, 1),
                         Shape pad = Shape(0, 0),
                         int num_group = 1,
                         int64_t workspace = 512) {
  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", dilate)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", true)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .CreateSymbol(symbol_name);
}

static const Symbol BN_BETA;
static const Symbol BN_GAMMA;

Symbol getConv(const std::string & name, Symbol data,
               int  num_filter,
               Shape kernel, Shape stride, Shape pad,
               bool with_relu,
               mx_float bn_momentum) {
  Symbol conv_w(name + "_w");
  Symbol conv = ConvolutionNoBias(name, data, conv_w,
                                  kernel, num_filter, stride, Shape(1, 1),
                                  pad, 1, 512);

  Symbol bn = BatchNorm(name + "_bn", conv, Symbol(), Symbol(), Symbol(),
                        Symbol(), 2e-5, bn_momentum, false);

  if (with_relu) {
    return Activation(name + "_relu", bn, "relu");
  } else {
    return bn;
  }
}

Symbol makeBlock(const std::string & name, Symbol data, int num_filter,
                 bool dim_match, mx_float bn_momentum) {
  Shape stride;
  if (dim_match) {
    stride = Shape(1, 1);
  } else {
    stride = Shape(2, 2);
  }

  Symbol conv1 = getConv(name + "_conv1", data, num_filter,
                         Shape(3, 3), stride, Shape(1, 1),
                         true, bn_momentum);

  Symbol conv2 = getConv(name + "_conv2", conv1, num_filter,
                         Shape(3, 3), Shape(1, 1), Shape(1, 1),
                         false, bn_momentum);

  Symbol shortcut;

  if (dim_match) {
    shortcut = data;
  } else {
    Symbol shortcut_w(name + "_proj_w");
    shortcut = ConvolutionNoBias(name + "_proj", data, shortcut_w,
                                 Shape(2, 2), num_filter,
                                 Shape(2, 2), Shape(1, 1), Shape(0, 0),
                                 1, 512);
  }

  Symbol fused = shortcut + conv2;
  return Activation(name + "_relu", fused, "relu");
}

Symbol getBody(Symbol data, int num_level, int num_block, int num_filter, mx_float bn_momentum) {
  for (int level = 0; level < num_level; level++) {
    for (int block = 0; block < num_block; block++) {
      data = makeBlock("level" + std::to_string(level + 1) + "_block" + std::to_string(block + 1),
                       data, num_filter * (std::pow(2, level)),
                       (level == 0 || block > 0), bn_momentum);
    }
  }
  return data;
}

Symbol ResNetSymbol(int num_class, int num_level = 3, int num_block = 9,
                    int num_filter = 16, mx_float bn_momentum = 0.9,
                    mxnet::cpp::Shape pool_kernel = mxnet::cpp::Shape(8, 8)) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol zscore = BatchNorm("zscore", data, Symbol(), Symbol(), Symbol(),
                            Symbol(), 0.001, bn_momentum);

  Symbol conv = getConv("conv0", zscore, num_filter,
                        Shape(3, 3), Shape(1, 1), Shape(1, 1),
                        true, bn_momentum);

  Symbol body = getBody(conv, num_level, num_block, num_filter, bn_momentum);

  Symbol pool = Pooling("pool", body, pool_kernel, PoolingPoolType::kAvg);

  Symbol flat = Flatten("flatten", pool);

  Symbol fc_w("fc_w"), fc_b("fc_b");
  Symbol fc = FullyConnected("fc", flat, fc_w, fc_b, num_class);

  return SoftmaxOutput("softmax", fc, data_label);
}

int main(int argc, char const *argv[]) {
  int batch_size = 50;
  int max_epoch = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto resnet = ResNetSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  args_map["data"] = NDArray(Shape(batch_size, 3, 256, 256), Context::gpu());
  args_map["data_label"] = NDArray(Shape(batch_size), Context::gpu());
  resnet.InferArgsMap(Context::gpu(), &args_map, args_map);

  auto train_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_train.lst")
      .SetParam("path_imgrec", "./sf1_train.rec")
      .SetParam("data_shape", Shape(3, 256, 256))
      .SetParam("batch_size", batch_size)
      .SetParam("shuffle", 1)
      .CreateDataIter();

  auto val_iter = MXDataIter("ImageRecordIter")
      .SetParam("path_imglist", "./sf1_val.lst")
      .SetParam("path_imgrec", "./sf1_val.rec")
      .SetParam("data_shape", Shape(3, 256, 256))
      .SetParam("batch_size", batch_size)
      .CreateDataIter();

  Optimizer* opt = OptimizerRegistry::Find("ccsgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10);

  auto *exec = resnet.SimpleBind(Context::gpu(), args_map);

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();
      exec->UpdateAll(opt, learning_rate, weight_decay);
      NDArray::WaitAll();
    }

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Accuracy: " << acu.Get();
  }
  delete exec;
  MXNotifyShutdown();
  return 0;
}

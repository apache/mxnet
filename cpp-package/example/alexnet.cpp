/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 */
#include <iostream>
#include <map>
#include <string>
#include "mxnet-cpp/MxNetCpp.h"


using namespace std;
using namespace mxnet::cpp;

Symbol AlexnetSymbol(int num_classes) {
  auto input_data = Symbol::Variable("data");
  auto target_label = Symbol::Variable("label");
  /*stage 1*/
  auto conv1 = Operator("Convolution")
                   .SetParam("kernel", Shape(11, 11))
                   .SetParam("num_filter", 96)
                   .SetParam("stride", Shape(4, 4))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(0, 0))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", input_data)
                   .CreateSymbol("conv1");
  auto relu1 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv1)
                   .CreateSymbol("relu1");
  auto pool1 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu1)
                   .CreateSymbol("pool1");
  auto lrn1 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool1)
                  .CreateSymbol("lrn1");
  /*stage 2*/
  auto conv2 = Operator("Convolution")
                   .SetParam("kernel", Shape(5, 5))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(2, 2))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn1)
                   .CreateSymbol("conv2");
  auto relu2 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv2)
                   .CreateSymbol("relu2");
  auto pool2 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu2)
                   .CreateSymbol("pool2");
  auto lrn2 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool2)
                  .CreateSymbol("lrn2");
  /*stage 3*/
  auto conv3 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn2)
                   .CreateSymbol("conv3");
  auto relu3 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv3)
                   .CreateSymbol("relu3");
  auto conv4 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu3)
                   .CreateSymbol("conv4");
  auto relu4 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv4)
                   .CreateSymbol("relu4");
  auto conv5 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu4)
                   .CreateSymbol("conv5");
  auto relu5 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", conv5)
                   .CreateSymbol("relu5");
  auto pool3 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max")
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu5)
                   .CreateSymbol("pool3");
  /*stage4*/
  auto flatten =
      Operator("Flatten").SetInput("data", pool3).CreateSymbol("flatten");
  auto fc1 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", flatten)
                 .CreateSymbol("fc1");
  auto relu6 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc1)
                   .CreateSymbol("relu6");
  auto dropout1 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu6)
                      .CreateSymbol("dropout1");
  /*stage5*/
  auto fc2 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout1)
                 .CreateSymbol("fc2");
  auto relu7 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc2)
                   .CreateSymbol("relu7");
  auto dropout2 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu7)
                      .CreateSymbol("dropout2");
  /*stage6*/
  auto fc3 = Operator("FullyConnected")
                 .SetParam("num_hidden", num_classes)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout2)
                 .CreateSymbol("fc3");
  auto softmax = Operator("SoftmaxOutput")
                     .SetParam("grad_scale", 1)
                     .SetParam("ignore_label", -1)
                     .SetParam("multi_output", false)
                     .SetParam("use_ignore", false)
                     .SetParam("normalization", "null") /*batch,null,valid */
                     .SetInput("data", fc3)
                     .SetInput("label", target_label)
                     .CreateSymbol("softmax");
  return softmax;
}

int main(int argc, char const *argv[]) {
  /*basic config*/
  int batch_size = 256;
  int max_epo = 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  /*context and net symbol*/
  auto ctx = Context::gpu();
  auto Net = AlexnetSymbol(10);

  /*args_map and aux_map is used for parameters' saving*/
  map<string, NDArray> args_map;
  map<string, NDArray> aux_map;

  /*we should tell mxnet the shape of data and label*/
  args_map["data"] = NDArray(Shape(batch_size, 3, 256, 256), ctx);
  args_map["label"] = NDArray(Shape(batch_size), ctx);

  /*with data and label, executor can be generated automatically*/
  auto *exec = Net.SimpleBind(ctx, args_map);
  auto arg_names = Net.ListArguments();
  aux_map = exec->aux_dict();
  args_map = exec->arg_dict();

  /*if fine tune from some pre-trained model, we should load the parameters*/
  // NDArray::Load("./model/alex_params_3", nullptr, &args_map);
  /*else, we should use initializer Xavier to init the params*/
  Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2.34);
  for (auto &arg : args_map) {
    /*be careful here, the arg's name must has some specific ends or starts for
     * initializer to call*/
    xavier(arg.first, &arg.second);
  }
  /*print out to check the shape of the net*/
  for (const auto &s : Net.ListArguments()) {
    LG << s;
    const auto &k = args_map[s].GetShape();
    for (const auto &i : k) {
      cout << i << " ";
    }
    cout << endl;
  }

  /*these binary files should be generated using im2rc tools, which can be found
   * in mxnet/bin*/
  auto train_iter = MXDataIter("ImageRecordIter")
                        .SetParam("path_imglist", "./data/train_rec.lst")
                        .SetParam("path_imgrec", "./data/train_rec.bin")
                        .SetParam("data_shape", Shape(3, 256, 256))
                        .SetParam("batch_size", batch_size)
                        .SetParam("shuffle", 1)
                        .CreateDataIter();
  auto val_iter = MXDataIter("ImageRecordIter")
                      .SetParam("path_imglist", "./data/val_rec.lst")
                      .SetParam("path_imgrec", "./data/val_rec.bin")
                      .SetParam("data_shape", Shape(3, 256, 256))
                      .SetParam("batch_size", batch_size)
                      .CreateDataIter();

  Optimizer* opt = OptimizerRegistry::Find("ccsgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);

  Accuracy acu_train, acu_val;
  LogLoss logloss_val;
  for (int iter = 0; iter < max_epo; ++iter) {
    LG << "Train Epoch: " << iter;
    /*reset the metric every epoch*/
    acu_train.Reset();
    /*reset the data iter every epoch*/
    train_iter.Reset();
    while (train_iter.Next()) {
      auto batch = train_iter.GetDataBatch();
      LG << train_iter.GetDataBatch().index.size();
      /*use copyto to feed new data and label to the executor*/
      batch.data.CopyTo(&args_map["data"]);
      batch.label.CopyTo(&args_map["label"]);
      exec->Forward(true);
      exec->Backward();
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }

      NDArray::WaitAll();
      acu_train.Update(batch.label, exec->outputs[0]);
    }
    LG << "ITER: " << iter << " Train Accuracy: " << acu_train.Get();

    LG << "Val Epoch: " << iter;
    acu_val.Reset();
    val_iter.Reset();
    logloss_val.Reset();
    while (val_iter.Next()) {
      auto batch = val_iter.GetDataBatch();
      LG << val_iter.GetDataBatch().index.size();
      batch.data.CopyTo(&args_map["data"]);
      batch.label.CopyTo(&args_map["label"]);
      exec->Forward(false);
      NDArray::WaitAll();
      acu_val.Update(batch.label, exec->outputs[0]);
      logloss_val.Update(batch.label, exec->outputs[0]);
    }
    LG << "ITER: " << iter << " Val Accuracy: " << acu_val.Get();
    LG << "ITER: " << iter << " Val LogLoss: " << logloss_val.Get();

    /*save the parameters*/
    stringstream ss;
    ss << iter;
    string iter_str;
    ss >> iter_str;
    string save_path_param = "./model/alex_param_" + iter_str;
    auto save_args = args_map;
    /*we do not want to save the data and label*/
    save_args.erase(save_args.find("data"));
    save_args.erase(save_args.find("label"));
    /*the alexnet does not get any aux array, so we do not need to save
     * aux_map*/
    LG << "ITER: " << iter << " Saving to..." << save_path_param;
    NDArray::Save(save_path_param, save_args);
  }
  /*don't foget to release the executor*/
  delete exec;
  MXNotifyShutdown();
  return 0;
}

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
 * Xin Li yakumolx@gmail.com
 */
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;

Symbol mlp(const vector<int> &layers) {
  auto x = Symbol::Variable("X");
  auto label = Symbol::Variable("label");

  vector<Symbol> weights(layers.size());
  vector<Symbol> biases(layers.size());
  vector<Symbol> outputs(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    weights[i] = Symbol::Variable("w" + to_string(i));
    biases[i] = Symbol::Variable("b" + to_string(i));
    Symbol fc = FullyConnected(
      i == 0? x : outputs[i-1],  // data
      weights[i],
      biases[i],
      layers[i]);
    outputs[i] = i == layers.size()-1? fc : Activation(fc, ActivationActType::kRelu);
  }

  return SoftmaxOutput(outputs.back(), label);
}

int main(int argc, char** argv) {
  const float MIN_SCORE = stof(argv[1]);

  const int image_size = 28;
  const vector<int> layers{128, 64, 10};
  const int batch_size = 100;
  const int max_epoch = 10;
  const float learning_rate = 0.1;
  const float weight_decay = 1e-2;

  auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./mnist_data/train-images-idx3-ubyte")
      .SetParam("label", "./mnist_data/train-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      .SetParam("flat", 1)
      .CreateDataIter();
  auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./mnist_data/t10k-images-idx3-ubyte")
      .SetParam("label", "./mnist_data/t10k-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      .SetParam("flat", 1)
      .CreateDataIter();

  auto net = mlp(layers);

  Context ctx = Context::gpu();  // Use GPU for training

  std::map<string, NDArray> args;
  args["X"] = NDArray(Shape(batch_size, image_size*image_size), ctx);
  args["label"] = NDArray(Shape(batch_size), ctx);
  // Let MXNet infer shapes of other parameters such as weights
  net.InferArgsMap(ctx, &args, args);

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = Uniform(0.01);
  for (auto& arg : args) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  // Create sgd optimizer
  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("rescale_grad", 1.0/batch_size)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);
  std::unique_ptr<LRScheduler> lr_sch(new FactorScheduler(5000, 0.1));
  opt->SetLRScheduler(std::move(lr_sch));

  // Create executor by binding parameters to the model
  auto *exec = net.SimpleBind(ctx, args);
  auto arg_names = net.ListArguments();

  float score = 0;
  // Start training
  for (int iter = 0; iter < max_epoch; ++iter) {
    int samples = 0;
    train_iter.Reset();

    auto tic = chrono::system_clock::now();
    while (train_iter.Next()) {
      samples += batch_size;
      auto data_batch = train_iter.GetDataBatch();
      // Data provided by DataIter are stored in memory, should be copied to GPU first.
      data_batch.data.CopyTo(&args["X"]);
      data_batch.label.CopyTo(&args["label"]);
      // CopyTo is imperative, need to wait for it to complete.
      NDArray::WaitAll();

      // Compute gradients
      exec->Forward(true);
      exec->Backward();
      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "X" || arg_names[i] == "label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
    }
    auto toc = chrono::system_clock::now();

    Accuracy acc;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args["X"]);
      data_batch.label.CopyTo(&args["label"]);
      NDArray::WaitAll();
      // Only forward pass is enough as no gradient is needed when evaluating
      exec->Forward(false);
      acc.Update(data_batch.label, exec->outputs[0]);
    }
    float duration = chrono::duration_cast<chrono::milliseconds>(toc - tic).count() / 1000.0;
    LG << "Epoch: " << iter << " " << samples/duration << " samples/sec Accuracy: " << acc.Get();
    score = acc.Get();
  }

  delete exec;
  MXNotifyShutdown();
  return score >= MIN_SCORE ? 0 : 1;
}

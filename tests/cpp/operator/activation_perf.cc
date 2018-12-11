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
 *  \file activation_perf.cc
 *  \brief Perf/profile run of ActivationOp
 *  \author Chris Olivier
 */

#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"
#include "../../src/operator/nn/activation-inl.h"

using namespace mxnet;

typedef std::vector<std::pair<std::string, std::string> > kwargs_t;
const kwargs_t basic_activation_args = { };

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(ACTIVATION_PERF, ExecuteBidirectional) {
  using namespace std;
  TShape shape({5, 5});
  vector<string> activations = {
    "relu",
    "sigmoid",
    "tanh",
    "softrelu",
    "softsign"
  };
  for (const string& activation : activations) {
    kwargs_t activation_args = {{"act_type", activation}};
    test::op::CoreOperatorRunner<float> runner;
    runner.RunBidirectional(false, { shape }, test::op::CoreOpExecutor<float>::ArgsWithOpName(
            activation_args, "Activation", "_backward_Activation"), 1);
  }
  for (const string& activation : activations) {
    kwargs_t activation_args = {{"act_type", activation}};
    test::op::CoreOperatorRunner<float> runner;
    runner.RunBidirectional(true, { shape }, test::op::CoreOpExecutor<float>::ArgsWithOpName(
            activation_args, "Activation", "_backward_Activation"), 1);
  }
}

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(ACTIVATION_PERF, TimingCPU) {
  kwargs_t kwargs = basic_activation_args;
  // Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"act_type", "tanh"});
  kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Activation",
                                                           "_backward_Activation");
  TShape shape({10, 10, 10, 10});
  test::op::CoreOperatorRunner<float> runner;
  runner.RunBidirectional(false, { shape }, kwargs, 1);

  std::vector <TShape> shapes;
  if (test::performance_run) {
    shapes = {
      {1,  1, 28,  28},
      {1,  3, 28,  28},
      {50, 1, 18,  32},
      {50, 3, 18,  32},
      {20, 3, 128, 128}
    };
  } else {
    shapes = {
      {1,  1, 28,  28},
      {50, 3, 18,  32},
    };
  }
  for (const TShape &shape : shapes) {
    runner.TimingTest("Activation Operator CPU", false, false, kwargs, 2, 10, { shape });
  }
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief ActivationOp timing test for GPU
 */
TEST(ACTIVATION_PERF, TimingGPU) {
  kwargs_t kwargs = basic_activation_args;
  // Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"act_type", "tanh"});
  kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Activation",
                                                           "_backward_Activation");
  TShape shape({10, 10, 10, 10});
  test::op::CoreOperatorRunner<float> runner;
  runner.RunBidirectional(true, { shape }, kwargs, 1);
  std::vector <TShape> shapes = {
      {1,  1, 28,  28},
      {1,  3, 28,  28},
      {50, 1, 18,  32},
      {50, 3, 18,  32},
      {20, 3, 128, 128}
    };
  for (const TShape &shape : shapes) {
    runner.TimingTest("Activation Operator GPU", true, false, kwargs, 2, 10, { shape });
  }
}
#endif  // MXNET_USE_CUDA == 1


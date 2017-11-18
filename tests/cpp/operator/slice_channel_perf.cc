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
#include "../include/test_legacy_op.h"
#include "../../src/operator/slice_channel-inl.h"

using namespace mxnet;

typedef std::vector<std::pair<std::string, std::string> > kwargs_t;
const kwargs_t basic_activation_args = { };

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(SLICE_CHANNEL_PERF, ExecuteBidirectional) {
  TShape shape({1, 160, 200});
  kwargs_t kwargs = basic_activation_args;
  kwargs.push_back({"num_outputs", "160"});
  test::op::LegacyOpRunner<mxnet::op::SliceChannelProp, float, float> runner;
  runner.RunBidirectional(false, { shape }, kwargs, 1);
}

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(SLICE_CHANNEL_PERF, TimingCPU) {
  kwargs_t kwargs = basic_activation_args;
  // Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"num_outputs", "160"});
  test::op::LegacyOpRunner<mxnet::op::SliceChannelProp, float, float> runner;
  runner.RunBidirectional(false,
                          { TShape({1, 160, 200}) },
                          kwargs, 1);  // prime code and cache
  std::vector <TShape> shapes;
  if (test::performance_run) {
    shapes = {
      {1, 160, 200},
      {10, 160, 200},
      {100, 160, 200},
      {10, 160, 500},
      {100, 160, 500}
    };
  } else {
    shapes = {
      {1, 160, 200},
      {1, 160, 200}
    };
  }
  for (const TShape &shape : shapes) {
    runner.TimingTest("SliceChannel Operator CPU", false, false, kwargs, 2, 10, { shape });
  }
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief ActivationOp timing test for GPU
 */
TEST(SLICE_CHANNEL_PERF, TimingGPU) {
  kwargs_t kwargs = basic_activation_args;
  // Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"num_outputs", "160"});
  test::OperatorRunner<mxnet::op::SliceChannelProp,
    test::op::LegacyOperatorExecutor<float, float>> runner;
  runner.RunBidirectional(true,
                          { TShape({1, 160, 200}) },
                          kwargs, 1);  // prime code and cache
  std::vector <TShape> shapes = {
      {1, 160, 200},
      {1, 160, 200},
      {1, 160, 200},
      {1, 160, 200},
      {1, 160, 200}
    };
  for (const TShape &shape : shapes) {
    runner.TimingTest("SliceChannel Operator GPU", true, false, kwargs, 2, 10, { shape });
  }
}
#endif  // MXNET_USE_CUDA == 1


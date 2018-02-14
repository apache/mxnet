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
 *  \file fully_conn_perf.cc
 *  \brief Sample for running C++ performance tests on a single operator.  This method is also
 *         useful for profiling with vtune or gprof, avoiding the "noise" of python and executor
 *  \author Chris Olivier
 */

#include <dmlc/logging.h>
#include <mxnet/tensor_blob.h>
#include <nnvm/tuple.h>
#include "../../src/operator/nn/fully_connected-inl.h"
#include "../include/test_op_runner.h"
#include "../include/test_legacy_op.h"

using namespace mxnet;

typedef std::vector<std::pair<std::string, std::string> > kwargs_t;

const kwargs_t basic_fullyconn_args = { {"num_hidden", "250"} };
/*!
 * \brief Generic bidirectional sanity test
 */
TEST(FULLY_CONNECTED, ExecuteBidirectionalFullyConnected) {
  TShape shape({5, 5});
  kwargs_t kwargs = basic_fullyconn_args;
  test::op::LegacyOpRunner<mxnet::op::FullyConnectedProp, float, float> runner;
  runner.RunBidirectional(false, { shape }, kwargs, 1);
}

/*!
 * \brief Timing test for CPU
 */
TEST(FULLY_CONNECTED, FullyConnectedTimingCPU) {
  kwargs_t kwargs = basic_fullyconn_args;
  test::op::LegacyOpRunner<mxnet::op::FullyConnectedProp, float, float> runner;
  runner.RunBidirectional(false,
                          { TShape({10, 10, 10, 10}) },
                          kwargs, 1);  // prime code and cache
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
  for (const TShape& shape : shapes) {
    runner.TimingTest("Fully connected CPU", false, false, kwargs, 2, 10, { shape });
  }
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief Timing test for GPU
 */
TEST(FULLY_CONNECTED, FullyConnectedTimingGPU) {
  kwargs_t kwargs = basic_fullyconn_args;
  test::OperatorRunner<mxnet::op::FullyConnectedProp,
    test::op::LegacyOperatorExecutor<float, float>>
    runner;
  runner.RunBidirectional(true,
                          { TShape({10, 10, 10, 10}) },
                          kwargs, 1);  // prime code and cache
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
  for (const TShape& shape : shapes) {
    runner.TimingTest("Fully connected GPU", true, false, kwargs, 2, 10, { shape });
  }
}
#endif  // MXNET_USE_CUDA == 1

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
#include "../include/test_core_op.h"

using namespace mxnet;

typedef std::vector<std::pair<std::string, std::string> > kwargs_t;

const kwargs_t basic_fullyconn_args = { {"num_hidden", "250"}, {"no_bias", "true"} };
/*!
 * \brief Generic bidirectional sanity test
 */
TEST(FULLY_CONNECTED, ExecuteBidirectionalFullyConnected) {
  TShape shape1({5, 5});
  TShape shape2({250, 5});
  kwargs_t kwargs = basic_fullyconn_args;
  test::op::CoreOperatorRunner<float> runner;
  runner.set_verbose(true);
  runner.RunGenericOperatorForward(false, { shape1, shape2 }, test::op::CoreOpExecutor<float>::ArgsWithOpName(
          kwargs, "FullyConnected", "_backward_FullyConnected"), 1);
}

/*!
 * \brief Timing test for CPU
 */
TEST(FULLY_CONNECTED, FullyConnectedTimingCPU) {
  kwargs_t kwargs = basic_fullyconn_args;
  TShape shape1({10, 10, 10, 10});
  TShape shape2({250, 1000});
  test::op::CoreOperatorRunner<float> runner;
  runner.RunGenericOperatorForward(false, { shape1, shape2 }, test::op::CoreOpExecutor<float>::ArgsWithOpName(
          kwargs, "FullyConnected", "_backward_FullyConnected"), 1);
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
    TShape shape2({250, shape.ProdShape(1, shape.ndim())});
    runner.TimingTest("Fully connected CPU", false, false, test::op::CoreOpExecutor<float>::ArgsWithOpName(
                      kwargs, "FullyConnected", "_backward_FullyConnected"), 2, 10, { shape, shape2 }, false);
  }
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief Timing test for GPU
 */
TEST(FULLY_CONNECTED, FullyConnectedTimingGPU) {
  kwargs_t kwargs = basic_fullyconn_args;
  TShape shape1({10, 10, 10, 10});
  TShape shape2({250, 1000});
  test::op::CoreOperatorRunner<float> runner;
  runner.RunGenericOperatorForward(true, { shape1, shape2 }, test::op::CoreOpExecutor<float>::ArgsWithOpName(
          kwargs, "FullyConnected", "_backward_FullyConnected"), 1);
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
    TShape shape2({250, shape.ProdShape(1, shape.ndim())});
    runner.TimingTest("Fully connected GPU", true, false, test::op::CoreOpExecutor<float>::ArgsWithOpName(
                      kwargs, "FullyConnected", "_backward_FullyConnected"), 2, 10, { shape, shape2 }, false);
  }
}
#endif  // MXNET_USE_CUDA == 1

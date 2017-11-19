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
 *  \file broadcast_perf.cc
 *  \brief Perf/profile run of broadcast kernel
 *  \author Chris Olivier
 */
#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"

using namespace mxnet;

using kwargs_t = test::op::kwargs_t;

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(BROADCAST_PERF, ExecuteBidirectional) {
  test::op::BasicRunCoreOpBidirectional(false, true, {},
                                        { {2, 3}, {2, 1} },
                                        "broadcast_add", "_backward_broadcast_add");
}

static const std::vector<std::vector<TShape>> broadcast_shapes() {
  std::vector<std::vector<TShape>> shapes;
  if (test::performance_run) {
    shapes = {
      { {28,  28},  {28, 1} },
      { {64,  28},  {1, 28} },
      { {28,  28, 28},  {28, 28, 1} },
      { {128, 128}, {1, 128} },
      { {1024, 12, 256}, {1024, 1, 1} },
      { {2560, 1280}, {2560, 1} }
    };
  } else {
    shapes = {
      // Non-performance dataset acts as a sanity test
      { {28,  28},  {28, 1} },
      { {128, 128}, {128, 1} },
      { {28,  28, 28},  {28, 28, 1} }
    };
  }
  return std::move(shapes);
}

template<typename DType = float>
static void RunCoreOpTimingTest(const bool isGPU,
                                const kwargs_t& op_kwargs,
                                const char *op_name,
                                const char *backward_op_name = "") {
  const kwargs_t kwargs = test::op::CoreOpExecutor<DType>::ArgsWithOpName(
    op_kwargs, op_name, backward_op_name);

  // prime code and cache before the performance runs
  test::op::CoreOperatorRunner<DType> runner;
  runner.RunBidirectional(false, { {2, 3}, {2, 1} }, kwargs, 1);

  // Do the performance runs
  std::vector<std::vector<TShape>> shapes = broadcast_shapes();
  const char *pu = isGPU ? "GPU" : "CPU";
  for (const std::vector<TShape> &shape : shapes) {
    runner.TimingTest(std::string(op_name) + " Operator " + pu, isGPU, false, kwargs,
                      2, 10, shape);
  }
}

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(BROADCAST_PERF, TimingCPU) {
  RunCoreOpTimingTest(false, {}, "broadcast_add", "_backward_broadcast_add");
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief ActivationOp timing test for GPU
 */
TEST(BROADCAST_PERF, TimingGPU) {
  RunCoreOpTimingTest(true, {}, "broadcast_add", "_backward_broadcast_add");
}
#endif  // MXNET_USE_CUDA == 1


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
#include "../../src/operator/nn/activation-inl.h"
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"

using namespace mxnet;

using kwargs_t = test::op::kwargs_t;

template<typename DType = float>
static void RunCoreOpBidirectional(const bool isGPU,
                                   const kwargs_t& op_kwargs,
                                   const char *op_name,
                                   const char *backward_op_name = "") {
  const TShape shape({5, 5});
  test::op::CoreOpExecutor<DType> op(isGPU, { shape });
  op.set_verbose(false);

  op.Init(op.ArgsWithOpName(op_kwargs, op_name, backward_op_name));

  PRINT_NDARRAYS(op.ctx().run_ctx, op.inputs());
  PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  op.Execute();
  PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  if (op.HasBackward()) {
    PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_inputs());
    PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
    op.ExecuteBackward();
    PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
  }
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
  runner.RunBidirectional(false, { {20, 3, 128, 128} }, kwargs, 1);

  // Do the performance runs
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
  const char *pu = isGPU ? "GPU" : "CPU";
  for (const TShape &shape : shapes) {
    runner.TimingTest(std::string(op_name) + " Operator " + pu, isGPU, false, kwargs,
                      2, 10, { shape });
  }
}

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(COREOP_PERF, ExecuteBidirectional) {
  std::cout << "NEGATIVE CLIP GRADIENT" << std::endl;
  RunCoreOpBidirectional(false, { {"lr", "0.01" }, { "clip_gradient", "-1" } },
                         "sgd_mom_update",
                         COREOP_BWD_OP_NAME_VALUE_NONE);
  std::cout << "POSITIVE CLIP GRADIENT" << std::endl;
  RunCoreOpBidirectional(false, { {"lr", "0.01" }, { "clip_gradient", "1" } },
                         "sgd_mom_update",
                         COREOP_BWD_OP_NAME_VALUE_NONE);
}

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(COREOP_PERF, TimingCPU) {
  std::cout << "NEGATIVE CLIP GRADIENT" << std::endl;
  RunCoreOpTimingTest(false, { {"lr", "0.01" }, { "clip_gradient", "-1" } },
                      "sgd_mom_update",
                      COREOP_BWD_OP_NAME_VALUE_NONE);
  std::cout << "POSITIVE CLIP GRADIENT" << std::endl;
  RunCoreOpTimingTest(false, { {"lr", "0.01" }, { "clip_gradient", "1" } },
                      "sgd_mom_update",
                      COREOP_BWD_OP_NAME_VALUE_NONE);
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief ActivationOp timing test for GPU
 */
TEST(COREOP_PERF, TimingGPU) {
  std::cout << "NEGATIVE CLIP GRADIENT" << std::endl;
  RunCoreOpTimingTest(true, { {"lr", "0.01" }, { "clip_gradient", "-1" } },
                      "sgd_mom_update",
                      COREOP_BWD_OP_NAME_VALUE_NONE);
  std::cout << "POSITIVE CLIP GRADIENT" << std::endl;
  RunCoreOpTimingTest(true, { {"lr", "0.01" }, { "clip_gradient", "1" } },
                      "sgd_mom_update",
                      COREOP_BWD_OP_NAME_VALUE_NONE);
}
#endif  // MXNET_USE_CUDA == 1


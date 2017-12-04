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
 * \file test_op_runner.h
 * \brief Run a generic operator
 * \author Chris Olivier
*/
#ifndef TEST_OP_RUNNER_H_
#define TEST_OP_RUNNER_H_

#include <string>
#include <vector>
#include <utility>
#include "./test_op.h"

namespace mxnet {
namespace test {

/*!
 * \brief Generic operator runner
 * \tparam OperatorProp property class for a given operator (i.e. FullyConnectedProp, BatchNormProp)
 * \tparam OperatorExecutor Data container for forward and backward passes for some given
 *         data types
 */
template<typename OperatorProp, typename OperatorExecutor>
class OperatorRunner {
 public:
  typedef typename OperatorExecutor::DataType    DType;

  OperatorRunner() {
#ifdef NDEBUG
    total_iterations_ = 50;
#else
    total_iterations_ = 5;
#endif
  }

  /*!
   * \brief Test operator forward pass
   * \param isGPU Whether this test is for GPU
   * \param inputShape Input data shape
   * \param kwargs Operator parameters
   * \param OutShapeFunction Output shape function override
   * \param count Number of times to run in each direction
   * \return OpInfo object for further opereator analysis
   */
  test::op::OpInfo<OperatorProp, OperatorExecutor>
  RunGenericOperatorForward(
    bool isGPU,
    const std::vector<TShape>& inputShapes,
    const std::vector<std::pair<std::string, std::string> > &kwargs,
    const size_t count = 1) {
#if MXNET_USE_CUDA
    if (isGPU && !test::unitTestsWithCuda) {
      LOG(INFO) << "GPU not found, running test as non-GPU";
    }
#else
    isGPU = false;
#endif
    test::op::OpInfo<OperatorProp, OperatorExecutor> info =
      test::op::createOpAndInfoF<OperatorProp, OperatorExecutor>(kwargs, isGPU, inputShapes);
    info.executor_->initForward(*info.prop_, &info.in_type_);
    info.executor_->forward(count);
    return info;
  }

  /*!
   * \brief Test operator backward pass
   * \param info OpInfo object from forward pass
   * \param count
   * \return OpInfo object for further opereator analysis
   */
  test::op::OpInfo<OperatorProp, OperatorExecutor> RunGenericOperatorBackward(
    test::op::OpInfo<OperatorProp, OperatorExecutor> *info,
    const size_t count = 1) {
    CHECK(info->executor_->HasBackward());
    info->executor_->initBackward(*info->prop_, &info->in_type_);
    info->executor_->backward(count);
    return *info;
  }

  /*!
   * \brief Run operator forward and backward
   * \param isGPU Whether this test is for GPU
   * \param inputShape Input data shape
   * \param kwargs Operator parameters
   * \param OutShapeFunction Output shape function override
   * \param count Number of times to run in each direction
   * \return
   */
  test::op::OpInfo<OperatorProp, OperatorExecutor> RunBidirectional(
    bool isGPU,
    const std::vector<TShape>& inputShapes,
    const std::vector<std::pair<std::string, std::string> > &kwargs,
    const size_t count = 1) {
    test::op::OpInfo<OperatorProp, OperatorExecutor> info =
      RunGenericOperatorForward(isGPU, inputShapes, kwargs, count);
    if (info.executor_->HasBackward()) {
      return RunGenericOperatorBackward(&info, count);
    }
    return info;
  }

  /*!
   * \brief Timing test a generic operator
   * \tparam PropType
   * \tparam DType Data type
   * \tparam AccReal Accumulative data type (if any)
   * \param label Label for performance output
   * \param isGPU Whether this test is for GPU
   * \param stochastic Whether shape should be random (batch size, channels, hm, w)
   * \param kwargs Operator parameters
   * \param dim Data dimensions
   * \param count Number of times to run in each direction
   */
  std::unordered_map<int, perf::TimingInstrument::Info>
  TimingTest(const std::string& label,
             const bool isGPU,
             const bool stochastic,
             const test::op::kwargs_t& kwargs,
             int dim = 0,
             size_t count = 1,
             const std::vector<TShape>& timing_shapes = {}) {
    if (mxnet::test::quick_test) {
      total_iterations_ = 2;
      count = 1;
    }

    test::perf::TimingInstrument timing;

    std::stringstream ss;
    ss << "Timing: " << total_iterations_ << " iterations of " << count << " calls";
    if (timing_shapes[0].ndim()) {
      size_t lhs_total = 0;
      ss << ", shape = ";
      for (size_t i = 0, n = timing_shapes.size(); i < n; ++i) {
        if (i) {
          ss << ", ";
        }
        ss << timing_shapes[i];
        if (!i) {
          lhs_total = timing_shapes[i].Size();
        }
      }
      ss << " = " << test::pretty_num(lhs_total) << " items " << std::endl << std::flush;
    }
    if (!mxnet::test::csv) {
      std::cout << ss.str();
    }

    for (size_t i = 0; i < total_iterations_; ++i) {
      index_t batchSize = 1;
      index_t channels = 1;
      index_t depth = 1;
      index_t height = 1;
      index_t width = 1;

      if (timing_shapes.empty()) {
        do {
          batchSize = stochastic ? test::rangedRand(1U, TEST_BATCH_SIZE * 2U) : TIMING_BATCH_SIZE;
          channels = stochastic ? test::rangedRand(1U, TEST_CHANNELS * 2U) : TIMING_CHANNELS;
          depth = stochastic ? test::rangedRand(1U, TEST_DEPTH * 2U) : TIMING_DEPTH;
          height = stochastic ? test::rangedRand(1U, TEST_DH * 2U) : TIMING_DH;
          width = stochastic ? test::rangedRand(1U, TEST_DW * 2U) : TIMING_DW;
        } while (stochastic && (height * width) == 1U);
      } else {
        dim = timing_shapes[0].ndim() - 1;
      }

      const size_t D = dim ? dim - 1U : test::rangedRand(0U, 2U);

      test::op::OpInfo<OperatorProp, OperatorExecutor> info;
      switch (D) {
        case 0:
          info = RunGenericOperatorForward(isGPU,
                                           !timing_shapes.empty()
                                           ? timing_shapes
                                           : std::vector<TShape>({TShape({batchSize,
                                                                          channels,
                                                                          width})}),
                                           kwargs,
                                           count);
          break;
        case 1:
          info = RunGenericOperatorForward(isGPU,
                                           !timing_shapes.empty()
                                           ? timing_shapes
                                           : std::vector<TShape>({ TShape({batchSize,
                                                                           channels,
                                                                           height,
                                                                           width})}),
                                           kwargs,
                                           count);
          break;
        case 2:
          info = RunGenericOperatorForward(isGPU,
                                           !timing_shapes.empty()
                                           ? timing_shapes
                                           : std::vector<TShape>({ TShape({batchSize,
                                                                           channels,
                                                                           depth,
                                                                           height,
                                                                           width})}),
                                           kwargs,
                                           count);
          break;
        default:
          CHECK(false) << "Unsupported dimension count: " << (D + 1);
      }
      if (info.executor_) {
        if (info.executor_->HasBackward()) {
          RunGenericOperatorBackward(&info, count);
        }
        timing += info.executor_->GetTiming();
      }
    }

    if (verbose_ && !mxnet::test::csv) {
      timing.print(&std::cout, label);
      std::cout << std::endl << std::flush;
    }
    return timing.data();
  }

  void set_verbose(bool verbose) { verbose_ = verbose; }

  void set_total_iterations(size_t iterations) { total_iterations_ = iterations; }

 protected:
  static constexpr int TEST_BATCH_SIZE = 5;
  static constexpr int TEST_CHANNELS = 3;
  static constexpr int TEST_DEPTH = 2;
  static constexpr int TEST_DH = 2;
  static constexpr int TEST_DW = 3;

  static constexpr int TIMING_BATCH_SIZE = 128;
  static constexpr int TIMING_CHANNELS = 3;
  static constexpr int TIMING_DEPTH = 2;
  static constexpr int TIMING_DH = 64;
  static constexpr int TIMING_DW = 64;
  /*! \brief verbose output */
  bool verbose_ = true;
  /*! \brief Tital iterations */
  size_t total_iterations_ = 10;
};

}  // namespace test
}  // namespace mxnet

#endif  // TEST_OP_RUNNER_H_

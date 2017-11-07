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
#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/activation-inl.h"
#include "../../src/operator/operator_tune-inl.h"
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"

using namespace mxnet;

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(OMP_TUNING, ShowAllTunedOps) {
  const std::unordered_set<std::string>& op_names = op::OperatorTune<float>::TunedOperatorNames();
  for (auto iter = op_names.begin(), e_iter = op_names.end(); iter != e_iter; ++iter) {
    std::cout << *iter << std::endl;
  }
}

using kwargs_t = test::op::kwargs_t;

/*!
 * \brief Rune a core op forward and backward
 * \tparam DType Data type
 * \param isGPU true if operation is to be run on the GPU
 * \param op_kwargs Operator parameters
 * \param op_name Operator name as registered with nnvm
 * \param backward_op_name Backwards operator name as registered with nnvm
 *        If blank, the runner will attempt to determine the backwards operator. If it fails,
 *        an exception will be thrown.
 *        If the string is [none], then no backward operator will be created or executed
 */
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

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(OMP_TUNING, ExecuteBidirectional) {
  RunCoreOpBidirectional(false, {}, "elemwise_add", "_backward_add");
}

template<typename DType>
class TuningTester {
 public:
  using bool_mode_pair = std::pair<bool, op::tune::TuningMode>;

  using shape_vect = std::vector<TShape>;
  using shape_vec_to_bool_map = std::map<shape_vect, bool_mode_pair, test::less_shapevect>;

 private:
  using ShapesToPerfTimingMap =
  std::map<shape_vect, test::perf::timing_map_t, test::less_shapevect>;

  /*!
   * \brief Run timing test on various data shapes and sizes
   * \param isGPU true if the GPU should be used for the timing test
   * \param op_kwargs operator parameters
   * \param op_name The operator's registered name (with nnvm)
   * \param backward_op_name The backward operator's registered name (with nnvm)
   * \return ShapesToPerfTimingMap map holsing timing data for shapes
   */
  static ShapesToPerfTimingMap RunCoreOpTimingTest(const bool isGPU,
                                                   const kwargs_t &op_kwargs,
                                                   const char *op_name,
                                                   const char *backward_op_name = "") {
    ShapesToPerfTimingMap res;
    const kwargs_t kwargs = test::op::CoreOpExecutor<DType>::ArgsWithOpName(
      op_kwargs, op_name, backward_op_name);

    // prime code and cache before the performance runs
    test::op::CoreOperatorRunner<DType> runner;
    runner.set_verbose(false);
    runner.RunBidirectional(false, {{10, 3, 18, 128}}, kwargs, 1);

    // Do the performance runs
    shape_vect shapes;
    if (test::performance_run) {
      shapes = {
        {1,  1, 28,  28},
        {1,  3, 28,  28},
        {50, 1, 18,  32},
        {25, 3, 64,  64},
        {10, 3, 128, 128},
        {20, 3, 256, 256}
      };
    } else {
      shapes = {
        // Non-performance dataset acts as a sanity test
        {1,  1, 28, 28},
        {50, 3, 18, 32}
      };
    }
    const char *pu = isGPU ? "GPU" : "CPU";
    for (const TShape &shape : shapes) {
      const shape_vect this_run_shapes = {shape};
      test::perf::timing_map_t tmap = runner.TimingTest(std::string(op_name) + " Operator " + pu,
                                                        isGPU, false, kwargs, 2, 10,
                                                        this_run_shapes);
      CHECK(res.find(this_run_shapes) == res.end());
      res[this_run_shapes] = tmap;
    }
    return std::move(res);
  }

  using tuned_timing_t = std::map<
    shape_vect,
    std::map<op::tune::TuningMode, test::perf::timing_map_t>, test::less_shapevect>;

  using modesort_t = std::multimap<double, op::tune::TuningMode>;

  /*!
   * \brief Check if the tuning succeeded
   * \param mode_sort modesort_t structure produced by 'CalculateModeSort'
   * \param closeness_factor fraction of largest standard time (omp, no omp) which is an acceptable
   *        range
   * \return a pair <bool, TuningMode> consisting of true or false signifying if the test appears to
   *         have made the correct decision, and the TuningMode which was closest in timing to
   *         the Auto mode.
   */
  static bool_mode_pair
  CheckCorrectTuning(const modesort_t &mode_sort,
                     const double closeness_factor = 0.25) {
    CHECK_EQ(mode_sort.size(), 3U);

    // Determine fastest normal mode
    op::tune::TuningMode fastest_standard_mode = op::tune::Auto;
    for (auto i = mode_sort.begin(), e = mode_sort.end(); i != e; ++i) {
      if (i->second != op::tune::Auto) {
        fastest_standard_mode = i->second;
        break;
      }
    }
    CHECK_NE(fastest_standard_mode, op::tune::Auto);

    // We should be closest to the faster of NeverOMP and AlwaysOMP
    // Take into account some variance, especially if NeverOMP and AlwaysOMP are close together
    std::map<op::tune::TuningMode, double> mode2time;
    for (auto i = mode_sort.begin(), e = mode_sort.end(); i != e; ++i) {
      mode2time[i->second] = i->first;
    }
    const double time_auto = mode2time[op::tune::Auto];
    const double time_no_omp = mode2time[op::tune::NeverOMP];
    const double time_omp = mode2time[op::tune::AlwaysOMP];

    // If difference between OMP and no OMP is < closeness_factor of largest of the two,
    // then we just want to make sure we are close to both of these
    const double fastest_standard_time = std::min(time_no_omp, time_omp);
    const double allowed_difference = closeness_factor * fastest_standard_time;
    const double mustbe_asfast = fastest_standard_time + allowed_difference;

    // Figure out which one we are closest to and return that to help in the analysis
    op::tune::TuningMode closest_to;
    if (fabs(time_auto - time_no_omp) < fabs(time_auto - time_omp)) {
      closest_to = op::tune::NeverOMP;
    } else {
      closest_to = op::tune::AlwaysOMP;
    }

    if (time_auto <= mustbe_asfast || closest_to == fastest_standard_mode) {
      return { true, closest_to };
    }
    return { false, closest_to };
  }

 public:
  /*!
   * \brief Given timing statistics, determine if 'Auto' mode made the correct choice.
   * \param direction Compute direction for which to check (Forward or Backward)
   * \param verbose If true, print the statistical info
   * \return A map of shape vectors to a pair <bool, TuningMode> consisting of true or false
   *         signifying if the test appears to have made the correct decision, and the TuningMode
   *         which was closest in timing to the Auto mode.
   */
  shape_vec_to_bool_map CalculateModeSort(const test::op::TimingDirection direction,
                                          bool verbose = true) {
    shape_vec_to_bool_map results;
    // Incredibly inefficient method of grouping the results
    for (const auto &i : timing_) {
      // print shapes
      const shape_vect &shapes = i.first;
      if (verbose) {
        for (size_t x = 0, n = shapes.size(); x < n; ++x) {
          const TShape &shape = shapes[x];
          if (x) {
            std::cout << ", ";
          }
          std::cout << shape;
        }
        const TShape& lhs_shape = shapes[0];
        std::cout << " lhs=" << test::pretty_num(lhs_shape.Size()) << " items";
        std::cout << "\t(" << TimingDirectionAsString(direction) << ")" << std::endl;
      }
      const auto &mode2timing = i.second;
      modesort_t mode_sort;
      for (const auto &j : mode2timing) {
        const op::tune::TuningMode mode = j.first;
        const test::perf::timing_map_t &tm = j.second;
        if (tm.find(direction) != tm.end()) {
          const test::perf::TimingInstrument::Info &info = tm.find(direction)->second;
          double duration = info.TimeEach();
          mode_sort.insert({duration, mode});
        }
      }
      if (!mode_sort.empty()) {
        // Now we have modes sorted by performance, fastest to slowest
        const bool_mode_pair result = CheckCorrectTuning(mode_sort);
        if (verbose) {
          for (const auto &k : mode_sort) {
            std::cout << "\t" << op::tune::TuningModeToString(k.second)
                      << ": " << k.first << " ms";
            if (k.second == op::tune::Auto) {
              std::cout << " (" << op::tune::TuningModeToString(result.second) << ")";
            }
            std::cout << std::endl;
          }
        }
        std::cout << std::flush;
        if (!result.first && verbose) {
          std::cout << "*** WARNING: Wrong OMP state selected ***" << std::endl << std::flush;
        }
        if (verbose) {
          std::cout << std::endl << std::flush;
        }
        CHECK(results.find(shapes) == results.end());
        results[shapes] = result;
      }
    }
    return std::move(results);
  }

  /*!
   * \brief Perform execution runs for a given forward (and optionally backward) operator
   * \param kwargs Parameters for the operator
   * \param op_name Name by which the operator is registered with nnvm
   * \param backward_op_name Backward operator name
   */
  void TestTunedOperator(const kwargs_t &kwargs,
                         const char *op_name,
                         const char *backward_op_name = COREOP_BWD_OP_NAME_VALUE_NONE) {
    timing_.clear();
    using namespace mxnet::op;
    tuned_timing_t timing;
    for (int x = 0; x < 1; ++x) {
      for (auto mode : {op::tune::AlwaysOMP,
                        op::tune::Auto,
                        op::tune::NeverOMP}) {
        std::cout << std::endl << op::tune::TuningModeToString(mode) << std::endl << std::flush;
        mxnet::op::OperatorTune<DType>::set_tuning_mode(mode);
        const ShapesToPerfTimingMap shapes2perfmap = RunCoreOpTimingTest(false,
                                                                         kwargs,
                                                                         op_name,
                                                                         backward_op_name);
        for (const auto &item : shapes2perfmap) {
          const shape_vect &shapes = item.first;
          const test::perf::timing_map_t &tm = item.second;
          timing_[shapes][mode] = tm;
        }
      }
    }
  }

 private:
  tuned_timing_t  timing_;
};

/* Some test results:
 * AWS c4.8xlarge:
  Success rate for type float: 0.90278
  Success rate for type double: 0.88889
  Success rate for type mshadow::half::half_t: 0.83333
  Success rate for type unsigned char: 0.86111
  Success rate for type int: 0.95833
  Success rate for type long: 0.88889
 * desktop: 12-core (6 real CPU cores + hyperthreading)
  Success rate for type float: 0.79167
  Success rate for type double: 0.75000
  Success rate for type unsigned char: 0.72222
  Success rate for type int: 0.94444
  Success rate for type long: 1.00000
 *
 */

/*!
 * \brief Rune a tuning evaluation
 * \tparam DType Data type for which to evaluate tuning
 */
template<typename DType>
static float EvaluateTune() {
  std::vector<std::pair<std::string, std::string>> binary_operators;
  if (test::performance_run) {
    binary_operators = {
      { "relu",    "" },  // Code can figure out what the backward op is for some
      { "sigmoid", "" },
      { "sqrt",    "" },
      { "elemwise_add", "_backward_add" },
      { "elemwise_mul", "_backward_mul" },
      { "elemwise_div", "_backward_div" }
    };
  } else {
    binary_operators = {
      { "elemwise_add", "_backward_add" }
    };
  }
  size_t count = 0, success = 0;
  for (size_t i = 0, n = binary_operators.size(); i < n; ++i) {
    TuningTester<DType> tuningTester;
    std::cout << "******************************" << std::endl;
    std::cout << "Operators: " << binary_operators[i].first
              << ", " << binary_operators[i].second
              << " for type: " << test::type_name<DType>()
              << std::endl;
    std::cout << "******************************" << std::endl;
    tuningTester.TestTunedOperator({},
                                   binary_operators[i].first.c_str(),
                                   binary_operators[i].second.c_str());
    typename TuningTester<DType>::shape_vec_to_bool_map res_fwd =
      tuningTester.CalculateModeSort(test::op::Forward);
    for (auto iter = res_fwd.begin(), e = res_fwd.end(); iter != e; ++iter) {
      ++count;
      if (iter->second.first) {
        ++success;
      }
    }
    typename TuningTester<DType>::shape_vec_to_bool_map res_bwd =
      tuningTester.CalculateModeSort(test::op::Backward);
    for (auto iter = res_bwd.begin(), e = res_bwd.end(); iter != e; ++iter) {
      ++count;
      if (iter->second.first) {
        ++success;
      }
    }
  }
  if (count) {
    return static_cast<float>(success) / static_cast<float>(count);
  }
  return 1.0f;  // nothing ventured, nothing failed (glass-is-half-full approach)
}

/*! \brief ActivationOp timing test for CPU for float */
TEST(OMP_TUNING, EvaluateTuneTestFloat) {
  typedef float DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for double */
TEST(OMP_TUNING, EvaluateTuneTestDouble) {
  typedef double DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
TEST(OMP_TUNING, EvaluateTuneTestFloat16) {
  typedef mshadow::half::half_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int8_t */
TEST(OMP_TUNING, EvaluateTuneTestInt8) {
  typedef uint8_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int32_t */
TEST(OMP_TUNING, EvaluateTuneTestInt32) {
  typedef int32_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int64_t */
TEST(OMP_TUNING, EvaluateTuneTestInt64) {
  typedef int64_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}


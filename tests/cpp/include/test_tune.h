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
 * \file test_tune.h
 * \brief operator tuning tester
 * \author Chris Olivier
*/

#ifndef TEST_TUNE_H_
#define TEST_TUNE_H_

#ifndef _WIN32
#include <sys/time.h>
#else
#include <Windows.h>
#endif

#include <dmlc/logging.h>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <map>
#include "../../src/operator/operator_tune-inl.h"
#include "./test_util.h"
#include "./test_op.h"
#include "./test_core_op.h"

namespace mxnet {
namespace test {
namespace tune {

/*!
 * \brief Tuning tests, which whether the correct tuning mode is selected by Auto
 * \note This class makes no attempt at being performant (i.e. it does all sorts of slow
 *       deep copies and that sort of thing), so don't insert any of thios code in the main
 *       trunk unless you've verified the performance characteristics for that chunk of code
 * \tparam DType Data type to test
 */
template<typename DType>
class TuningTester {
 public:
  using kwargs_t = test::op::kwargs_t;

  using bool_mode_pair = std::pair<bool, ::mxnet::op::tune::TuningMode>;

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
  ShapesToPerfTimingMap RunCoreOpTimingTest(const bool isGPU,
                                            const kwargs_t &op_kwargs,
                                            const std::vector<shape_vect>& shapes,
                                            const char *op_name,
                                            const char *backward_op_name = "") {
    ShapesToPerfTimingMap res;
    const kwargs_t kwargs = test::op::CoreOpExecutor<DType>::ArgsWithOpName(
      op_kwargs, op_name, backward_op_name);

    // prime code and cache before the performance runs
    test::op::CoreOperatorRunner<DType> runner;
    runner.set_total_iterations(total_iterations_);
    runner.set_verbose(false);
    runner.RunBidirectional(false, {{10, 3, 18, 128}}, kwargs, 1);

    // Do the performance runs
    const char *pu = isGPU ? "GPU" : "CPU";
    for (const std::vector<TShape> &this_run_shapes : shapes) {
      test::perf::timing_map_t tmap = runner.TimingTest(std::string(op_name) + " Operator " + pu,
                                                        isGPU, false, kwargs,
                                                        0, calls_per_iteration_,
                                                        this_run_shapes);
      CHECK(res.find(this_run_shapes) == res.end());
      res[this_run_shapes] = tmap;
    }
    return res;
  }

  using tuned_timing_t = std::map<
    shape_vect,
    std::map<::mxnet::op::tune::TuningMode, test::perf::timing_map_t>, test::less_shapevect>;

  using modesort_t = std::multimap<double, ::mxnet::op::tune::TuningMode>;

  /*!
   * \brief Check if the tuning succeeded
   * \param mode_sort modesort_t structure produced by 'CalculateModeSort'
   * \param closeness_factor fraction of largest standard time (omp, no omp) which is an acceptable
   *        range
   * \return a pair <bool, TuningMode> consisting of true or false signifying if the test appears to
   *         have made the correct decision, and the TuningMode which was closest in timing to
   *         the Auto mode.
   */
  static bool_mode_pair CheckCorrectTuning(const modesort_t &mode_sort,
                                           const double closeness_factor = 0.25) {
    CHECK_EQ(mode_sort.size(), 3U);

    // Determine fastest normal mode
    ::mxnet::op::tune::TuningMode fastest_standard_mode = ::mxnet::op::tune::kAuto;
    for (auto i = mode_sort.begin(), e = mode_sort.end(); i != e; ++i) {
      if (i->second != ::mxnet::op::tune::kAuto) {
        fastest_standard_mode = i->second;
        break;
      }
    }
    CHECK_NE(fastest_standard_mode, ::mxnet::op::tune::kAuto);

    // We should be closest to the faster of kNeverOMP and kAlwaysOMP
    // Take into account some variance, especially if kNeverOMP and kAlwaysOMP are close together
    std::map<::mxnet::op::tune::TuningMode, double> mode2time;
    for (auto i = mode_sort.begin(), e = mode_sort.end(); i != e; ++i) {
      mode2time[i->second] = i->first;
    }
    const double time_auto = mode2time[::mxnet::op::tune::kAuto];
    const double time_no_omp = mode2time[::mxnet::op::tune::kNeverOMP];
    const double time_omp = mode2time[::mxnet::op::tune::kAlwaysOMP];

    // Figure out which one we are closest to and return that to help in the analysis
    ::mxnet::op::tune::TuningMode closest_to;
    if (fabs(time_auto - time_no_omp) < fabs(time_auto - time_omp)) {
      closest_to = ::mxnet::op::tune::kNeverOMP;
    } else {
      closest_to = ::mxnet::op::tune::kAlwaysOMP;
    }

    // If difference between OMP and no OMP is < closeness_factor of largest of the two,
    // then we just want to make sure we are close to both of these
    const double fastest_standard_time = std::min(time_no_omp, time_omp);
    const double allowed_difference = closeness_factor * fastest_standard_time;
    const double mustbe_asfast = fastest_standard_time + allowed_difference;

    return { time_auto <= mustbe_asfast || closest_to == fastest_standard_mode,
             closest_to };
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
                                          bool verbose = true) const {
    if (test::csv) {
      verbose = false;
    }
    shape_vec_to_bool_map results;
    // Incredibly inefficient method of grouping the results
    for (const auto &i : timing_) {
      // print shapes
      const shape_vect &shapes = i.first;
      if (verbose || test::csv) {
        if (!test::csv) {
          for (size_t x = 0, n = shapes.size(); x < n; ++x) {
            const TShape &shape = shapes[x];
            if (x) {
              std::cout << ", ";
            }
            std::cout << shape;
          }
          const TShape &lhs_shape = shapes[0];
          std::cout << " lhs=" << test::pretty_num(lhs_shape.Size()) << " items";
          std::cout << "\t(" << TimingDirectionAsString(direction) << ")" << std::endl;
        } else {
          std::cout << test::pretty_num(shapes[0].Size()) << ",";
        }
      }
      const auto &mode2timing = i.second;
      modesort_t mode_sort;
      for (const auto &j : mode2timing) {
        const ::mxnet::op::tune::TuningMode mode = j.first;
        const test::perf::timing_map_t &tm = j.second;
        if (tm.find(direction) != tm.end()) {
          const test::perf::TimingInstrument::Info &info = tm.find(direction)->second;
          double duration = info.TimeEach();
          mode_sort.insert({duration, mode});
          if (test::csv) {
            std::cout << TimingDirectionAsString(direction) << ","
                      << ::mxnet::op::tune::TuningModeToString(mode) << ","
                      << duration << ",";
          }
        }
      }
      if (test::csv) {
        std::cout << std::endl << std::flush;
      }
      if (!mode_sort.empty()) {
        // Now we have modes sorted by performance, fastest to slowest
        const bool_mode_pair result = CheckCorrectTuning(mode_sort);
        if (verbose && !test::csv) {
          for (const auto &k : mode_sort) {
            std::cout << "\t" << ::mxnet::op::tune::TuningModeToString(k.second)
                      << ": " << k.first << " ms";
            if (k.second == ::mxnet::op::tune::kAuto) {
              std::cout << " (" << ::mxnet::op::tune::TuningModeToString(result.second) << ")";
            }
            std::cout << std::endl;
          }
          std::cout << std::flush;
          if (!result.first) {
            std::cout << "*** WARNING: Wrong OMP state selected ***" << std::endl << std::flush;
          }
        }
        CHECK(results.find(shapes) == results.end()) << "Duplicate entry for set of shapes";
        results[shapes] = result;
      }
    }
    return results;
  }

  /*!
   * \brief Perform execution runs for a given forward (and optionally backward) operator
   * \param kwargs Parameters for the operator
   * \param op_name Name by which the operator is registered with nnvm
   * \param backward_op_name Backward operator name
   */
  void TestTunedOperator(const kwargs_t &kwargs,
                         const bool verbose,
                         const std::vector<shape_vect>& shapevec_vectors,
                         const char *op_name,
                         const char *backward_op_name = COREOP_BWD_OP_NAME_VALUE_NONE) {
    timing_.clear();
    using namespace mxnet::op;
    tuned_timing_t timing;
    for (int x = 0; x < 1; ++x) {
      for (auto mode : {::mxnet::op::tune::kNeverOMP,
                        ::mxnet::op::tune::kAuto,
                        ::mxnet::op::tune::kAlwaysOMP
                        }) {
        if (verbose && !test::csv) {
          std::cout << std::endl << ::mxnet::op::tune::TuningModeToString(mode)
                    << std::endl << std::flush;
        }

        mxnet::op::OperatorTune<DType>::set_tuning_mode(mode);
        const ShapesToPerfTimingMap shapes2perfmap = RunCoreOpTimingTest(false,
                                                                         kwargs,
                                                                         shapevec_vectors,
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

  /*!
   * \brief Calculate the success rate of the run based upon Auto being close to the faster
   *        OMP/non-OMP attempt
   * \param modes List of directions to use in calculation (Forward, Backward). Empty list means all
   * \param verbose Whether to print info
   * \return Success rate ratio (#success/#TOTAL) (0.0-1.0)
   */
  float CalculateSuccessRate(std::vector<test::op::TimingDirection> directions = {},
                             bool verbose = true) const {
    size_t count = 0, success = 0;
    if (directions.empty()) {
      directions = {test::op::kForward, test::op::kBackward};
    }
    for (const test::op::TimingDirection direction : directions) {
      typename test::tune::TuningTester<DType>::shape_vec_to_bool_map res_fwd =
        CalculateModeSort(direction, verbose);
      for (auto iter = res_fwd.begin(), e = res_fwd.end(); iter != e; ++iter) {
        ++count;
        if (iter->second.first) {
          ++success;
        }
      }
    }
    if (count) {
      return static_cast<float>(success) / static_cast<float>(count);
    }
    return 1.0f;  // nothing ventured, nothing failed (glass-is-half-full angle)
  }

  void set_calls_per_iteration(size_t calls_per_iterations) {
    calls_per_iteration_ = calls_per_iterations;
  }
  size_t calls_per_iteration(size_t calls_per_iterations) const {
    return calls_per_iteration_;
  }
  void set_total_iterations(size_t iterations) { total_iterations_ = iterations; }
  size_t total_iterations(size_t iterations) const { return total_iterations_; }

 private:
  /*! \brief Number of iterations */
  size_t          total_iterations_ = 10;
  /*! \brief Calls per iteration */
  size_t          calls_per_iteration_ = 50;
  /*! \brief Raw timing data */
  tuned_timing_t  timing_;
};

}  // namespace tune
}  // namespace test
}  // namespace mxnet

#endif  // TEST_TUNE_H_

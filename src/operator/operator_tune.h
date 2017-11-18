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
#ifndef MXNET_OPERATOR_OPERATOR_TUNE_H_
#define MXNET_OPERATOR_OPERATOR_TUNE_H_

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <vector>
#include <set>
#include <atomic>

namespace mxnet {
namespace op {

#define WORKLOAD_COUNT_SHIFT  11

/*!
 * \brief Shared data for all data types being tuned, acts as a base class for the higher-level
 *        templated tunin classes
 */
class OperatorTuneBase {
 public:
  typedef int64_t duration_t;

 protected:
  /*! \brief Have calculated omp_overhead_ yet? */
  static std::atomic<bool> calculated_;
  /*! \brief Time in nanoseconds for OMP overhead */
  static duration_t omp_overhead_ns_;
  /*! \brief Print debug/trace output for tuning info */
  static bool verbose_tuning_info_;
  /*! \brief Tuning scale factor */
  static double tuning_weight_scale_;

 public:
  typedef std::chrono::high_resolution_clock::time_point Tick;

  /*!
   * \brief Get timestamp for "now"
   * \return Tick object representing the current itmestamp
   */
  static MSHADOW_CINLINE Tick Now() {
    return std::move(std::chrono::high_resolution_clock::now());
  }

  /*!
   * \brief Get duration in nanoseconds
   * \param t1 Start time tick
   * \param t2 End time tick
   * \return duration in nanoseconds between t1 and t2
   */
  static MSHADOW_CINLINE duration_t GetDurationInNanoseconds(const Tick &t1, const Tick &t2) {
    return static_cast<duration_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
  }

  /*!
   * \brief Get duration in nanoseconds between the given 'since' value and now
   * \param since Reference time which to calculate the duration
   * \return Duration in nanoseconds between the given 'since' value and now
   */
  static MSHADOW_CINLINE duration_t GetDurationInNanoseconds(const Tick &since) {
    return GetDurationInNanoseconds(since, Now());
  }

  /*! \brief Loop size to be timed (single op nanos may be too small to store accurately) */
  static constexpr duration_t WORKLOAD_COUNT = (1 << WORKLOAD_COUNT_SHIFT);

  /*!
   * \brief Timer convenience class, sets start time as "now" in the constructor
   */
  struct Timer {
    /*!
     * \brief Constructor, sets start time
     */
    MSHADOW_CINLINE Timer()
      : start_(OperatorTuneBase::Now()) {}
    /*!
     * \brief Get duration in nanoseconds since construction
     * \return Duration in nanoseconds since construction
     */
    MSHADOW_CINLINE int64_t duration() const {
      return OperatorTuneBase::GetDurationInNanoseconds(start_);
    }

    /*!
     * \brief Reference start time, set in constructor
     */
    const OperatorTuneBase::Tick start_;
  };

  /*!
   * \brief Estimate the time to compute with and without OMP, then return whether OMP is faster
   * \param N - Number of iterations desired
   * \param thread_count - Number of OMP threads available to perform the iterations
   * \returns Whether it's faster to use OMP for these iterations
   */
  inline static bool IsOMPFaster(size_t N, size_t thread_count, const uint64_t serial_workload) {
    if (thread_count >= 2) {
      // Compute serial time required
      const uint64_t total_serial_time_ns = serial_workload >> WORKLOAD_COUNT_SHIFT;

      // Compute time required for OMP + # items per thread
      const uint64_t omp_compute_time_ns = (serial_workload / thread_count) >> WORKLOAD_COUNT_SHIFT;
      const uint64_t total_omp_time_ns = omp_overhead_ns_ + omp_compute_time_ns;

      const bool rc = total_omp_time_ns < total_serial_time_ns;
      return rc;
    }
    return false;
  }
};

namespace tune {
/*!
 * \brief Tuning mode for registered kernel operators
 */
enum TuningMode {
  kAuto,         // Based upon tuning data, choose whether to use OMP for kernel CPU Launch() loops
  kNeverOMP,     // Don't use OMP for parallelism (legacy behavior for GPU builds)
  kAlwaysOMP     // Don't use OMP for parallelism (legacy behavior for CPU builds)
};
}  // namespace tune

template<typename DType>
class OperatorTuneByType : public OperatorTuneBase {
 public:
  /*!
   * \brief Set tuning mode
   * \param tuning_mode The tune::TuningMode tuning mode value to set
   */
  static MSHADOW_CINLINE void set_tuning_mode(const tune::TuningMode tuning_mode) {
    // Use const_cast to get past "assigning non-volatile to volatile warning
    const_cast<tune::TuningMode &>(tuning_mode_) = tuning_mode;
  }

  /*!
   * \brief Get the current tuning mode
   * \return tune::TuningMode value for the current tuning mode
   */
  static MSHADOW_CINLINE volatile tune::TuningMode tuning_mode() {
    return tuning_mode_;
  }

  /*!
   * \brief Determine whether to use OMP based upon both timing and configuration
   * \param N - Number of iterations desired
   * \param thread_count - Number of OMP threads available to perform the iterations
   * \returns Whether it's faster to use OMP for these iterations
   */
  inline static bool UseOMP(size_t N, size_t thread_count, const uint64_t serial_workload) {
#ifdef MXNET_USE_OPERATOR_TUNING
    switch (tuning_mode()) {
      case tune::kAuto:
        return OperatorTuneBase::IsOMPFaster(N, thread_count, serial_workload);
      case tune::kNeverOMP:
        return false;
      case tune::kAlwaysOMP:
      default:
        return thread_count > 1;
    }
#else
    return true;
#endif
  }

 protected:
  /*! \brief Tuning mode */
  static volatile tune::TuningMode tuning_mode_;
};

namespace mxnet_op {
/*!
 * \brief Kernel operator wrapper used for tuning data
 */
template<typename Operation, typename DType>
struct tuned_op : public Operation {
  /*! \brief nanoseconds to perform WORKLOAD_COUNT operations
   *  \note It is conceivable that a vector of values could be used for more complex tuning,
   *        but the need hasn't yet arisen
   *  \remarks This variable generally needs to be implemented somewhere.  Currently this is mostly
   *           done via macros in operator_tune.cc.  If you get undefined reference errors when
   *           linking, then try to use one of the macros in that file to instantiate the required
   *           data/functions
   */
  static size_t workload_;

  /*!
   * \brief Extra workload-calculating information (ie times for sub-portions of the calculation)
   */
  static std::vector<float> workload_ex_;

  /*!
   * \brief Calls parent class (Operation)'s UseOMP
   * \tparam Args Variable arguments passed
   * \param N Number of iterations
   * \param thread_count Number of threads available
   * \param args Variable arguments passed
   * \return true if OMP parallelism is recommended
   */
  template<typename ...Args>
  static MSHADOW_CINLINE bool UseOMP(size_t N, size_t thread_count, Args... args) {
    return Operation::UseOMP(N, thread_count, args...);
  }

  /*!
   * \brief Call a standard UseOMP() implementation (if it exists). Currently, these
   *        are implemented in operator_tune.cc for standard unary, binary,
   *        and argumentless kernels (i.e. mshadow_op::sqrt)
   * \tparam Args Variable arguments passed
   * \param N Number of iterations
   * \param thread_count Number of threads available
   * \param args Variable arguments passed
   * \return true if OMP parallelism is recommended
   */
  static bool UseOMP(size_t N, size_t thread_count);
};
}  // namespace mxnet_op

/*!
 * \brief Calculate workload for a given lambda function
 * \tparam Function Lambda type to time for WORKLOAD_COUNT calls
 * \param function Lambda to time for WORKLOAD_COUNT calls
 * \return median workload for function call (nanoseconds for WORKLOAD_COUNT calls)
 */
template<typename Function>
inline int64_t get_workload(Function function) {
  std::multiset<int64_t> durations;
  typename OperatorTuneBase::Timer timer;
  for (int pass = 0; pass < 3; ++pass) {
    for (int i = 0; i < OperatorTuneBase::WORKLOAD_COUNT; ++i) {
      function();
    }
  }
  const OperatorTuneBase::duration_t dd = timer.duration();
  durations.insert(dd);
  return *++durations.begin();  // return median value
}

/*!
 * \brief Declare a template specialization for the Kernel::Launch call for the given OP
 *        wrapped with mxnet_op::op_with_req, using the given OpReqType as the 'req'
 *        template parameter for 'op_with_req'.  This is useful for the standard mshadow_op
 *        operators which need to be wrapped with op_with_req in order to be used with the
 *        Kernel::Launch command.
 *
 * \note Expects to be used within the mxnet::op namespace
 *
 * For example:
 *
 * namespace mxnet_op {
 * template <>
 * template <typename... Args>
 * inline void Kernel<typename mxnet_op::op_with_req<mshadow::op::identity, kNullOp>, cpu>
 *   ::Launch(mshadow::Stream<cpu>* s, const int N, Args... args) {
 *   ::mxnet::op::mxnet_op::Kernel<typename mxnet_op::op_with_req<mshadow::op::identity, kNullOp>,
 *     cpu>::LaunchMShadowOpEx(s, N, args...);
 *   }
 * }
 *
 */
#define MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, __req$) \
  namespace mxnet_op { \
  template<> template<typename ...Args> \
  inline void Kernel<typename mxnet_op::op_with_req<__op$, __req$>, ::mshadow::cpu>:: \
    Launch(mshadow::Stream<::mshadow::cpu> *s, const int N, Args... args) { \
      /* Launch via LaunchMShadowOpEx() */ \
      KernelWrapper<typename mxnet_op::op_with_req<__op$, __req$>, ::mshadow::cpu>:: \
        LaunchMShadowOpEx(s, N, args...); \
  } \
  }  /* namespace mxnet_op */

/*!
 * \brief Declare template specializations for the Kernel::Launch call for the given OP
 *        wrapped with mxnet_op::op_with_req, using the all supported OpReqType as the 'req'
 *        template parameter for 'op_with_req'.  This is useful for the standard mshadow_op
 *        operators which need to be wrapped with op_with_req in order to be used with the
 *        Kernel::Launch command.
 * \note Expects to be used within the mxnet::op namespace
 */
#define MXNET_TUNABLE_MSHADOW_OP(__op$) \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kNullOp); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kWriteTo); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kWriteInplace); \
  MXNET_TUNABLE_MSHADOW_OP_WITH_REQ(__op$, kAddTo);

#define MXNET_TUNABLE_MSHADOW_OP_BACKWARD(__op$) \
  MXNET_TUNABLE_MSHADOW_OP(mxnet::op::mxnet_op::backward_grad<__op$>)

#define MXNET_TUNABLE_MSHADOW_OP_FWD_AND_BWD(__op$) \
  MXNET_TUNABLE_MSHADOW_OP(__op$) \
  MXNET_TUNABLE_MSHADOW_OP_BACKWARD(__op$)

/*!
 * \brief mxnet::op::mxnet_op format ops (work directly with Kernel<>::Launch()
 *        Used from within mxnet::op::mxnet_op namespace
 */
#define _MXNET_TUNABLE_MXNET_OP_FWD(__op$) \
  template<> template<typename ...Args> inline void Kernel<__op$, ::mshadow::cpu>::Launch( \
    mshadow::Stream<::mshadow::cpu> *s, const int N, Args... args) { \
      /* Launch via LaunchMXNetOpEx() */ \
      KernelWrapper<__op$, ::mshadow::cpu>::LaunchMXNetOpEx(s, N, args...); \
  }

/*!
 * \brief mxnet::op::mxnet_op format ops (work directly with Kernel<>::Launch()
 *        Used from within mxnet::op
 */
#define MXNET_TUNABLE_MXNET_OP_FWD(__op$) \
  namespace mxnet_op { _MXNET_TUNABLE_MXNET_OP_FWD(__op$) }  /* namespace mxnet_op */

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_OPERATOR_TUNE_H_

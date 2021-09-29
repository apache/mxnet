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
 * \file test_op.h
 * \brief operator unit test utility functions
 * \author Chris Olivier
 *
 * These classes offer a framework for developing, testing and debugging operators
 * in C++.  They work for both CPU and GPU modes, as well as offer a timing
 * infrastructure in order to test inidividual operator performance.
 *
 * Operator data can be validated against general logic,
 * stored scalar values (which can be generated by this code from an existing operator via
 * BasicOperatorData::dumpC(), as well as against each other (ie check that
 * GPU, CPU, MKL, and CUDNN operators produce the same output given the same input.
 *
 * test_util.h: General testing utility functionality
 * test_perf.h: Performance-related classes
 * test_op.h:   Operator-specific testing classes
 */
#ifndef TEST_OP_H_
#define TEST_OP_H_

#include <mxnet/op_attr_types.h>
#include <ndarray/ndarray_function.h>
#include <mshadow/base.h>
#include <mshadow/stream_gpu-inl.h>
#include <atomic>
#include <algorithm>
#include <map>
#include <list>
#include <string>
#include <vector>
#include <utility>
#include "./test_perf.h"
#include "./test_util.h"

namespace mxnet {
namespace test {
namespace op {

#if MXNET_USE_CUDA
#define MXNET_CUDA_ONLY(__i$) __i$
#else
#define MXNET_CUDA_ONLY(__i$) ((void)0)
#endif

#if MXNET_USE_CUDA
/*!
 * \brief Maintain the lifecycle of a GPU stream
 */
struct GPUStreamScope {
  explicit inline GPUStreamScope(OpContext *opContext)
    : opContext_(*opContext) {
    CHECK_EQ(opContext_.run_ctx.stream == nullptr, true)
      << "Invalid runtime context stream state";
    opContext_.run_ctx.stream = mshadow::NewStream<gpu>(true, true, opContext_.run_ctx.ctx.dev_id);
    CHECK_EQ(opContext_.run_ctx.stream != nullptr, true)
      << "Unable to allocate a GPU stream";
  }
  inline ~GPUStreamScope() {
    if (opContext_.run_ctx.stream) {
      mshadow::DeleteStream<gpu>(static_cast<mshadow::Stream<gpu> *>(opContext_.run_ctx.stream));
      opContext_.run_ctx.stream = nullptr;
    }
  }
  OpContext& opContext_;
};
#endif  // MXNET_USE_CUDA

/*!
 * \brief Base class for operator test-data classes
 */
template<typename DType>
class OperatorDataInitializer {
 public:
  OperatorDataInitializer()
  : generator_(new std::mt19937()) {
  }
  virtual ~OperatorDataInitializer() {}

  /*!
   * \brief Fill a blob with random values
   * \param blob Blob which to fill with random values
   */
  void FillRandom(const RunContext& run_ctx, const TBlob& blob) const {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
    std::uniform_real_distribution<> dis_real(-5.0, 5.0);
    std::uniform_int_distribution<> dis_int(-128, 127);
    test::patternFill(run_ctx, &blob, [this, &dis_real, &dis_int]() -> DType {
      if (!std::is_integral<DType>::value) {
        DType val;
        do {
          val = static_cast<DType>(dis_real(this->generator()));
        } while (std::abs(val) < 1e-5);  // If too close to zero, try again
        return val;
      } else {
        DType val;
        do {
          val = static_cast<DType>(dis_int(this->generator()));
        } while (!val);  // If zero, try again
        return val;
      }
    });
#pragma clang diagnostic pop
  }

  void FillZero(const RunContext& run_ctx, const TBlob& blob) const {
    test::patternFill(run_ctx, &blob, []() -> DType { return DType(0); });
  }

 private:
  /*!
   * \brief mt19937 generator for random number generator
   * \return reference to mt19937 generator object
   */
  std::mt19937& generator() const { return *generator_; }

  /*! \brief Per-test generator */
  std::unique_ptr<std::mt19937> generator_;
};

class OperatorExecutorTiming {
 public:
  inline test::perf::TimingInstrument& GetTiming() { return timing_; }

 private:
  /*! Timing instrumentation */
  test::perf::TimingInstrument timing_;
};

/*! \brief Top-level operator test state info structure */
template<typename OperatorProp, typename OperatorExecutor>
struct OpInfo {
  /*! \brief The operator data */
  std::shared_ptr< OperatorExecutor > executor_;
  /*! \brief The operator prop class */
  std::shared_ptr<OperatorProp> prop_;
  /*! \brief The input type(s) */
  std::vector<int> in_type_;
};

/*! \brief Pair of op info objects, generally for validating ops against each other */
template<typename OperatorProp1, typename OperatorProp2, typename OperatorExecutor>
struct OpInfoPair {
  /*! \brief Operator item 1 */
  test::op::OpInfo<OperatorProp1, OperatorExecutor>  info_1_;
  /*! \brief Operator item 2 */
  test::op::OpInfo<OperatorProp2, OperatorExecutor>  info_2_;
};

/*! \brief Base validator class for validating test data */
template<typename DType, typename AccReal>
class Validator {
 public:
  static inline DType ERROR_BOUND() {
    switch (sizeof(DType)) {
      case sizeof(mshadow::half::half_t):
        return 0.01f;
      default:
        return 0.001f;
    }
  }

  static inline DType ErrorBound(const TBlob *blob) {
    // Due to eps, for a small number of entries, the error will be a bit higher for one pass
    if (blob->shape_.ndim() >= 3) {
      if (blob->Size() / blob->shape_[1] <=4) {
        return ERROR_BOUND() * 15;
      } else {
        return ERROR_BOUND();
      }
    } else {
      // Probably just a vector
      return ERROR_BOUND();
    }
  }

  /*! \brief Adjusted error based upon significant digits */
  template<typename DTypeX>
  static inline DType ErrorBound(const TBlob *blob, const DTypeX v1, const DTypeX v2) {
    const DType initialErrorBound = ErrorBound(blob);
    DType kErrorBound = initialErrorBound;  // This error is based upon the range [0.1x, 0.9x]
    DTypeX avg = static_cast<DTypeX>((fabs(v1) + fabs(v2)) / 2);
    if (avg >= 1) {
      uint64_t vv = static_cast<uint64_t>(avg + 0.5);
      do {
        kErrorBound *= 10;  // shift error to the right one digit
      } while (vv /= 10);
    }
    return kErrorBound;
  }

  template<typename DTypeX>
  static bool isNear(const DTypeX v1, const DTypeX v2, const AccReal error) {
    return error >= fabs(v2 - v1);
  }

  /*! \brief Convenient setpoint for macro-expanded failures */
  template<typename Type1, typename Type2>
  static void on_failure(const size_t i, const size_t n,
                         const Type1 v1, const Type1 v2, const Type2 kErrorBound) {
    LOG(WARNING)
      << "Near test failure: at i = " << i << ", n = "
      << n << ", kErrorBound = " << kErrorBound << std::endl
      << std::flush;
  }

  /*! \brief Compare blob data */
  static bool compare(const TBlob& b1, const TBlob& b2) {
    if (b1.shape_ == b2.shape_) {
      CHECK_EQ(b1.type_flag_, b2.type_flag_) << "Can't compare blobs of different data types";
      MSHADOW_REAL_TYPE_SWITCH(b1.type_flag_, DTypeX, {
        const DTypeX *d1 = b1.dptr<DTypeX>();
        const DTypeX *d2 = b2.dptr<DTypeX>();
        CHECK_NE(d1, d2);  // don't compare the same memory
        for (size_t i = 0, n = b1.Size(), warningCount = 0; i < n; ++i) {
          const DTypeX v1 = *d1++;
          const DTypeX v2 = *d2++;
          const DType kErrorBound = ErrorBound(&b1, v1, v2);
          EXPECT_NEAR(v1, v2, kErrorBound);
          if (!isNear(v1, v2, kErrorBound) && !warningCount++) {
            on_failure(i, n, v1, v2, kErrorBound);
            return false;
          }
        }
      });
      return true;
    }
    return false;
  }

  /*! \brief Compare blob data to a pointer to data */
  template<typename DTypeX>
  static bool compare(const TBlob& b1, const DTypeX *valuePtr) {
    const DTypeX *d1 = b1.dptr<DType>();
    CHECK_NE(d1, valuePtr);  // don't compare the same memory
    const DType kErrorBound = ErrorBound(&b1);
    for (size_t i = 0, n = b1.Size(), warningCount = 0; i < n; ++i) {
      const DTypeX v1 = *d1++;
      const DTypeX v2 = *valuePtr++;
      EXPECT_NEAR(v1, v2, kErrorBound);
      if (!isNear(v1, v2, kErrorBound) && !warningCount++) {
        on_failure(i, n, v1, v2, kErrorBound);
      }
    }
    return true;
  }
};

/*! \brief Operator Prop argument key/value pairs */
typedef std::vector<std::pair<std::string, std::string> > kwargs_t;

/*! \brief Create operator data, prop, the operator itself and init default forward input */
template<
  typename OperatorProp,
  typename OperatorExecutor,
  typename ...Args>
static test::op::OpInfo<OperatorProp, OperatorExecutor> createOpAndInfoF(const kwargs_t &kwargs,
                                                                         Args... args) {
  test::op::OpInfo<OperatorProp, OperatorExecutor> info;
  info.executor_ = std::make_shared<OperatorExecutor>(args...);
  info.prop_ = std::make_shared<OperatorProp>();
  info.in_type_ = { mshadow::DataType<typename OperatorExecutor::DataType>::kFlag };
  info.prop_->Init(kwargs);
  info.executor_->initForward(*info.prop_, &info.in_type_);
  return info;
}

inline mxnet::ShapeVector ShapesOf(const std::vector<NDArray>& arrays) {
  mxnet::ShapeVector res;
  res.reserve(arrays.size());
  for (const NDArray& ar : arrays) {
    res.emplace_back(ar.shape());
  }
  return res;
}

}  // namespace op
}  // namespace test
}  // namespace mxnet

#endif  // TEST_OP_H_

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
 * \file batchnorm_test.cc
 * \brief batchnorm operator unit tests and utility functions
 * \author Chris Olivier
 */

#include <dmlc/logging.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/nn/batch_norm-inl.h"
#include "../../src/operator/operator_common.h"
#include "./test_legacy_op.h"
#include "./test_core_op.h"
#include "imperative/exec_pass.h"

using namespace mxnet;

#define SIMPLE_DIMENSIONS  0
#define DISABLE_VALIDATION 0  // If performance profiling, may do things
// that cause validation to fail

#if !SIMPLE_DIMENSIONS
static constexpr int BATCH_SIZE = 5;
static constexpr int CHANNELS   = 3;
static constexpr int DEPTH      = 2;
static constexpr int DH         = 2;
static constexpr int DW         = 3;
#else
static constexpr int BATCH_SIZE = 1;
static constexpr int CHANNELS   = 1;
static constexpr int DEPTH      = 1;
static constexpr int DH         = 3;
static constexpr int DW         = 2;
#endif

static constexpr int TIMING_BATCH_SIZE = 128;
static constexpr int TIMING_CHANNELS   = 3;
static constexpr int TIMING_DEPTH      = 2;
static constexpr int TIMING_DH         = 28;
static constexpr int TIMING_DW         = 28;

#define PRT(__lbl$, __var$) \
  test::print(ctx.run_ctx, &(std::cout << (__lbl$) << ": "), (__var$), true)

/*!
 * \brief Forward
 */
enum ForwardInputs {
  /* in_data */ kForInData,
  kForGamma,
  kForBeta,
  /* aux_states */ kForMovingMean,
  kForMovingVar
};
enum ForwardOutputs {
  /* outputs */ kForOutData,
  kForOutMean,
  kForOutVar
};

/*!
 * \brief Backward
 */
enum BackwardInputs {
  /* out_grad */ bwd_out_grad_Grad,
  /* out_data */ bwd_out_data_Mean,
  bwd_out_data_Var,
  /* in_data */ bwd_in_data_Data,
  bwd_in_data_Gamma,
  bwd_in_data_Beta,
  /* aux_states */ bwd_aux_states_MovingMean,
  bwd_aux_states_MovingVar
};
enum BackwardOutputs {
  /* in_grad */ bwd_in_grad_Data /* Original input data */,
  /* weight, bias*/ bwd_in_grad_Gamma,
  bwd_in_grad_Beta
};

/**
 *  _____        _           _____       _  _
 * |  __ \      | |         |_   _|     (_)| |
 * | |  | | __ _| |_  __ _    | |  _ __  _ | |_
 * | |  | |/ _` | __|/ _` |   | | | '_ \| || __|
 * | |__| | (_| | |_| (_| |  _| |_| | | | || |_
 * |_____/ \__,_|\__|\__,_| |_____|_| |_|_| \__|
 *
 *
 */
/*! \brief BatchNorm-specific test data  */
template <typename DType, typename AccReal>
class BNOperatorExecutor : public test::op::CoreOpExecutor<DType, AccReal> {
  using Super = typename test::op::CoreOpExecutor<DType, AccReal>;

 public:
  using Super::ctx;

  BNOperatorExecutor(const bool isGPU,
                     const mxnet::TShape& inputShape,
                     const test::op::kwargs_t& kwargs,
                     const bool hasWeightAndBias = false)
      : test::op::CoreOpExecutor<DType, AccReal>(isGPU, {inputShape}),
        hasWeightAndBias_(hasWeightAndBias) {
    param_.Init(kwargs);
  }

  const NDArray* GetForwardInArray(const ForwardInputs idx) const {
    const std::vector<NDArray>& arrs = Super::inputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray* GetForwardOutArray(const ForwardOutputs idx) const {
    const std::vector<NDArray>& arrs = Super::outputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray* GetBackwardInArray(const BackwardInputs idx) {
    const std::vector<NDArray>& arrs = Super::bwd_inputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray* GetBackwardOutArray(const BackwardOutputs idx) const {
    const std::vector<NDArray>& arrs = Super::bwd_outputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  NDArray* GetArray(const ForwardInputs idx) {
    return const_cast<NDArray*>(GetForwardInArray(idx));
  }

  NDArray* GetArray(const ForwardOutputs idx) {
    return const_cast<NDArray*>(GetForwardOutArray(idx));
  }

  NDArray* GetArray(const BackwardOutputs idx) {
    return const_cast<NDArray*>(GetBackwardOutArray(idx));
  }

  NDArray* GetArray(const BackwardInputs idx) {
    return const_cast<NDArray*>(GetBackwardInArray(idx));
  }

  inline const TBlob& Blob(const NDArray* arr) {
    return arr->data();
  }

  template <typename EnumType>
  const TBlob& GetBlob(const EnumType idx) const {
    return const_cast<BNOperatorExecutor<DType, AccReal>*>(this)->GetArray(idx)->data();
  }

  void resetForward() override {
    Super::resetForward();

    // Start by filling all inputs and outputs with an arbitrary values
    for (size_t i = 0, n = Super::inputs().size(); i < n; ++i) {
      test::try_fill(ctx().run_ctx, &Super::inputs()[i].data(), 0.1234);
    }
    for (size_t i = 0, n = Super::outputs().size(); i < n; ++i) {
      test::try_fill(ctx().run_ctx, &Super::outputs()[i].data(), 0.5678);
    }
    for (size_t i = 0, n = Super::bwd_inputs().size(); i < n; ++i) {
      test::try_fill(ctx().run_ctx, &Super::bwd_inputs()[i].data(), 0.9012);
    }
    for (size_t i = 0, n = Super::outputs().size(); i < n; ++i) {
      test::try_fill(ctx().run_ctx, &Super::bwd_outputs()[i].data(), 0.3456);
    }
    // Init input data
    double val = 0;
    test::patternFill(ctx().run_ctx, &GetBlob(kForInData), [&val]() -> double { return val += 1; });

    MSHADOW_TYPE_SWITCH(GetBlob(kForGamma).type_flag_, DTypeX, {
      const TBlob& blob = GetBlob(kForGamma);
      test::fill(ctx().run_ctx, blob, DTypeX(1));
      if (hasWeightAndBias_) {
        if (blob.size(0) > 1) {
          blob.dptr<DTypeX>()[1] = DTypeX(3);
        }
      }
    });
    MSHADOW_TYPE_SWITCH(GetBlob(kForBeta).type_flag_, DTypeX, {
      const TBlob& blob = GetBlob(kForBeta);
      if (!hasWeightAndBias_) {
        test::fill(ctx().run_ctx, blob, DTypeX(0));
      } else {  // This will cause forward pass check to fail when calculating sum == 0
        test::fill(ctx().run_ctx, blob, DTypeX(1));
        if (blob.size(0) > 0) {
          blob.dptr<DTypeX>()[0] = DTypeX(3);
        }
      }
    });

    // Init the moving data (all mean = 0, all var = 1)
    test::try_fill(ctx().run_ctx, &GetBlob(kForMovingMean), 0);
    test::try_fill(ctx().run_ctx, &GetBlob(kForMovingVar), 1);
    test::try_fill(ctx().run_ctx, &GetBlob(kForOutMean), 0);
    test::try_fill(ctx().run_ctx, &GetBlob(kForOutVar), 1);
  }

  void resetBackward() override {
    Super::resetBackward();

    // Join forward input and in_data array
    double val = 0;
    test::patternFill(
        ctx().run_ctx, &GetBlob(bwd_in_data_Data), [&val]() -> double { return val += 1; });

    MSHADOW_TYPE_SWITCH(GetBlob(bwd_in_data_Gamma).type_flag_, DTypeX, {
      const TBlob& blob = GetBlob(bwd_in_data_Gamma);
      test::fill(ctx().run_ctx, blob, DTypeX(1));
      if (hasWeightAndBias_) {
        if (blob.size(0) > 1) {
          blob.dptr<DTypeX>()[1] = DTypeX(3);
        }
      }
    });
    MSHADOW_TYPE_SWITCH(GetBlob(bwd_in_data_Beta).type_flag_, DTypeX, {
      const TBlob& blob = GetBlob(bwd_in_data_Beta);
      if (!hasWeightAndBias_) {
        test::fill(ctx().run_ctx, blob, DTypeX(0));
      } else {  // This will cause forward pass check to fail when calculating sum == 0
        test::fill(ctx().run_ctx, blob, DTypeX(1));
        if (blob.size(0) > 0) {
          blob.dptr<DTypeX>()[0] = DTypeX(3);
        }
      }
    });

    // Join aux arrays
    test::try_fill(ctx().run_ctx, &GetBlob(bwd_aux_states_MovingMean), 0);
    test::try_fill(ctx().run_ctx, &GetBlob(bwd_aux_states_MovingVar), 1);

    test::try_fill(ctx().run_ctx, &GetBlob(bwd_out_data_Mean), 0.0);
    test::try_fill(ctx().run_ctx, &GetBlob(bwd_out_data_Var), 1.0);

    val = -.001;
    test::patternFill(
        ctx().run_ctx, &GetBlob(bwd_out_grad_Grad), [&val]() -> double { return val += 0.01; });
  }

  const bool hasWeightAndBias_;  // This will cause forward pass validation to fail
  op::BatchNormParam param_;
};

/**
 * __      __    _  _     _       _
 * \ \    / /   | |(_)   | |     | |
 *  \ \  / /__ _| | _  __| | __ _| |_  ___  _ __
 *   \ \/ // _` | || |/ _` |/ _` | __|/ _ \| '__|
 *    \  /| (_| | || | (_| | (_| | |_| (_) | |
 *     \/  \__,_|_||_|\__,_|\__,_|\__|\___/|_|
 *
 *
 */
/*! \brief Validate batch norm test outputs */
template <typename DType, typename AccReal>
class BatchNormValidator : public test::op::Validator<DType, AccReal> {
  typedef test::op::Validator<DType, AccReal> Super;

  /*! \brief Only static functions in this class */
  BatchNormValidator() = delete;  // NOLINT

  /*! \brief Check batch norm output - 1D */
  static void checkBatchNorm1D(const TBlob* blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 3U);

    const size_t num      = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t length   = blob->shape_[2];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < length; ++k) {
          const AccReal data = test::data_at<DType>(blob, {i, j, k});
          sum += data;
          var += data * data;
          ++itemCount;
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= length * num;
      var /= length * num;

      if (itemCount > 1) {
        // Due to eps, for a small number of entries, the error will be a bit higher for one pass
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero: " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1: " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 2D */
  static void checkBatchNorm2D(const TBlob* blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 4U);

    const size_t num      = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t height   = blob->shape_[2];
    const size_t width    = blob->shape_[3];

    size_t itemCount = 0, nonZero = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < height; ++k) {
          for (size_t l = 0; l < width; ++l) {
            const AccReal data = test::data_at<DType>(blob, {i, j, k, l});
            sum += data;
            var += data * data;
            ++itemCount;
            if (data != 0) {
              ++nonZero;
            }
          }
        }
      }

      CHECK_GT(itemCount, 1U);  // Not a valid check for one item
      CHECK_NE(nonZero, 0);

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= height * width * num;
      var /= height * width * num;

      if (itemCount > 1) {
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero: " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
          test::print(RunContext(), &(std::cerr << "Mean problem:" << std::endl), *blob);
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1: " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
          test::print(RunContext(), &(std::cerr << "Variance problem:" << std::endl), *blob);
        }
      }
    }
  }

  /*! \brief Check batch norm output - 3D */
  static void checkBatchNorm3D(const TBlob* blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 5U);
    const size_t num      = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t depth    = blob->shape_[2];
    const size_t height   = blob->shape_[3];
    const size_t width    = blob->shape_[4];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t d = 0; d < depth; ++d) {
          for (size_t k = 0; k < height; ++k) {
            for (size_t l = 0; l < width; ++l) {
              const AccReal data = test::data_at<DType>(blob, {i, j, d, k, l});
              sum                = sum + data;
              var                = var + (data * data);
              ++itemCount;
            }
          }
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= depth * height * width * num;
      var /= depth * height * width * num;

      if (itemCount > 1) {
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1 " << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

 public:
  template <typename ExecutorType1, typename ExecutorType2, typename EnumType>
  static inline bool compare(const ExecutorType1& i1,
                             const ExecutorType2& i2,
                             const EnumType idx,
                             bool print = false) {
    test::CAccessAsCPU cpu1(i1.ctx().run_ctx, i1.GetBlob(idx), false),
        cpu2(i2.ctx().run_ctx, i2.GetBlob(idx), false);
    const TBlob& b1 = cpu1();
    const TBlob& b2 = cpu2();
    if (print && test::debug_output) {
      test::print(i1.ctx().run_ctx, &(std::cout << "Blob 1:"), b1, true, true);
      test::print(i2.ctx().run_ctx, &(std::cout << "Blob 2:"), b2, true, true);
    }
    const bool rc = test::op::Validator<DType, AccReal>::compare(b1, b2);
    if (!rc) {
      test::print(i1.ctx().run_ctx, &(std::cerr << "ERROR Blob 1:"), b1, true, true);
      test::print(i2.ctx().run_ctx, &(std::cerr << "ERROR Blob 2:"), b2, true, true);
    }
    return rc;
  }

  /*! \brief Check batch norm output */
  template <typename BNOperatorProp>
  static void validateForward(const RunContext& run_ctx, const BNOperatorProp& data) {
    const TBlob& outputBlob = data.GetBlob(ForwardOutputs::kForOutData);
    if (test::debug_output) {
      test::print(run_ctx, &(std::cout << "Fwd Output Blob:"), outputBlob, true, true);
    }
    test::AccessAsCPU(outputBlob, run_ctx, [](const TBlob& blob) {
      switch (blob.ndim()) {
        case 3:
          checkBatchNorm1D(&blob);
          break;
        case 4:
          checkBatchNorm2D(&blob);
          break;
        case 5:
          checkBatchNorm3D(&blob);
          break;
        default:
          CHECK(false) << "Supplied shape is not supported for this test";
          break;
      }
    });
  }

#define TEST_ISTRUE(__args$)        \
  do {                              \
    bool _rc;                       \
    EXPECT_TRUE((_rc = (__args$))); \
    if (!_rc) {                     \
      rc = false;                   \
    }                               \
  } while (0)

  /*! \brief Compare entire operator data between two test sets */
  template <typename PropType1, typename PropType2>
  static bool compare(
      const test::op::OpInfo<PropType1, BNOperatorExecutor<DType, AccReal>>& info_1,
      const test::op::OpInfo<PropType2, BNOperatorExecutor<DType, AccReal>>& info_2) {
    bool rc = true;
    // Input
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardInputs::kForInData));
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardInputs::kForGamma));
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardInputs::kForBeta));
    // Output
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardOutputs::kForOutData));
    CHECK_EQ(info_2.prop_->getParam().use_global_stats, info_1.prop_->getParam().use_global_stats);

#if MXNET_USE_CUDNN != 1 /* CUDNN takes a different approach here on first pass */
    // Aux
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardOutputs::kForOutMean));
    TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, ForwardOutputs::kForOutVar));
#endif

    if (!info_2.prop_->getParam().use_global_stats) {
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, BackwardInputs::bwd_out_data_Mean));
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, BackwardInputs::bwd_out_data_Var));
      // InGrad
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, BackwardOutputs::bwd_in_grad_Data));
#if 0
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_,
                          BackwardOutputs::bwd_in_grad_Gamma));
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_,
                          BackwardOutputs::bwd_in_grad_Beta));
#endif
      // OutGrad
      TEST_ISTRUE(compare(*info_1.executor_, *info_2.executor_, BackwardInputs::bwd_out_grad_Grad));
    }
    return rc;
  }
};

/**
 *  _____                                  _
 * |  __ \                                | |
 * | |__) |__ _ _ __  __ _ _ __ ___   ___ | |_  ___  _ __  ___
 * |  ___// _` | '__|/ _` | '_ ` _ \ / _ \| __|/ _ \| '__|/ __|
 * | |   | (_| | |  | (_| | | | | | |  __/| |_|  __/| |   \__ \
 * |_|    \__,_|_|   \__,_|_| |_| |_|\___| \__|\___||_|   |___/
 *
 *
 */
static const test::op::kwargs_t blank_kwargs;
static const test::op::kwargs_t blank_kwargs_nocudnn          = {{"cudnn_off", "True"}};
static const test::op::kwargs_t nonfixgamma_kwargs            = {{"fix_gamma", "False"}};
static const test::op::kwargs_t nonfixgamma_kwargs_nocudnn    = {{"fix_gamma", "False"},
                                                              {"cudnn_off", "True"}};
static const test::op::kwargs_t useglobalstats_kwargs         = {{"use_global_stats", "True"}};
static const test::op::kwargs_t useglobalstats_kwargs_nocudnn = {{"use_global_stats", "True"},
                                                                 {"cudnn_off", "True"}};
static const test::op::kwargs_t nfs_ugs_kwargs                = {{"fix_gamma", "False"},
                                                  {"use_global_stats", "True"}};
static const test::op::kwargs_t nfs_ugs_kwargs_nocudnn        = {{"fix_gamma", "False"},
                                                          {"use_global_stats", "True"},
                                                          {"cudnn_off", "True"}};

#if !DISABLE_VALIDATION
static bool isUGS(const test::op::kwargs_t& kwargs) {
  for (const auto& kwarg : kwargs) {
    if (!kwarg.first.compare("use_global_stats")) {
      return kwarg.second.compare("True") == 0;
    }
  }
  return false;
}
#endif  // DISABLE_VALIDATION

/**
 *  _____        _                    ____        _               _
 * |  __ \      | |                  / __ \      | |             | |
 * | |  | | ___ | |__  _   _  __ _  | |  | |_   _| |_ _ __  _   _| |_
 * | |  | |/ _ \| '_ \| | | |/ _` | | |  | | | | | __| '_ \| | | | __|
 * | |__| |  __/| |_) | |_| | (_| | | |__| | |_| | |_| |_) | |_| | |_
 * |_____/ \___||_.__/ \__,_|\__, |  \____/ \__,_|\__| .__/ \__,_|\__|
 *                            __/ |                  | |
 *                           |___/                   |_|
 */
template <typename StreamType, typename OperatorExecutor, typename BlobType>
static StreamType& _DBPRT(const RunContext& run_ctx,
                          const char* label,
                          StreamType* os,
                          const OperatorExecutor& obj,
                          const BlobType type) {
  *os << label << ": ";
  test::print(RunContext(), os, test::CAccessAsCPU(run_ctx, obj.GetBlob(type), false)());
  return *os;
}

#define DBPRT(__os, __obj, __type$) _DBPRT(run_ctx, #__type$, __os, __obj, __type$)

template <typename StreamType, typename Prop, typename OperatorExecutor>
static StreamType& dumpF(StreamType* os,
                         const test::op::OpInfo<Prop, OperatorExecutor>& prop,
                         const size_t x   = 0,
                         const bool force = test::debug_output) {
  if (force) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }
    const RunContext run_ctx = prop.executor_->ctx().run_ctx;
    DBPRT(os, *prop.executor_, ForwardInputs::kForInData);
    DBPRT(os, *prop.executor_, ForwardInputs::kForGamma);
    DBPRT(os, *prop.executor_, ForwardInputs::kForBeta);

    DBPRT(os, *prop.executor_, ForwardInputs::kForMovingMean);
    DBPRT(os, *prop.executor_, ForwardInputs::kForMovingVar);

    DBPRT(os, *prop.executor_, ForwardOutputs::kForOutData);
    DBPRT(os, *prop.executor_, ForwardOutputs::kForOutMean);
    DBPRT(os, *prop.executor_, ForwardOutputs::kForOutVar);
  }
  return *os;
}

template <typename StreamType, typename Prop, typename OperatorExecutor>
static StreamType& dumpB(StreamType* os,
                         const test::op::OpInfo<Prop, OperatorExecutor>& prop,
                         const size_t x   = 0,
                         const bool force = test::debug_output) {
  if (force) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }

    const RunContext run_ctx = prop.executor_->ctx().run_ctx;
    DBPRT(os, *prop.executor_, BackwardOutputs::bwd_in_grad_Data);
    DBPRT(os, *prop.executor_, BackwardOutputs::bwd_in_grad_Gamma);
    DBPRT(os, *prop.executor_, BackwardOutputs::bwd_in_grad_Beta);

    DBPRT(os, *prop.executor_, BackwardInputs::bwd_aux_states_MovingMean);
    DBPRT(os, *prop.executor_, BackwardInputs::bwd_aux_states_MovingVar);

    DBPRT(os, *prop.executor_, BackwardInputs::bwd_out_grad_Grad);
  }
  return *os;
}

/**
 *  _______         _     ______                _   _
 * |__   __|       | |   |  ____|              | | (_)
 *    | | ___  ___ | |_  | |__ _   _ _ __   ___| |_ _  ___  _ __   ___
 *    | |/ _ \/ __|| __| |  __| | | | '_ \ / __| __| |/ _ \| '_ \ / __|
 *    | |  __/\__ \| |_  | |  | |_| | | | | (__| |_| | (_) | | | |\__ \
 *    |_|\___||___/ \__| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_||___/
 *
 *
 */
/*! \brief Test batch norm operator forward pass */
template <typename OperatorProp, typename OperatorExecutor>
static test::op::OpInfo<OperatorProp, OperatorExecutor> TestBatchNormOperatorForward(
    bool isGPU,
    const mxnet::TShape& inputShape,
    const std::vector<std::pair<std::string, std::string>>& kwargs,
    const size_t count = 1) {
#if MXNET_USE_CUDA
  if (isGPU && !test::unitTestsWithCuda) {
    LOG(INFO) << "GPU not found, running test as non-GPU";
  }
#else
  isGPU        = false;
#endif

  test::op::OpInfo<OperatorProp, OperatorExecutor> info =
      test::op::createOpAndInfoF<OperatorProp, OperatorExecutor>(
          OperatorExecutor::ArgsWithOpName(kwargs, "BatchNorm", "_backward_BatchNorm"),
          isGPU,
          inputShape,
          kwargs);

  info.executor_->initForward(*info.prop_, &info.in_type_);

  info.executor_->forward(count);

#if !DISABLE_VALIDATION
  if (!isUGS(kwargs)) {
    BatchNormValidator<typename OperatorExecutor::DataType,
                       typename OperatorExecutor::AccRealType>::validateForward(info.executor_
                                                                                    ->ctx()
                                                                                    .run_ctx,
                                                                                *info.executor_);
  }
#endif

  return info;
}

/*! \brief Test batch norm operator backward pass */
template <typename OperatorProp, typename OperatorExecutor>
static test::op::OpInfo<OperatorProp, OperatorExecutor> runOperatorBackward(
    test::op::OpInfo<OperatorProp, OperatorExecutor>* info,
    const size_t count = 1) {
  info->executor_->initBackward(*info->prop_, &info->in_type_);

  info->executor_->backward(count);
  return *info;
}

static constexpr size_t CYCLE_COUNT = 3;

template <typename OperatorProp1, typename OperatorProp2, typename OperatorExecutor>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, OperatorExecutor> testForwardAndBackward(
    const bool isGPU1,
    const bool isGPU2,
    const mxnet::TShape& inputShape,
    const test::op::kwargs_t& kwargs,
    const size_t count      = 1,
    const size_t cycleCount = CYCLE_COUNT) {
  test::op::OpInfo<OperatorProp1, OperatorExecutor> info_1 =
      TestBatchNormOperatorForward<OperatorProp1, OperatorExecutor>(
          isGPU1, inputShape, kwargs, count);

  test::op::OpInfo<OperatorProp2, OperatorExecutor> info_2 =
      TestBatchNormOperatorForward<OperatorProp2, OperatorExecutor>(
          isGPU2, inputShape, kwargs, count);

  size_t thisCount = 0;

  using DType   = typename OperatorExecutor::DataType;
  using AccReal = typename OperatorExecutor::AccRealType;

  do {
    const bool isLast = thisCount == cycleCount - 1;

    if (thisCount) {
      info_1.executor_->forward(count);
      info_2.executor_->forward(count);
    }

    if (isLast) {
      dumpF(&std::cout, info_1, 1);
      dumpF(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the forward pass
    const bool b1 = BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

    const bool b2 = BatchNormValidator<DType, AccReal>::compare(
        *info_1.executor_, *info_2.executor_, kForInData, false);
    if (!b1 || !b2) {
      dumpF(&std::cout, info_1, 1, true);
      dumpF(&std::cout, info_2, 2, true);
    }

    if (!thisCount) {
      // return backward
      runOperatorBackward(&info_1, count);
      runOperatorBackward(&info_2, count);
    } else {
      info_1.executor_->backward(count);
      info_2.executor_->backward(count);
    }

    if (isLast) {
      dumpB(&std::cout, info_1, 1);
      dumpB(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the backward pass
    if (!BatchNormValidator<DType, AccReal>::compare(info_1, info_2)) {
      dumpF(&std::cout, info_1, 1, true);
      dumpF(&std::cout, info_2, 2, true);
      dumpB(&std::cout, info_1, 1, true);
      dumpB(&std::cout, info_2, 2, true);
    }
  } while (++thisCount < cycleCount);

  return {info_1, info_2};
}
template <typename OperatorProp1, typename OperatorProp2, typename OperatorExecutor>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, OperatorExecutor> testForwardAndBackward(
    const bool isGPU,
    const mxnet::TShape& inputShape,
    const test::op::kwargs_t kwargs,
    const size_t count      = 1,
    const size_t cycleCount = CYCLE_COUNT) {
  return testForwardAndBackward<OperatorProp1, OperatorProp2, OperatorExecutor>(
      isGPU, isGPU, inputShape, kwargs, count, cycleCount);
}

/**
 *   ____          _____
 *  / __ \        |  __ \
 * | |  | |_ __   | |__) |_ __  ___  _ __
 * | |  | | '_ \  |  ___/| '__|/ _ \| '_ \
 * | |__| | |_) | | |    | |  | (_) | |_) |
 *  \____/| .__/  |_|    |_|   \___/| .__/
 *        | |                       | |
 *        |_|                       |_|
 */

// NOTE: This should know which version to use (V1, mkl, etc)
struct BatchNormCoreOpProp : public mxnet::test::op::CoreOpProp {
  void Init(const mxnet::test::op::kwargs_t& kwargs) override {
    mxnet::test::op::CoreOpProp::Init(kwargs);
    params_.Init(kwargs, dmlc::parameter::kAllowUnknown);
  }

  const mxnet::op::BatchNormParam& getParam() const {
    return params_;
  }

  mxnet::op::BatchNormParam params_;
};

template <typename OperatorExecutor>
static test::op::OpInfoPair<BatchNormCoreOpProp, BatchNormCoreOpProp, OperatorExecutor>
testBNForwardAndBackward2D(const bool isGPU,
                           const mxnet::TShape& inputShape,
                           const test::op::kwargs_t& kwargs) {
  CHECK_EQ(inputShape.ndim(), 4);  // V1 can only handle 2D
  return testForwardAndBackward<BatchNormCoreOpProp, BatchNormCoreOpProp, OperatorExecutor>(
      isGPU, isGPU, inputShape, kwargs);
}

template <typename OperatorExecutor>
static test::op::OpInfoPair<BatchNormCoreOpProp, BatchNormCoreOpProp, OperatorExecutor>
testBNForwardAndBackward(const bool isGPU,
                         const mxnet::TShape& inputShape,
                         const test::op::kwargs_t& kwargs) {
  return testForwardAndBackward<BatchNormCoreOpProp, BatchNormCoreOpProp, OperatorExecutor>(
      isGPU, isGPU, inputShape, kwargs);
}

/**
 *   _____             _  _
 *  / ____|           (_)| |
 * | (___   __ _ _ __  _ | |_ _   _
 *  \___ \ / _` | '_ \| || __| | | |
 *  ____) | (_| | | | | || |_| |_| |
 * |_____/ \__,_|_| |_|_| \__|\__, |
 *                             __/ |
 *                            |___/
 */
TEST(BATCH_NORM, TestSanityForwaredAndBackward) {
  MSHADOW_REAL_TYPE_SWITCH_EX(mshadow::kFloat32, DType, AccReal, {
    testBNForwardAndBackward2D<BNOperatorExecutor<DType, AccReal>>(
        false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
  });
}

/**
 *   _____                            _                          _______         _
 *  / ____|                          | |                        |__   __|       | |
 * | |      ___  _ __ _ __  ___   ___| |_ _ __   ___  ___  ___     | | ___  ___ | |_  ___
 * | |     / _ \| '__| '__|/ _ \ / __| __| '_ \ / _ \/ __|/ __|    | |/ _ \/ __|| __|/ __|
 * | |____| (_) | |  | |  |  __/| (__| |_| | | |  __/\__ \\__ \    | |  __/\__ \| |_ \__ \
 *  \_____|\___/|_|  |_|   \___| \___|\__|_| |_|\___||___/|___/    |_|\___||___/ \__||___/
 *
 *
 */
static const std::vector<mshadow::TypeFlag> v2_types = {mshadow::kFloat32,
                                                        mshadow::kFloat64,
                                                        mshadow::kFloat16};

TEST(BATCH_NORM, Test1DForward) {
  for (const mshadow::TypeFlag type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      testBNForwardAndBackward<BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DW}, blank_kwargs);
    });
  }
}

TEST(BATCH_NORM, Test2DForward) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      testBNForwardAndBackward<BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
    });
  }
}

TEST(BATCH_NORM, Test3DForward) {
  for (const mshadow::TypeFlag type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      testBNForwardAndBackward<BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DEPTH, DH, DW}, blank_kwargs);
    });
  }
}

template <typename PropType, typename OperatorExecutor>
static void timingTest(const std::string& label,
                       const bool isGPU,
                       const bool stochastic,
                       const test::op::kwargs_t& kwargs,
                       const int dim = 0,
                       size_t count  = 1) {
  std::cout << std::endl << std::flush;

#ifdef NDEBUG
  size_t COUNT = 50;
#else
  size_t COUNT = 5;
#endif
  if (mxnet::test::quick_test) {
    COUNT = 2;
    count = 1;
  }

  test::perf::TimingInstrument timing;

  std::stringstream ss;
  ss << "Timing: " << COUNT << " iterations";

  for (size_t i = 0; i < COUNT; ++i) {
    index_t batchSize;
    index_t channels;
    index_t depth;
    index_t height;
    index_t width;

    do {
      batchSize = stochastic ? test::rangedRand(1U, BATCH_SIZE * 2U) : TIMING_BATCH_SIZE;
      channels  = stochastic ? test::rangedRand(1U, CHANNELS * 2U) : TIMING_CHANNELS;
      depth     = stochastic ? test::rangedRand(1U, DEPTH * 2U) : TIMING_DEPTH;
      height    = stochastic ? test::rangedRand(1U, DH * 2U) : TIMING_DH;
      width     = stochastic ? test::rangedRand(1U, DW * 2U) : TIMING_DW;
    } while (stochastic && (height * width) == 1U);

    const size_t D = dim ? dim - 1U : test::rangedRand(0U, 2U);

    test::op::OpInfo<PropType, OperatorExecutor> info;
    switch (D) {
      case 0:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
            isGPU, {batchSize, channels, width}, kwargs, count);
        break;
      case 1:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
            isGPU, {batchSize, channels, height, width}, kwargs, count);
        break;
      case 2:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
            isGPU, {batchSize, channels, depth, height, width}, kwargs, count);
        break;
      default:
        CHECK(false) << "rangedRand() returned unexpected value";
    }
    if (info.executor_.get()) {
      runOperatorBackward<PropType, OperatorExecutor>(&info, count);
      timing += info.executor_->GetTiming();
    }
  }

  timing.print(&std::cout, label);
  std::cout << std::endl << std::flush;
}

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#define GPU_TEST_DIMENSIONS 2 /* Only support 2D */
#else
#define GPU_TEST_DIMENSIONS 0 /* Allow stochastic */
#endif                        // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5

/*! \brief Stress-test random batch size/channels/dimension(s) */
TEST(BATCH_NORM, DISABLED_TestStochasticTiming_2D) {
  // Test is disabled due to suspected flakiness
  // https://github.com/apache/mxnet/issues/14411
  MSHADOW_REAL_TYPE_SWITCH_EX(mshadow::kFloat32, DType, AccReal, {
    timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
        "RANDOM: BatchNormCoreOpProp<cpu>", false, true, blank_kwargs_nocudnn, GPU_TEST_DIMENSIONS);
  });
#if MXNET_USE_CUDA
  if (test::unitTestsWithCuda) {
    MSHADOW_REAL_TYPE_SWITCH_EX(mshadow::kFloat32, DType, AccReal, {
      timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          "RANDOM: BatchNormCoreOpProp<gpu>",
          true,
          true,
          blank_kwargs_nocudnn,
          GPU_TEST_DIMENSIONS);
    });
  }
#endif
}

/*! \brief Performance tests */
#ifndef _WIN32
TEST(BATCH_NORM, TestTiming_2D) {
#ifdef NDEBUG
  size_t THISCOUNT = 10;
#else
  size_t THISCOUNT = 2;
#endif
  if (mxnet::test::quick_test) {
    THISCOUNT = 1;
  }
  MSHADOW_REAL_TYPE_SWITCH_EX(mshadow::kFloat32, DType, AccReal, {
#if MXNET_USE_ONEDNN == 1
    // MKL
    timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
        "MKL BatchNormProp<cpu> 2D", false, false, blank_kwargs_nocudnn, 2, THISCOUNT);
#endif  // MXNET_USE_ONEDNN == 1
    // CPU
    test::ScopeSet<volatile bool> disableMKL(&mxnet::op::batchnorm::disable_mkl, true);
    timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
        "BatchNormProp<cpu> 2D", false, false, blank_kwargs_nocudnn, 2, THISCOUNT);
#if MXNET_USE_CUDA
    if (test::unitTestsWithCuda) {
      // CUDA
      timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          "BatchNormProp<gpu> 2D", true, false, blank_kwargs_nocudnn, 2, THISCOUNT);
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
      // CUDA-CUDNN
      timingTest<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          "CUDNN BatchNormProp<gpu> 2D", true, false, blank_kwargs, 2, THISCOUNT);
#endif
    }
#endif
  });
}
#endif  // _WIN32

inline std::ostream& operator<<(std::ostream& os, const test::op::kwargs_t& kwargs) {
  if (!kwargs.empty()) {
    os << "[";
    size_t count = 0;
    for (const auto& item : kwargs) {
      if (count++) {
        os << ", ";
      }
      os << item.first << "=" << item.second;
    }
    os << "]";
  }
  return os;
}

#if 0
TEST(BATCH_NORM, TestIterAll) {
  mxnet::TShape shapes[] = {
    mxnet::TShape({BATCH_SIZE, CHANNELS, DH}),
    mxnet::TShape({BATCH_SIZE, CHANNELS, DH, DW}),
    mxnet::TShape({BATCH_SIZE, CHANNELS, DEPTH, DH, DW})
  };
  int pass = 0;
  const char *tof[2] = { "False", "True" };
  test::op::kwargs_t kwargs;
  for (size_t x1 = 0; x1 < 2U; ++x1) {
    kwargs.push_back({ "fix_gamma", tof[x1] });
    for (size_t x2 = 0; x2 < 2U; ++x2) {
      kwargs.push_back({ "use_global_stats", tof[x2] });
      for (size_t x3 = 0; x3 < 2U; ++x3) {
        if (x3) {
          kwargs.push_back({ "cudnn_off", "True" });
        }
        for (mxnet::TShape shape : shapes) {
          for (bool g1 : { false, true }) {
            for (bool g2 : { false, true }) {
              for (int type : v2_types) {
                std::cout << shape << ", " << op::type_string(type) << ", "
                          << kwargs << ", g1 = "
                          << g1 << ", g2 = " << g2 << std::endl;
                std::cout << "." << std::flush;
                MSHADOW_REAL_TYPE_SWITCH_EX(
                  type, DType, AccReal,
                  {
                    test::op::OpInfoPair<BatchNormCoreOpProp, BatchNormCoreOpProp,
                      BNOperatorExecutor<DType, AccReal>>
                      bi = testForwardAndBackward<BatchNormCoreOpProp,
                      BatchNormCoreOpProp,
                      BNOperatorExecutor<DType, AccReal>>(
                      g1, g2, shape, kwargs);  // Keep it simple
                  });
                std::cout << std::endl;
                ++pass;
              }
            }
          }
        }
        if (x3) {
          kwargs.pop_back();
        }
      }
      kwargs.pop_back();
    }
    kwargs.pop_back();
  }
}
#endif

#ifndef _WIN32
TEST(BATCH_NORM, TestBackward3D) {
  MSHADOW_REAL_TYPE_SWITCH_EX(mshadow::kFloat32, DType, AccReal, {
    const mxnet::TShape inputShape({2, 3, 2, 3, 5});
    test::op::OpInfo<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>> info =
        TestBatchNormOperatorForward<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
            false, inputShape, blank_kwargs);
    info.executor_->initBackward(*info.prop_, &info.in_type_);
    runOperatorBackward(&info);
  });
}
#endif  // _WIN32

template <typename DType>
class ChannelAxisTestData {
 protected:
  enum Mode { LOAD, SAVE };

  void loadOrSave(const RunContext& run_ctx, const TBlob& blob, int channel_axis, const Mode mode) {
    test::CAccessAsCPU cpu_blob(run_ctx, blob, true);
    mxnet::op::batchnorm::BNTensor3<DType> tensor3(cpu_blob(), channel_axis);
    const mxnet::TShape& shape = blob.shape_;
    CHECK_GT(shape.ndim(), 0);
    if (channel_axis < 0) {
      channel_axis = shape.ndim() + channel_axis;
    }
    CHECK_LT(channel_axis, shape.ndim());
    const size_t channel_count = shape[channel_axis];
    std::vector<size_t> indexes(channel_count, 0);
    for (size_t outer = 0, outerCount = tensor3.OuterSize(); outer < outerCount; ++outer) {
      for (size_t channel = 0, channelCount = tensor3.ChannelCount(); channel < channelCount;
           ++channel) {
        CHECK_LT(channel, channel_data_.size());
        for (size_t inner = 0, innerCount = tensor3.InnerSize(); inner < innerCount; ++inner) {
          CHECK_LT(indexes[channel], channel_data_[channel].size());
          if (mode == SAVE) {
            tensor3.get_ref(outer, channel, inner) = channel_data_[channel][indexes[channel]++];
          } else {  // mode == LOAD
            channel_data_[channel][indexes[channel]++] = tensor3.get_ref(outer, channel, inner);
          }
        }
      }
    }
  }

 public:
  std::vector<std::vector<DType>> channel_data_;

  static void print(const std::string& label, const std::vector<std::vector<DType>>& m) {
    if (test::debug_output) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      for (size_t i = 0, n = m.size(); i < n; ++i) {
        const std::vector<DType>& vec = m[i];
        for (size_t j = 0, jn = vec.size(); j < jn; ++j) {
          if (j) {
            std::cout << ", ";
          }
          const DType val = vec[j];
          std::cout << std::fixed << std::setw(7)
                    << std::setprecision(mxnet::test::MPRINT_PRECISION) << std::right << val;
        }
        std::cout << std::endl;
      }
      std::cout << "-----" << std::endl << std::flush;
    }
  }

  static void print(const RunContext& run_ctx, const std::string& label, const TBlob& blob) {
    if (test::debug_output) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      test::CAccessAsCPU cpu_blob(run_ctx, blob, true);
      const size_t totalSize = blob.Size();
      for (size_t i = 0; i < totalSize; ++i) {
        const float val = cpu_blob().dptr<DType>()[i];
        if (i) {
          std::cout << ", ";
        }
        std::cout << std::fixed << std::setw(7) << std::setprecision(mxnet::test::MPRINT_PRECISION)
                  << std::right << val;
      }
      std::cout << std::endl << std::flush;
    }
  }

  void save(const RunContext& run_ctx, const TBlob& blob, const int channel_axis) {
    loadOrSave(run_ctx, blob, channel_axis, SAVE);
  }

  void load(const RunContext& run_ctx, const TBlob& blob, const int channel_axis) {
    loadOrSave(run_ctx, blob, channel_axis, LOAD);
  }
};

template <typename DType, typename AccReal>
static void compare(const RunContext& run_ctx, const TBlob& blob, const std::vector<DType>& vals) {
  CHECK_EQ(blob.Size(), vals.size());
  test::CAccessAsCPU cpu_blob(run_ctx, blob, false);
  const DType* v = cpu_blob().dptr<DType>();
  for (size_t i = 0, n = vals.size(); i < n; ++i) {
    const DType vBlob = v[i];
    const DType vVect = vals[i];
    const bool near   = BatchNormValidator<DType, AccReal>::isNear(
        vBlob, vVect, BatchNormValidator<DType, AccReal>::ErrorBound(&cpu_blob()));
    ASSERT_TRUE(near);
    if (!near) {
      LOG(WARNING) << vBlob << " is not near enough to " << vVect << std::endl;
    }
  }
}

#ifndef _WIN32
template <typename DType, typename AccReal>
static void compare(const std::vector<std::vector<float>>& d1,
                    const std::vector<std::vector<float>>& d2) {
  CHECK_EQ(d1.size(), d2.size());
  for (size_t x = 0, xn = d1.size(); x < xn; ++x) {
    const std::vector<float>& vec1 = d1[x];
    const std::vector<float>& vec2 = d2[x];
    CHECK_EQ(vec1.size(), vec2.size());
    for (size_t i = 0, n = vec1.size(); i < n; ++i) {
      const DType v1  = vec1[i];
      const DType v2  = vec2[i];
      const bool near = BatchNormValidator<DType, AccReal>::isNear(
          v1, v2, BatchNormValidator<DType, AccReal>::ERROR_BOUND());
      if (!near) {
        LOG(WARNING) << v1 << " is not near enough to " << v2 << std::endl;
        ASSERT_TRUE(near);
      }
    }
  }
}

template <typename DType, typename AccReal>
static void testSaveAndLoad(const std::vector<size_t>& dims,
                            const int channelAxis,
                            const std::vector<std::vector<DType>>& inputChannelData,
                            const std::vector<DType>& expectedBlobData) {
  ChannelAxisTestData<DType> data;
  data.channel_data_ = inputChannelData;

  mxnet::TShape shape(dims.size(), -1);
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    shape[i] = index_t(dims[i]);
  }

  RunContext cpu_run_ctx;
  cpu_run_ctx.ctx.dev_type = Context::kCPU;
  cpu_run_ctx.ctx.dev_id   = 0;
  cpu_run_ctx.stream       = nullptr;
  std::unique_ptr<test::StandaloneBlob> blob(
      new test::StandaloneBlob(shape, false, mshadow::DataType<DType>::kFlag));

  data.save(cpu_run_ctx, *blob, channelAxis);
  ChannelAxisTestData<DType>::print(cpu_run_ctx, "saved to blob", *blob);
  compare<DType, AccReal>(cpu_run_ctx, *blob, expectedBlobData);
  data.load(cpu_run_ctx, *blob, channelAxis);
  compare<DType, AccReal>(data.channel_data_, inputChannelData);
}

/*! \brief Check normalization/denormalization of various channel positions */
TEST(BATCH_NORM, TestChannelAxisSaveAndLoad) {
  std::cout << std::endl << std::flush;

  using DType   = float;
  using AccReal = float;

  const std::vector<std::vector<DType>> myData = {
      {1.0f, 1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f, 3.0f}};

  testSaveAndLoad<DType, AccReal>(
      {1, 3, 2, 2},
      1,
      myData,
      {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>(
      {1, 2, 2, 3},
      3,
      myData,
      {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>(
      {1, 2, 3, 2},
      2,
      myData,
      {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f});
}

/*! \brief Insert the channel field `channelCount` into the shape at `channelAxis` position */
static mxnet::TShape MakeShape(const std::vector<index_t>& shape,
                               signed int channelAxis,
                               const size_t channelCount) {
  if (channelAxis < 0) {
    channelAxis += shape.size() + 1;
  }
  CHECK_LT(channelAxis, shape.size() + 1);
  const index_t dim = index_t(shape.size()) + 1;
  mxnet::TShape newShape(dim, -1);
  for (size_t x = 0; x < static_cast<size_t>(channelAxis); ++x) {
    newShape[x] = index_t(shape[x]);
  }
  newShape[channelAxis] = index_t(channelCount);
  for (index_t x = channelAxis + 1; x < dim; ++x) {
    newShape[x] = shape[x - 1];
  }
  return newShape;
}

/*! \brief Create and arrange equivalent data with different channel axes, then compare
 * normalized results */
static void runChannelAxisTest(const bool isGPU1,
                               const bool isGPU2,
                               const test::op::kwargs_t& base_kwargs,
                               const std::vector<index_t> shape,
                               const signed int channelAxis1,
                               const signed int channelAxis2,
                               const size_t channelCount,
                               const bool simpleData,
                               const size_t numberOfPasses = 5

) {
  using DType   = float;
  using AccReal = float;

  size_t spatialSize = 1;
  for (size_t x = 1, n = shape.size(); x < n; ++x) {
    spatialSize *= shape[x];
  }

  const size_t batchSize = shape[0];

  // Create normalized input and output-grad data (inputs to forward and backward pass)
  std::vector<std::vector<DType>> myData, myGradOut;
  DType ival = 1.0f, gval = 0.1f;
  myData.resize(batchSize);
  myData.resize(channelCount);
  myGradOut.resize(channelCount);
  for (size_t c = 0; c < channelCount; ++c) {
    for (size_t i = 0; i < spatialSize; ++i) {
      if (!simpleData) {
        myData[c].push_back(ival += 1.0f);
        myGradOut[c].push_back(gval += 0.1f);
      } else {
        myData[c].push_back(c + 1);
        myGradOut[c].push_back(DType(c + 1) / 10.0f);
      }
    }
  }

  ChannelAxisTestData<DType>::print("myData", myData);
  ChannelAxisTestData<DType>::print("myGradOut", myGradOut);
  ChannelAxisTestData<DType> data_c1, data_c2, grad_c1, grad_c2;

  // For forward pass
  data_c1.channel_data_ = data_c2.channel_data_ = myData;

  // For backward pass
  grad_c1.channel_data_ = grad_c2.channel_data_ = myGradOut;

  test::op::kwargs_t kwargs = base_kwargs;

  // Insert the channel field into the shape at channelAxis position
  const mxnet::TShape shape_c1 = MakeShape(shape, channelAxis1, channelCount);
  const mxnet::TShape shape_c2 = MakeShape(shape, channelAxis2, channelCount);

  // Create operator 1 with ChannelAxis2 (normally the experimental one)
  kwargs.push_back({"axis", std::to_string(channelAxis1)});
  test::op::OpInfo<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>> info_c1 =
      test::op::createOpAndInfoF<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          BNOperatorExecutor<DType, AccReal>::ArgsWithOpName(
              kwargs, "BatchNorm", "_backward_BatchNorm"),
          isGPU1,
          shape_c1,
          kwargs);
  kwargs.pop_back();

  // Create operator 2 with ChannelAxis2 (normally the control one)
  kwargs.push_back({"axis", std::to_string(channelAxis2)});
  test::op::OpInfo<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>> info_c2 =
      test::op::createOpAndInfoF<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          BNOperatorExecutor<DType, AccReal>::ArgsWithOpName(
              kwargs, "BatchNorm", "_backward_BatchNorm"),
          isGPU2,
          shape_c2,
          kwargs);
  kwargs.pop_back();

  // Init operators
  info_c1.executor_->initForward(*info_c1.prop_, &info_c1.in_type_);
  info_c1.executor_->initBackward(*info_c1.prop_, &info_c1.in_type_);
  info_c2.executor_->initForward(*info_c2.prop_, &info_c2.in_type_);
  info_c2.executor_->initBackward(*info_c2.prop_, &info_c2.in_type_);

  // Save input data to blob with new shape 1
  data_c1.save(info_c1.executor_->ctx().run_ctx,
               info_c1.executor_->GetBlob(ForwardInputs::kForInData),
               channelAxis1);
  ChannelAxisTestData<DType>::print(info_c1.executor_->ctx().run_ctx,
                                    "blob 1 input",
                                    info_c1.executor_->GetBlob(ForwardInputs::kForInData));

  // Save input data to blob with new shape 2
  data_c2.save(info_c2.executor_->ctx().run_ctx,
               info_c2.executor_->GetBlob(ForwardInputs::kForInData),
               channelAxis2);
  ChannelAxisTestData<DType>::print(info_c2.executor_->ctx().run_ctx,
                                    "blob 2 input",
                                    info_c2.executor_->GetBlob(ForwardInputs::kForInData));

  // Save output grad to blob with new shape 1
  grad_c1.save(info_c1.executor_->ctx().run_ctx,
               info_c1.executor_->GetBlob(BackwardInputs::bwd_out_grad_Grad),
               channelAxis1);
  ChannelAxisTestData<DType>::print(info_c1.executor_->ctx().run_ctx,
                                    "blob 1 output grad",
                                    info_c1.executor_->GetBlob(BackwardInputs::bwd_out_grad_Grad));

  // Save output grad to blob with new shape 2
  grad_c2.save(info_c2.executor_->ctx().run_ctx,
               info_c2.executor_->GetBlob(BackwardInputs::bwd_out_grad_Grad),
               channelAxis2);
  ChannelAxisTestData<DType>::print(info_c2.executor_->ctx().run_ctx,
                                    "blob 2 output grad",
                                    info_c2.executor_->GetBlob(BackwardInputs::bwd_out_grad_Grad));

  // Run both operators forward and backwards several times
  for (index_t x = 0; x < numberOfPasses; ++x) {
    info_c1.executor_->forward(1);
    info_c2.executor_->forward(1);
    info_c1.executor_->backward(1);
    info_c2.executor_->backward(1);
    break;  // REMOVE ME
  }

  //
  // Check forward pass
  //
  // Transform operator 1's blob output to a normalized shape
  data_c1.load(info_c1.executor_->ctx().run_ctx,
               info_c1.executor_->GetBlob(ForwardOutputs::kForOutData),
               channelAxis1);
  ChannelAxisTestData<DType>::print("channel data 1", data_c1.channel_data_);

  // Transform operator 2's blob output to a normalized shape
  data_c2.load(info_c2.executor_->ctx().run_ctx,
               info_c2.executor_->GetBlob(ForwardOutputs::kForOutData),
               channelAxis2);
  ChannelAxisTestData<DType>::print("channel data 2", data_c2.channel_data_);

  // Compare the operators' output data while they're in a normalized shape
  compare<DType, AccReal>(data_c1.channel_data_, data_c2.channel_data_);

  //
  // Check backward pass
  //
  // Transform operator 1's input-grad blob to a normalized shape
  grad_c1.load(info_c1.executor_->ctx().run_ctx,
               info_c1.executor_->GetBlob(BackwardOutputs::bwd_in_grad_Data),
               channelAxis1);
  ChannelAxisTestData<DType>::print("input grad 1", grad_c1.channel_data_);

  // Transform operator 2's input-grad blob to a normalized shape
  grad_c2.load(info_c2.executor_->ctx().run_ctx,
               info_c2.executor_->GetBlob(BackwardOutputs::bwd_in_grad_Data),
               channelAxis2);
  ChannelAxisTestData<DType>::print("input grad 2", grad_c2.channel_data_);

  // Compare the operators' input grad data while they're in a normalized shape
  compare<DType, AccReal>(grad_c1.channel_data_, grad_c2.channel_data_);
}

TEST(BATCH_NORM, TestChannelAxisSimple) {
  std::cout << std::endl << std::flush;
  const size_t CHANNEL_COUNT       = 4;
  const int DEFAULT_AXIS           = 1;
  const int NEW_AXIS               = -2;
  const bool useSimpleData         = true;  // change to true sometimes for troubleshooting
  const std::vector<index_t> shape = {1, 2, 3};
  // Check against base-case of channel axis position 1
  runChannelAxisTest(false,
                     false,
                     useglobalstats_kwargs_nocudnn,
                     shape,
                     DEFAULT_AXIS,
                     NEW_AXIS,
                     CHANNEL_COUNT,
                     useSimpleData);
}

/*! \brief Test varying channel axis shapes
 *  For several channel counts (1-3), test that result data (after reshape) is
 *  equivalent for the default (channel position 1) and all other channel positions
 *  in the shape vector
 *  Channel position 1 (default) is checked everywhere else, so for and
 *  backward result equivalence here implies correctness for other channel positions
 */
#if 0
TEST(BATCH_NORM, TestChannelAxis) {
  test::ScopeSet<bool> noDebugOutput(&test::debug_output, false);

  test::op::kwargs_t kwargs;
  const std::vector<std::vector<index_t>> shapes =
    {{1, 2},
     {1, 2, 1},
     {1, 2, 3},
     {1, 2, 3, 4}};
  const char *tof[2] = {"False", "True"};

  size_t pass = 0;
  for (size_t x1 = 0; x1 < 2U; ++x1) {
    kwargs.push_back({"fix_gamma", tof[x1]});
    for (size_t x2 = 0; x2 < 2U; ++x2) {
      kwargs.push_back({"use_global_stats", tof[x2]});
      for (size_t x3 = 0; x3 < 2U; ++x3) {
        kwargs.push_back({"cudnn_off", tof[x3]});
        for (bool g1 : { true }) {
        for (bool g1 : { false, true }) {
          for (bool g2 : { false, true }) {
            for (const std::vector<index_t> &simpleShape : shapes) {
              const int dim = static_cast<int>(simpleShape.size());
              for (signed int channelAxis = -dim, shapeDim = dim;
                   channelAxis <= shapeDim;
                   ++channelAxis) {
                for (size_t channelCount = 1; channelCount <= 3; ++channelCount) {
                  // Check against base-case of channel axis position 1
                  runChannelAxisTest(g1, g2, kwargs, simpleShape,
                                     1, channelAxis, channelCount, false);
                  ++pass;
                }
              }
            }
          }
        }
        kwargs.pop_back();
      }
      kwargs.pop_back();
    }
    kwargs.pop_back();
  }
}
#endif

#if MXNET_USE_CUDA

TEST(BATCH_NORM, Test2DForward2D_gpu) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      TestBatchNormOperatorForward<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          true, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
      TestBatchNormOperatorForward<BatchNormCoreOpProp, BNOperatorExecutor<DType, AccReal>>(
          true, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs_nocudnn);
    });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({1, 1, 2, 1});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs_nocudnn);
    });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs_nocudnn);
    });
  }
}

// nonfixgamma_kwargs

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_nfg) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({1, 1, 2, 1});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn);
    });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_nfg) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn);
    });
  }
}

// useglobalstats_kwargs

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_ugs) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({2, 3, 2, 2});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs);
    });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_ugs) {
  for (int type : v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(type, DType, AccReal, {
      const mxnet::TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs);
      testForwardAndBackward<BatchNormCoreOpProp,
                             BatchNormCoreOpProp,
                             BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn);
    });
  }
}

#endif  // MXNET_USE_CUDA

#endif

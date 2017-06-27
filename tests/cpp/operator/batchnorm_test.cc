/*!
 * Copyright (c) 2017 by Contributors
 * \file batchnorm_test.cc
 * \brief operator unit test utility functions
 * \author Chris Olivier
*/

#include <dmlc/logging.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/batch_norm-inl.h"
#include "../../src/operator/batch_norm_v1-inl.h"
#include "test_op.h"
#include "executor/exec_pass.h"

using namespace mxnet;

#define SIMPLE_DIMENSIONS  0
#define MXNET_DUMP_C  0
#define DISABLE_VALIDATION 0  // If performance profiling, may do things
                              // that cause validation to fail

#if !SIMPLE_DIMENSIONS
static constexpr int BATCH_SIZE = 5;
static constexpr int CHANNELS = 3;
static constexpr int DEPTH = 2;
static constexpr int DH = 2;
static constexpr int DW = 3;
#else
static constexpr int BATCH_SIZE = 1;
static constexpr int CHANNELS = 1;
static constexpr int DEPTH = 1;
static constexpr int DH = 2;
static constexpr int DW = 1;
#endif

static constexpr int TIMING_BATCH_SIZE = 128;
static constexpr int TIMING_CHANNELS = 3;
static constexpr int TIMING_DEPTH = 2;
static constexpr int TIMING_DH = 28;
static constexpr int TIMING_DW = 28;

/*! \brief Validate batch norm test outputs */
template<typename DType, typename AccReal>
class BatchNormValidator : public test::op::Validator<DType, AccReal> {
  typedef test::op::Validator<DType, AccReal> Super;
  using Super::compare;

  /*! \brief Only static functions in this class */
  BatchNormValidator() = delete;

  /*! \brief Check batch norm output - 1D */
  static void checkBatchNorm1D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 3U);

    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t length = blob->shape_[2];

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
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1"
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 2D */
  static void checkBatchNorm2D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 4U);

    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t height = blob->shape_[2];
    const size_t width = blob->shape_[3];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < height; ++k) {
          for (size_t l = 0; l < width; ++l) {
            const AccReal data = test::data_at<DType>(blob, {i, j, k, l});
            sum += data;
            var += data * data;
            ++itemCount;
          }
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= height * width * num;
      var /= height * width * num;

      if (itemCount > 1) {
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1"
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 3D */
  static void checkBatchNorm3D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 5U);
    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t depth = blob->shape_[2];
    const size_t height = blob->shape_[3];
    const size_t width = blob->shape_[4];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t d = 0; d < depth; ++d) {
          for (size_t k = 0; k < height; ++k) {
            for (size_t l = 0; l < width; ++l) {
              const AccReal data = test::data_at<DType>(blob, {i, j, d, k, l});
              sum = sum + data;
              var = var + (data * data);
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
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1"
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

 public:
  /*! \brief Check batch norm output */
  template<typename BNOperatorProp>
  static void validateForward(const BNOperatorProp& data) {
    const TBlob& outputBlob = data.c_.blob_output_vec_[mxnet::op::batchnorm::kData];
    switch (outputBlob.ndim()) {
      case 3:
        checkBatchNorm1D(&outputBlob);
        break;
      case 4:
        checkBatchNorm2D(&outputBlob);
        break;
      case 5:
        checkBatchNorm3D(&outputBlob);
        break;
      default:
        CHECK(false) << "Supplied shape is not supported for this test";
        break;
    }
  }

  /*! \brief Compare entire operator data between two test sets */
  template<typename PropType1, typename PropType2>
  static void compare(const test::op::OpInfo<PropType1, DType, AccReal>& info_1,
                      const test::op::OpInfo<PropType2, DType, AccReal>& info_2) {
    // Input
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput,
                        op::batchnorm::kData));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput,
                        op::batchnorm::kGamma));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput,
                        op::batchnorm::kBeta));
    // Output
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kOutput,
                        op::batchnorm::kOut));
    CHECK_EQ(info_2.prop_->getParam().use_global_stats,
             info_1.prop_->getParam().use_global_stats);

#if MXNET_USE_CUDNN != 1 /* CUDNN takes a different approach here on first pass */
    // Aux
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kAux,
                        op::batchnorm::kMovingMean));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kAux,
                        op::batchnorm::kMovingVar));
#endif
    if (!info_2.prop_->getParam().use_global_stats) {
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kOutput,
                          op::batchnorm::kMean));
      // InGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad,
                          op::batchnorm::kData));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad,
                          op::batchnorm::kGamma));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad,
                          op::batchnorm::kBeta));
      // OutGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kOutGrad,
                          op::batchnorm::kData));
    }
  }
};

/*! \brief BatchNorm-specific test data  */
template <typename DType, typename AccReal>
class BNOperatorData : public test::op::BasicOperatorData<DType, AccReal> {
 public:
  BNOperatorData(const bool isGPU, const TShape& inputShape, const bool hasWeightAndBias = false)
    : test::op::BasicOperatorData<DType, AccReal>(isGPU, inputShape)
      , hasWeightAndBias_(hasWeightAndBias) {
  }

  void resetForward() override {
    // Init input data
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_input_vec_[mxnet::op::batchnorm::kData].type_flag_,
      DTypeX,
      {
        DTypeX val = 0;
        test::patternFill<DTypeX>(&this->c_.blob_input_vec_[mxnet::op::batchnorm::kData],
                                  [&val]{ return val += 1; }); });

    MSHADOW_TYPE_SWITCH(
      this->c_.blob_input_vec_[mxnet::op::batchnorm::kGamma].type_flag_,
      DTypeX, {
      const TBlob& blob = this->c_.blob_input_vec_[mxnet::op::batchnorm::kGamma];
        test::fill(blob, DTypeX(1));
      if (hasWeightAndBias_) {
        if (blob.size(0) > 1) {
          blob.dptr<DTypeX>()[1] = DTypeX(3);
        }
      }
      });
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_input_vec_[mxnet::op::batchnorm::kBeta].type_flag_,
      DTypeX, {
        const TBlob& blob = this->c_.blob_input_vec_[mxnet::op::batchnorm::kBeta];
      if (!hasWeightAndBias_) {
        test::fill(blob, DTypeX(0));
      } else {  // This will cause forward pass check to fail when calculating sum == 0
        test::fill(blob, DTypeX(1));
        if (blob.size(0) > 0) {
          blob.dptr<DTypeX>()[0] = DTypeX(3);
        }
      }
      });

    // Init the moving data (all mean = 0, all var = 1)
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingMean].type_flag_,
      DTypeX, {
        test::fill(this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingMean], DTypeX(0));
      });
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingVar].type_flag_,
      DTypeX, {
        test::fill(this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingVar], DTypeX(1));});

    for (size_t i = 0, n = this->c_.blob_output_vec_.size(); i < n; ++i) {
      const int dtype = this->c_.blob_output_vec_[i].type_flag_;
      MSHADOW_TYPE_SWITCH(dtype, DTypeX,
                          { test::fill(this->c_.blob_output_vec_[i], DTypeX(0.1234)); });
    }
  }

  void resetBackward() override {
    DType val = -.001;
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_out_grad_[mxnet::op::batchnorm::kOut].type_flag_,
      DTypeX, {
        test::patternFill<DTypeX>(&this->c_.blob_out_grad_[mxnet::op::batchnorm::kOut],
                                  [&val]{ return val += 1; });
      });

    // out-grad weights
    if (mxnet::op::batchnorm::kGamma < this->c_.blob_out_grad_.size()) {
      MSHADOW_TYPE_SWITCH(
        this->c_.blob_out_grad_[mxnet::op::batchnorm::kGamma].type_flag_,
        DTypeX,
        { test::try_fill(this->c_.blob_out_grad_, mxnet::op::batchnorm::kGamma, DTypeX(0.1)); });
    }

    // out-grad biases
    if (mxnet::op::batchnorm::kBeta < this->c_.blob_out_grad_.size()) {
      MSHADOW_TYPE_SWITCH(
        this->c_.blob_out_grad_[mxnet::op::batchnorm::kBeta].type_flag_,
        DTypeX,
        { test::try_fill(this->c_.blob_out_grad_, mxnet::op::batchnorm::kBeta, DTypeX(0.1)); });
    }

    // in-grad
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_in_grad_[mxnet::op::batchnorm::kData].type_flag_,
      DTypeX,
      { test::try_fill(this->c_.blob_in_grad_, mxnet::op::batchnorm::kData, DTypeX(0)); });

    // in-grad weights
    if (mxnet::op::batchnorm::kGamma < this->c_.blob_in_grad_.size()) {
      MSHADOW_TYPE_SWITCH(
        this->c_.blob_in_grad_[mxnet::op::batchnorm::kGamma].type_flag_,
        DTypeX,
        { test::try_fill(this->c_.blob_in_grad_, mxnet::op::batchnorm::kGamma, DTypeX(0)); });
    }

    // in-grad biases
    if (mxnet::op::batchnorm::kBeta < this->c_.blob_in_grad_.size()) {
      MSHADOW_TYPE_SWITCH(
        this->c_.blob_in_grad_[mxnet::op::batchnorm::kBeta].type_flag_,
        DTypeX,
        { test::try_fill(this->c_.blob_in_grad_, mxnet::op::batchnorm::kBeta, DTypeX(0)); });
    }
  }

  const bool hasWeightAndBias_;  // This will cause forward pass validation to fail
};

static const test::op::kwargs_t blank_kwargs;
static const test::op::kwargs_t blank_kwargs_nocudnn = {
  {"cudnn_off", "True"} };
static const test::op::kwargs_t nonfixgamma_kwargs = {
  {"fix_gamma", "False"} };
static const test::op::kwargs_t nonfixgamma_kwargs_nocudnn = {
  {"fix_gamma", "False"}, {"cudnn_off", "True"} };
static const test::op::kwargs_t useglobalstats_kwargs = {
  {"use_global_stats", "True"} };
static const test::op::kwargs_t useglobalstats_kwargs_nocudnn = {
  {"use_global_stats", "True"}, {"cudnn_off", "True"} };
static const test::op::kwargs_t nfs_ugs_kwargs = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}};
static const test::op::kwargs_t nfs_ugs_kwargs_nocudnn = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}, {"cudnn_off", "True"}  };

#if !DISABLE_VALIDATION
static bool isUGS(const test::op::kwargs_t& kwargs) {
  for (test::op::kwargs_t::const_iterator i = kwargs.begin(),
        e = kwargs.end(); i != e; ++i) {
    if (!i->first.compare("use_global_stats")) {
      return i->second.compare("True") == 0;
    }
  }
  return false;
}
#endif  // DISABLE_VALIDATION

template<typename StreamType, typename DType, typename AccReal>
static StreamType& PRT(
  StreamType *os,
  const test::op::BasicOperatorData<DType, AccReal>& obj,
  const typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType bvt,
  const size_t idx) {
  *os << test::op::BasicOperatorData<DType, AccReal>::bvt2String(bvt) << ": " << idx
      << ": ";
  const TBlob& blob = obj.getBlobVect(bvt)[idx];
  MSHADOW_REAL_TYPE_SWITCH(blob.type_flag_, DTypeX, { test::print_blob<DTypeX>(os, blob); });
  return *os;
}

template<typename StreamType, typename Prop, typename DType, typename AccReal>
static StreamType& dumpF(StreamType *os,
                         const test::op::OpInfo<Prop, DType, AccReal>& prop,
                         const size_t x = 0) {
  if (test::debugOutput) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }
    typedef typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType BlobVectorType;
    PRT(os, *prop.data_, BlobVectorType::kInput, op::batchnorm::kData);
    PRT(os, *prop.data_, BlobVectorType::kInput, op::batchnorm::kGamma);
    PRT(os, *prop.data_, BlobVectorType::kInput, op::batchnorm::kBeta);

    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingMean);
    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingVar);

    PRT(os, *prop.data_, BlobVectorType::kOutput, op::batchnorm::kOut);
    PRT(os, *prop.data_, BlobVectorType::kOutput, op::batchnorm::kMean);
    PRT(os, *prop.data_, BlobVectorType::kOutput, op::batchnorm::kVar);
  }
  return *os;
}

template<typename StreamType, typename Prop, typename DType, typename AccReal>
static StreamType& dumpB(StreamType *os,
                         const test::op::OpInfo<Prop, DType, AccReal>& prop,
                         const size_t x = 0) {
  if (test::debugOutput) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }

    typedef typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType BlobVectorType;
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kData);
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kGamma);
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kBeta);

    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingMean);
    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingVar);

    PRT(os, *prop.data_, BlobVectorType::kOutGrad, op::batchnorm::kOut);
  }
  return *os;
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType, typename AccReal>
static StreamType& dumpF(StreamType *os,
                         const test::op::OpInfoPair<Prop1, Prop2, DType, AccReal>& bi) {
  return dumpF(&dumpF(os, bi.info_1_, 1), bi.info_2_, 2);
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType, typename AccReal>
static StreamType& dumpB(StreamType *os,
                         const test::op::OpInfoPair<Prop1, Prop2, DType, AccReal>& bi) {
  return dumpB(&dumpB(os, bi.info_1_, 1), bi.info_2_, 2);
}

/*! \brief Test batch norm operator forward pass */
template<typename OperatorProp, typename DType, typename AccReal>
static test::op::OpInfo<OperatorProp, DType, AccReal> TestBatchNormOperatorForward(
  bool isGPU,
  const TShape& inputShape,
  const std::vector<std::pair<std::string, std::string> >& kwargs,
  const size_t count = 1) {
#if MXNET_USE_CUDA
  if (isGPU && !test::unitTestsWithCuda) {
    LOG(INFO) << "GPU not found, running test as non-GPU";
  }
#else
  isGPU = false;
#endif

  test::op::OpInfo<OperatorProp, DType, AccReal> info = test::op::createOpAndInfoF<
    OperatorProp, BNOperatorData<DType, AccReal>, DType, AccReal>(isGPU, inputShape, kwargs);

  info.data_->initForward(*info.prop_, &info.in_type_);

  info.data_->forward(count);

#if !DISABLE_VALIDATION
  if (!isUGS(kwargs)) {
    BatchNormValidator<DType, AccReal>::validateForward(*info.data_);
  }
#endif

  return info;
}

/*! \brief Test batch norm operator backward pass */
template<typename DType, typename AccReal, typename OperatorProp>
static test::op::OpInfo<OperatorProp, DType, AccReal> runOperatorBackward(
  test::op::OpInfo<OperatorProp, DType, AccReal> *info,
  const size_t count = 1) {
  info->data_->initBackward(*info->prop_, &info->in_type_);

  info->data_->backward(count);
  return *info;
}

static constexpr size_t CYCLE_COUNT = 3;

template<typename OperatorProp1, typename OperatorProp2, typename DType, typename AccReal>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType, AccReal> testForwardAndBackward(
  const bool isGPU1,
  const bool isGPU2,
  const TShape &inputShape,
  const test::op::kwargs_t& kwargs,
  const bool dumpC,
  const size_t count = 1,
  const size_t cycleCount = CYCLE_COUNT) {
  test::op::OpInfo<OperatorProp1, DType, AccReal> info_1 =
    TestBatchNormOperatorForward<OperatorProp1, DType, AccReal>(isGPU1, inputShape,
                                                                kwargs, count);

  test::op::OpInfo<OperatorProp2, DType, AccReal> info_2 =
    TestBatchNormOperatorForward<OperatorProp2, DType, AccReal>(isGPU2, inputShape,
                                                                kwargs, count);

  size_t thisCount = 0;

  do {
    const bool isLast = thisCount == cycleCount - 1;

    if (thisCount) {
      info_1.data_->forward(count);
      info_2.data_->forward(count);
    }

    if (isLast) {
      dumpF(&std::cout, info_1, 1);
      dumpF(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the forward pass
    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

    test::op::Validator<DType, AccReal>::compare(
      *info_1.data_, *info_2.data_,
      test::op::BasicOperatorData<DType, AccReal>::kInput,
      op::batchnorm::kData);

    if (!thisCount) {
      // return backward
      runOperatorBackward(&info_1, count);
      runOperatorBackward(&info_2, count);
    } else {
      info_1.data_->backward(count);
      info_2.data_->backward(count);
    }

    if (isLast) {
      dumpB(&std::cout, info_1, 1);
      dumpB(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the backward pass
    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);
  } while (++thisCount < cycleCount);

  if (dumpC) {
    info_1.data_->dumpC(&std::cerr, "BN_testForwardAndBackward");
  }

  return  { info_1, info_2 };
}

template<typename OperatorProp1, typename OperatorProp2, typename DType, typename AccReal>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType, AccReal>
testForwardAndBackward(const bool isGPU,
                       const TShape &inputShape,
                       const test::op::kwargs_t kwargs,
                       const bool dumpC = false,
                       const size_t count = 1,
                       const size_t cycleCount = CYCLE_COUNT
) {
  return testForwardAndBackward<OperatorProp1, OperatorProp2, DType, AccReal>(
    isGPU,
    isGPU,
    inputShape,
    kwargs,
    dumpC,
    count,
    cycleCount);
}

template<typename DType, typename AccReal>
static test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>
testBNForwardAndBackward2D(const bool isGPU,
                         const TShape &inputShape,
                         const test::op::kwargs_t kwargs,
                         const bool dumpC = false) {
  CHECK_EQ(inputShape.ndim(), 4);  // V1 can only handle 2D
  return testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
    isGPU,
    isGPU,
    inputShape,
    kwargs,
    dumpC);
}

/*
 * Forward tests
 */
TEST(BATCH_NORM, Test2DForwardV1V2) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32,
    DType,
    AccReal,
    {
      auto infoA = testBNForwardAndBackward2D<DType, AccReal>(
        false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
    });
}

static const std::vector<int> v2_types = {mshadow::kFloat32,
                                          mshadow::kFloat64,
                                          mshadow::kFloat16};

TEST(BATCH_NORM, Test1DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          false, {BATCH_SIZE, CHANNELS, DW}, blank_kwargs);
      });
  }
}

TEST(BATCH_NORM, Test2DForwardV1) {
  TestBatchNormOperatorForward<op::BatchNormProp, float, float>(
    false,
    {BATCH_SIZE, CHANNELS, DH, DW},
    blank_kwargs);
}

TEST(BATCH_NORM, Test2DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        auto opInfoFloatH = TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
      });
  }
}

TEST(BATCH_NORM, Test3DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          false, {BATCH_SIZE, CHANNELS, DEPTH, DH, DW}, blank_kwargs);
      });
  }
}

template<typename PropType, typename DType, typename AccReal>
static void timingTest(const std::string& label,
                       const bool isGPU,
                       const bool stochastic,
                       const test::op::kwargs_t& kwargs,
                       const int dim = 0,
                       const size_t count = 1) {
  std::cout << std::endl << std::flush;

#ifdef NDEBUG
  const size_t COUNT = 50;
#else
  const size_t COUNT = 5;
#endif

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
      channels = stochastic ? test::rangedRand(1U, CHANNELS * 2U) : TIMING_CHANNELS;
      depth = stochastic ? test::rangedRand(1U, DEPTH * 2U) : TIMING_DEPTH;
      height = stochastic ? test::rangedRand(1U, DH * 2U) : TIMING_DH;
      width = stochastic ? test::rangedRand(1U, DW * 2U) : TIMING_DW;
    } while (stochastic && (height * width) == 1U);

    const size_t D = dim ? dim - 1U : test::rangedRand(0U, 2U);

    test::op::OpInfo<PropType, DType, AccReal> info;
    switch (D) {
      case 0:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, width},
          kwargs, count);
        break;
      case 1:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, height, width},
          kwargs, count);
        break;
      case 2:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, depth, height, width},
          kwargs, count);
        break;
      default:
        CHECK(false) << "rangedRand() returned unexpected value";
    }
    if (info.data_.get()) {
      runOperatorBackward<DType, AccReal>(&info, count);
      timing += info.data_->timing_;
    }
  } while (false);

  timing.print(&std::cout, label);
  std::cout << std::endl << std::flush;
}

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#define GPU_TEST_DIMENSIONS  2  /* Only support 2D */
#else
#define GPU_TEST_DIMENSIONS  0  /* Allow stochastic */
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5

/*! \brief Stress-test random batch size/channels/dimension(s) */
TEST(BATCH_NORM, TestStochasticTiming_2D) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      timingTest<op::BatchNormProp, DType, AccReal>("RANDOM: BatchNormProp<cpu>",
                                                    false, true,
                                                    blank_kwargs_nocudnn,
                                                    GPU_TEST_DIMENSIONS); });
#if MXNET_USE_CUDA
  if (test::unitTestsWithCuda) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      mshadow::kFloat32, DType, AccReal,
      {
        timingTest<op::BatchNormProp, DType, AccReal>("RANDOM: BatchNormProp<gpu>",
                                                      true, true,
                                                      blank_kwargs_nocudnn,
                                                      GPU_TEST_DIMENSIONS); });
  }
#endif
}

/*! \brief Performance tests */
TEST(BATCH_NORM, TestTiming_2D) {
#ifdef NDEBUG
  const size_t THISCOUNT = 10;
#else
  const size_t THISCOUNT = 2;
#endif
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      timingTest<op::BatchNormV1Prop, DType, AccReal>("BatchNormV1Prop<cpu> 2D",
                                                      false, false,
                                                      blank_kwargs,
                                                      2, THISCOUNT);
#if MXNET_USE_MKL2017 == 1
      timingTest<op::BatchNormProp, DType, AccReal>("MKL BatchNormProp<cpu> 2D",
                                                    false, false,
                                                    blank_kwargs_nocudnn,
                                                    2, THISCOUNT);
#endif
      test::ScopeSet<volatile bool> disableMKL(&mxnet::op::batchnorm::disable_mkl, true);
      timingTest<op::BatchNormProp, DType, AccReal>("BatchNormProp<cpu> 2D",
                                                    false, false,
                                                    blank_kwargs_nocudnn,
                                                    2, THISCOUNT);
#if MXNET_USE_CUDA
      if (test::unitTestsWithCuda) {
        timingTest<op::BatchNormV1Prop, DType, AccReal>("BatchNormV1Prop<gpu> 2D",
                                                        true, false,
                                                        blank_kwargs,
                                                        2, THISCOUNT);
        timingTest<op::BatchNormProp, DType, AccReal>("BatchNormProp<gpu> 2D",
                                                      true, false,
                                                      blank_kwargs_nocudnn,
                                                      2, THISCOUNT);
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
        timingTest<op::BatchNormProp, DType, AccReal>("CUDNN BatchNormProp<gpu> 2D",
                                                      true, false,
                                                      blank_kwargs,
                                                      2, THISCOUNT);
#endif
      }
#endif
    });
}

/**
 * Backward tests (generally include forward tests as well)
 */

template<typename DType, typename AccReal>
struct BothInfo {
  test::op::OpInfo<op::BatchNormV1Prop, DType, AccReal>  info_v1_;
  test::op::OpInfo<op::BatchNormProp, DType, AccReal>    info_;
};

TEST(BATCH_NORM, TestBackward2D_Simple) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, inputShape, blank_kwargs);  // Keep it simple
    });
}

TEST(BATCH_NORM, TestIterAll) {
  TShape shapes[] = {
    TShape({BATCH_SIZE, CHANNELS, DH}),
    TShape({BATCH_SIZE, CHANNELS, DH, DW}),
    TShape({BATCH_SIZE, CHANNELS, DEPTH, DH, DW})
  };
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
        for (TShape shape : shapes) {
          for (int g1 = 0; g1 < 2U; ++g1) {
            for (int g2 = 0; g2 < 2U; ++g2) {
              for (int type : v2_types) {
                MSHADOW_REAL_TYPE_SWITCH_EX(
                  type, DType, AccReal,
                  {
                    test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType, AccReal>
                      bi = testForwardAndBackward<op::BatchNormProp, op::BatchNormProp,
                      DType, AccReal>(
                      g1 != 0, g2 != 0, shape, kwargs, false);  // Keep it simple
                    if (shape.ndim() == 4 && type == mshadow::kFloat32 && !x3) {
                      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>
                        bi = testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp,
                        DType, AccReal>(
                        g1 != 0, g2 != 0, shape, kwargs, false);  // Keep it simple
                    }
                  });
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

template<typename DType, typename AccReal>
static void test_V1_V2_2D(const test::op::kwargs_t &kwargs, const size_t count) {
  const bool tf[] = { false, true };

  for (size_t xx = 0; xx < sizeof(tf)/sizeof(tf[0]); ++xx) {
    for (size_t yy = 0; yy < sizeof(tf)/sizeof(tf[0]); ++yy) {
      const bool gpu_V1 = tf[xx];
      const bool gpu_V2 = tf[yy];

      TShape shapes[2] = {2, 3};
      const TShape inputShape({2, 3});

      test::op::OpInfo<op::BatchNormV1Prop, DType, AccReal> info_1 = test::op::createOpAndInfoF<
        op::BatchNormV1Prop,
        BNOperatorData<DType, AccReal>,
        DType, AccReal>(gpu_V1, inputShape, kwargs);

      test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_2 = test::op::createOpAndInfoF<
        op::BatchNormProp, BNOperatorData<DType, AccReal>, DType, AccReal>(
        gpu_V2, inputShape, kwargs);

      info_1.data_->initForward(*info_1.prop_, &info_1.in_type_);
      info_2.data_->initForward(*info_1.prop_, &info_1.in_type_);
      info_1.data_->initBackward(*info_1.prop_, &info_1.in_type_);
      info_2.data_->initBackward(*info_1.prop_, &info_1.in_type_);

      TBlob &blob1 = info_1.data_->c_.blob_input_vec_[op::batchnorm::kData];
      test::data_ref<DType>(&blob1, {0, 0}) = -0.05f;
      test::data_ref<DType>(&blob1, {0, 1}) = -0.19f;
      test::data_ref<DType>(&blob1, {0, 2}) = 0.02f;
      test::data_ref<DType>(&blob1, {1, 0}) = -0.12f;
      test::data_ref<DType>(&blob1, {1, 1}) = 0.06f;
      test::data_ref<DType>(&blob1, {1, 2}) = -0.01f;

      TBlob &blob2 = info_2.data_->c_.blob_input_vec_[op::batchnorm::kData];
      test::data_ref<DType>(&blob2, {0, 0}) = -0.05f;
      test::data_ref<DType>(&blob2, {0, 1}) = -0.19f;
      test::data_ref<DType>(&blob2, {0, 2}) = 0.02f;
      test::data_ref<DType>(&blob2, {1, 0}) = -0.12f;
      test::data_ref<DType>(&blob2, {1, 1}) = 0.06f;
      test::data_ref<DType>(&blob2, {1, 2}) = -0.01f;

      test::data_ref<DType>(&info_1.data_->c_.blob_input_vec_[op::batchnorm::kGamma], {1}) = 3;
      test::data_ref<DType>(&info_2.data_->c_.blob_input_vec_[op::batchnorm::kGamma], {1}) = 3;

      test::data_ref<DType>(&info_1.data_->c_.blob_input_vec_[op::batchnorm::kBeta], {0}) = 3;
      test::data_ref<DType>(&info_2.data_->c_.blob_input_vec_[op::batchnorm::kBeta], {0}) = 3;

      for (size_t x = 0; x < count; ++x) {
        info_1.data_->forward();
        info_2.data_->forward();

        BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

        info_1.data_->backward();
        info_2.data_->backward();

        BatchNormValidator<DType, AccReal>::compare(info_1, info_2);
      }
    }
  }
}

TEST(BATCH_NORM, Test_V1_V2_2D_Permutations) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      test_V1_V2_2D<DType, AccReal>(blank_kwargs, 20);
      test_V1_V2_2D<DType, AccReal>(nonfixgamma_kwargs, 20);
      test_V1_V2_2D<DType, AccReal>(useglobalstats_kwargs, 20);
      test_V1_V2_2D<DType, AccReal>(nfs_ugs_kwargs, 20);
    });
}

TEST(BATCH_NORM, TestBackward2D_SimpleNFG) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, inputShape, nonfixgamma_kwargs);
    });
}

TEST(BATCH_NORM, Test2DBackward_Complex) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      test::ScopeSet<bool> noDebugOutput(&test::debugOutput, false);
      const TShape inputShape({9, 14, 16, 91});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, inputShape, blank_kwargs);
    });
}

struct Test2DBackward2DPlusLoadAndCompareLogicUtil {
  template <typename DType, typename AccReal>
  static void test() {
    const TShape inputShape({1, 1, 2, 1});
    test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
      testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
        false, inputShape, blank_kwargs, false, 1, 5);

#if MXNET_DUMP_C
    bi.info_1_.data_->dumpC(&std::cerr, "Test2DBackward2DPlusLoadAndCompareLogic");
#endif

    static const std::vector< std::vector< std::vector<DType> > >
      ___Test2DBackward2DPlusLoadAndCompareLogic_data_shape_1_1_2_1___ = {
      { /* kInput */
        { 1.0f, 2.0f },
        { 1.0f },
        { 0.0f }
      },
      { /* kOutput */
        { -0.998006f, 0.998006f },
        { 1.5f },
        { 0.25f }
      },
      { /* kAux */
        { 0.614265f },
        { 0.692868f }
      },
      { /* kInGrad */
        { -0.00397611f, 0.00397611f },
        { 0.0f },
        { 2.998f }
      },
      { /* kOutGrad */
        { 0.999f, 1.999f }
      }
    };
    // Expected data state when running forward+backward starting with default values
    // Note: This data structure generated by dumpC()
    // Test loaded data agsinst calculated data
    test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_checkLoad =
      test::op::createOpAndInfoF<op::BatchNormProp, BNOperatorData<DType, AccReal>,
        DType, AccReal>(false, inputShape, blank_kwargs);
    info_checkLoad.data_->initForward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
    info_checkLoad.data_->initBackward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
    info_checkLoad.data_->load(___Test2DBackward2DPlusLoadAndCompareLogic_data_shape_1_1_2_1___);
    BatchNormValidator<DType, AccReal>::compare(bi.info_1_, info_checkLoad);
  }
};


TEST(BATCH_NORM, Test2DBackward2DPlusLoadAndCompareLogic) {
  test::ScopeSet<volatile bool> disableMKL(&mxnet::op::batchnorm::disable_mkl, true);
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      Test2DBackward2DPlusLoadAndCompareLogicUtil::test<DType, AccReal>();
    });
}

template<typename PropType, typename DType, typename AccReal>
void compare(const bool isGPU,
             const test::op::OpInfo<PropType, DType, AccReal>& object,
             const std::vector< std::vector< std::vector<DType> > >& values) {
  test::op::OpInfo<PropType, DType, AccReal> info_checkLoad =
    test::op::createOpAndInfoF<PropType, BNOperatorData<DType, AccReal>, DType, AccReal>(
      isGPU, object.data_->c_.blob_input_vec_[0].shape_, blank_kwargs);
  info_checkLoad.data_->initForward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
  info_checkLoad.data_->initBackward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
  info_checkLoad.data_->load(values);
  BatchNormValidator<DType, AccReal>::compare(object, info_checkLoad);
}

TEST(BATCH_NORM, TestBackward1D_Simple) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DTypeX, AccReal,
    {
      const TShape inputShape({1, 1, 2});
      test::op::OpInfo<op::BatchNormProp, DTypeX, AccReal> info =
        TestBatchNormOperatorForward<op::BatchNormProp, DTypeX, AccReal>(false,
                                                                        inputShape,
                                                                        blank_kwargs);
      info.data_->initBackward(*info.prop_, &info.in_type_);
      runOperatorBackward(&info);

#if MXNET_DUMP_C
      info.data_->dumpC(&std::cerr, "BN_TestBackward1D_Simple");
#endif

      // Expected data state when running forward+backward starting with default values
      // Note: This data structure generated by dumpC()
      static const std::vector< std::vector< std::vector<DTypeX> > >
        ___BN_TestBackward1D_Simple_data_shape_1_1_2___ = {
          { /* kInput */
            { 1.0f, 2.0f },
            { 1.0f },
            { 0.0f }
          },
          { /* kOutput */
            { -0.998006f, 0.998006f },
            { 1.5f },
            { 0.25f }
          },
          { /* kAux */
            { 0.15f },
            { 0.925f }
          },
          { /* kInGrad */
            { -0.00397621f, 0.00397609f },
            { 0.0f },
            { 2.998f }
          },
          { /* kOutGrad */
            { 0.999f, 1.999f }
          }
        };
      compare(false, info, ___BN_TestBackward1D_Simple_data_shape_1_1_2___);
    });
}

TEST(BATCH_NORM, TestBackward3D) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({2, 3, 2, 3, 5});
      test::op::OpInfo<op::BatchNormProp, DType, AccReal> info =
        TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          false, inputShape, blank_kwargs);
      info.data_->initBackward(*info.prop_, &info.in_type_);
      runOperatorBackward(&info);
#if MXNET_DUMP_C
      info.data_->dumpC(&std::cerr, "TestBackward3D");
#endif
    });
}

// nonfixgamma_kwargs
TEST(BATCH_NORM, Test2DBackwardMixed_cpu_cpu_nfg) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, false, inputShape, nonfixgamma_kwargs, false);
      dumpF(&std::cout, bi);
      dumpB(&std::cout, bi);
    });
}

// useglobalstats_kwargs
TEST(BATCH_NORM, Test2DBackwardMixed_cpu_cpu_ugs) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, false, inputShape, useglobalstats_kwargs, false);
      dumpF(&std::cout, bi);
      dumpB(&std::cout, bi);
    });
}

template<typename DType>
class ChannelAxisTestData {
 protected:
  enum Mode { LOAD, SAVE };

  void loadOrSave(const TBlob& blob, int channel_axis, const Mode mode) {
    mxnet::op::batchnorm::BNTensor3<DType> tensor3(blob, channel_axis);
    const TShape &shape = blob.shape_;
    CHECK_GT(shape.ndim(), 0);
    if (channel_axis < 0) {
      channel_axis = shape.ndim() + channel_axis;
    }
    CHECK_LT(channel_axis, shape.ndim());
    const size_t channel_count = shape[channel_axis];
    std::vector<size_t> indexes(channel_count, 0);
    for (size_t outer = 0, outerCount = tensor3.OuterSize(); outer < outerCount; ++outer) {
      for (size_t channel = 0, channelCount = tensor3.ChannelCount();
          channel < channelCount; ++channel) {
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
  std::vector<std::vector<DType>>   channel_data_;

  static void print(const std::string& label, const std::vector<std::vector<DType>>& m) {
    if (test::debugOutput) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      for (size_t i = 0, n = m.size(); i < n; ++i) {
        const std::vector<DType> &vec = m[i];
        for (size_t j = 0, jn = vec.size(); j < jn; ++j) {
          if (j) {
            std::cout << ", ";
          }
          const DType val = vec[j];
          std::cout << std::fixed << std::setw(7)
                    << std::setprecision(mxnet::test::MPRINT_PRECISION)
                    << std::right << val;
        }
        std::cout << std::endl;
      }
      std::cout << "-----" << std::endl << std::flush;
    }
  }

  static void print(const std::string& label, const TBlob& blob) {
    if (test::debugOutput) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      const size_t totalSize = blob.Size();
      for (size_t i = 0; i < totalSize; ++i) {
        const float val = blob.dptr<DType>()[i];
        if (i) {
          std::cout << ", ";
        }
        std::cout << std::fixed << std::setw(7) << std::setprecision(mxnet::test::MPRINT_PRECISION)
                  << std::right << val;
      }
      std::cout << std::endl << std::flush;
    }
  }

  void save(const TBlob& blob, const int channel_axis) {
      loadOrSave(blob, channel_axis, SAVE);
  }

  void load(const TBlob& blob, const int channel_axis) {
    loadOrSave(blob, channel_axis, LOAD);
  }
};

template<typename DType, typename AccReal>
static void compare(const TBlob& blob, const std::vector<DType>& vals) {
  CHECK_EQ(blob.Size(), vals.size());
  const DType *v = blob.dptr<DType>();
  for (size_t i = 0, n = vals.size(); i < n; ++i) {
    const DType vBlob = v[i];
    const DType vVect = vals[i];
    const bool near = test::op::Validator<DType, AccReal>::isNear(
      vBlob, vVect, test::op::Validator<DType, AccReal>::ErrorBound(&blob));
    EXPECT_TRUE(near);
    if (!near) {
      LOG(WARNING) << vBlob << " is not near enough to " << vVect << std::endl;
    }
  }
}

template<typename DType, typename AccReal>
static void compare(const std::vector<std::vector<float>>& d1,
                    const std::vector<std::vector<float>>& d2) {
  CHECK_EQ(d1.size(), d2.size());
  for (size_t x = 0, xn = d1.size(); x < xn; ++x) {
    const std::vector<float> &vec1 = d1[x];
    const std::vector<float> &vec2 = d2[x];
    CHECK_EQ(vec1.size(), vec2.size());
    for (size_t i = 0, n = vec1.size(); i < n; ++i) {
      const DType v1 = vec1[i];
      const DType v2 = vec2[i];
      const bool near = test::op::Validator<DType, AccReal>::isNear(
        v1, v2, test::op::Validator<DType, AccReal>::ERROR_BOUND());
      EXPECT_TRUE(near);
      if (!near) {
        LOG(WARNING) << v1 << " is not near enough to " << v2 << std::endl;
      }
    }
  }
}

template<typename DType, typename AccReal>
static void testSaveAndLoad(const std::vector<size_t>& dims,
                            const int channelAxis,
                            const std::vector<std::vector<DType>>& inputChannelData,
                            const std::vector<DType>& expectedBlobData) {
  ChannelAxisTestData<DType> data;
  data.channel_data_ = inputChannelData;

  TShape shape(dims.size());
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    shape[i] = index_t(dims[i]);
  }

  std::unique_ptr<test::StandaloneBlob> blob(new test::StandaloneBlob(
    shape, false, mshadow::DataType<DType>::kFlag));

  data.save(*blob, channelAxis);
  ChannelAxisTestData<DType>::print("saved to blob", *blob);
  compare<DType, AccReal>(*blob, expectedBlobData);
  data.load(*blob, channelAxis);
  compare<DType, AccReal>(data.channel_data_, inputChannelData);
}

/*! \brief Check normalization/denormalization of various channel positions */
TEST(BATCH_NORM, TestChannelAxisSaveAndLoad) {
  std::cout << std::endl << std::flush;

  typedef float DType;
  typedef float AccReal;

  const std::vector<std::vector<DType>> myData =
    { { 1.0f, 1.0f, 1.0, 1.0 },
      { 2.0f, 2.0f, 2.0f, 2.0f },
      { 3.0f, 3.0f, 3.0f, 3.0f } };

  testSaveAndLoad<DType, AccReal>({ 1, 3, 2, 2 }, 1, myData,
                                  { 1.0f, 1.0f, 1.0f, 1.0f,
                                    2.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>({ 1, 2, 2, 3 }, 3, myData,
                                  { 1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>({ 1, 2, 3, 2 }, 2, myData,
                                  { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
                                    1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f});
}

/*! \brief Insert the channel field `channelCount` into the shape at `channelAxis` position */
static TShape MakeShape(const std::vector<index_t>& shape,
                        signed int channelAxis,
                        const size_t channelCount) {
  if (channelAxis < 0) {
    channelAxis += shape.size() + 1;
  }
  CHECK_LT(channelAxis, shape.size() + 1);
  const index_t dim = index_t(shape.size()) + 1;
  TShape newShape(dim);
  for (size_t x = 0; x < channelAxis; ++x) {
    newShape[x] = index_t(shape[x]);
  }
  newShape[channelAxis] = index_t(channelCount);
  for (int x = channelAxis + 1; x < dim; ++x) {
    newShape[x] = shape[x - 1];
  }
  return newShape;
}

/*! \brief Create and arrange equivalent data with different channel axes, then compare
 * normalized results */
static void runChannelAxisTest(
  const bool isGPU1,
  const bool isGPU2,
  const test::op::kwargs_t& base_kwargs,
  const std::vector<index_t> shape,
  const signed int channelAxis1,
  const signed int channelAxis2,
  const size_t channelCount,
  const bool simpleData,
  const size_t numberOfPasses = 5

) {
  typedef float DType;
  typedef float AccReal;

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
  const TShape shape_c1 = MakeShape(shape, channelAxis1, channelCount);
  const TShape shape_c2 = MakeShape(shape, channelAxis2, channelCount);

  // Create operator 1 with ChannelAxis2 (normally the experimental one)
  kwargs.push_back({"axis", std::to_string(channelAxis1)});
  test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_c1 = test::op::createOpAndInfoF<
    op::BatchNormProp, BNOperatorData<DType, AccReal>, DType, AccReal>(
    isGPU1, shape_c1, kwargs);

  // Create operator 2 with ChannelAxis2 (normally the control one)
  kwargs.pop_back();
  kwargs.push_back({"axis", std::to_string(channelAxis2)});
  test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_c2 = test::op::createOpAndInfoF<
    op::BatchNormProp, BNOperatorData<DType, AccReal>, DType, AccReal>(
    isGPU2, shape_c2, kwargs);
  kwargs.pop_back();

  // Init operators
  info_c1.data_->initForward(*info_c1.prop_, &info_c1.in_type_);
  info_c1.data_->initBackward(*info_c1.prop_, &info_c1.in_type_);
  info_c2.data_->initForward(*info_c2.prop_, &info_c2.in_type_);
  info_c2.data_->initBackward(*info_c2.prop_, &info_c2.in_type_);

  // Save input data to blob with new shape 1
  data_c1.save(info_c1.data_->c_.blob_input_vec_[0], channelAxis1);
  ChannelAxisTestData<DType>::print("blob 1 input", info_c1.data_->c_.blob_input_vec_[0]);

  // Save input data to blob with new shape 2
  data_c2.save(info_c2.data_->c_.blob_input_vec_[0], channelAxis2);
  ChannelAxisTestData<DType>::print("blob 2 input", info_c2.data_->c_.blob_input_vec_[0]);

  // Save output grad to blob with new shape 1
  grad_c1.save(info_c1.data_->c_.blob_out_grad_[0], channelAxis1);
  ChannelAxisTestData<DType>::print("blob 1 output grad", info_c1.data_->c_.blob_out_grad_[0]);

  // Save output grad to blob with new shape 2
  grad_c2.save(info_c2.data_->c_.blob_out_grad_[0], channelAxis2);
  ChannelAxisTestData<DType>::print("blob 2 output grad", info_c2.data_->c_.blob_out_grad_[0]);

  // Run both operators forward and backwards several times
  for (int x = 0; x < numberOfPasses; ++x) {
    info_c1.data_->forward();
    info_c2.data_->forward();

    info_c1.data_->backward();
    info_c2.data_->backward();
  }

  // Transform operator 1's blob output to a normalized shape
  data_c1.load(info_c1.data_->c_.blob_output_vec_[0], channelAxis1);
  ChannelAxisTestData<DType>::print("channel data 1", data_c1.channel_data_);

  // Transform operator 2's blob output to a normalized shape
  data_c2.load(info_c2.data_->c_.blob_output_vec_[0], channelAxis2);
  ChannelAxisTestData<DType>::print("channel data 2", data_c2.channel_data_);

  // Compare the operators' output data while they're in a normalized shape
  compare<DType, AccReal>(data_c1.channel_data_, data_c2.channel_data_);

  // Transform operator 1's input-grad blob to a normalized shape
  grad_c1.load(info_c1.data_->c_.blob_in_grad_[0], channelAxis1);
  ChannelAxisTestData<DType>::print("input grad 1", grad_c1.channel_data_);

  // Transform operator 2's input-grad blob to a normalized shape
  grad_c2.load(info_c2.data_->c_.blob_in_grad_[0], channelAxis2);
  ChannelAxisTestData<DType>::print("input grad 2", grad_c2.channel_data_);

  // Compare the operators' input grad data while they're in a normalized shape
  compare<DType, AccReal>(grad_c1.channel_data_, grad_c2.channel_data_);
}

TEST(BATCH_NORM, TestChannelAxisSimple) {
  std::cout << std::endl << std::flush;
  const size_t CHANNEL_COUNT = 4;
  const int DEFAULT_AXIS = 1;
  const int NEW_AXIS = -2;
  const bool useSimpleData = true;  // change to true sometimes for troubleshooting
  const std::vector<index_t> shape = {1, 2, 3};
  // Check against base-case of channel axis position 1
  runChannelAxisTest(false, false,
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
TEST(BATCH_NORM, TestChannelAxis) {
  test::ScopeSet<bool> noDebugOutput(&test::debugOutput, false);

  test::op::kwargs_t kwargs;
  const std::vector<std::vector<index_t>> shapes =
    { {1, 2}, {1, 2, 1}, {1, 2, 3}, {1, 2, 3, 4} };
  const char *tof[2] = { "False", "True" };

  for (size_t x1 = 0; x1 < 2U; ++x1) {
    kwargs.push_back({"fix_gamma", tof[x1]});
    for (size_t x2 = 0; x2 < 2U; ++x2) {
      kwargs.push_back({"use_global_stats", tof[x2]});
      for (size_t x3 = 0; x3 < 2U; ++x3) {
        kwargs.push_back({"cudnn_off", tof[x3]});
        for (int g1 = 0; g1 < 2U; ++g1) {
          for (int g2 = 0; g2 < 2U; ++g2) {
            for (const std::vector<index_t> &simpleShape : shapes) {
              const int dim = static_cast<int>(simpleShape.size());
              for (signed int channelAxis = -dim, shapeDim = dim;
                   channelAxis <= shapeDim;
                   ++channelAxis) {
                for (size_t channelCount = 1; channelCount <= 3; ++channelCount) {
                  // Check against base-case of channel axis position 1
                  runChannelAxisTest(g1 != 0, g2 != 0, kwargs, simpleShape,
                                     1, channelAxis, channelCount, false);
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

#if MXNET_USE_CUDA

TEST(BATCH_NORM, Test2DForwardV12D_gpu) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      TestBatchNormOperatorForward<op::BatchNormV1Prop, DType, AccReal>(
        true,
        {BATCH_SIZE, CHANNELS, DH, DW},
        blank_kwargs);
      TestBatchNormOperatorForward<op::BatchNormV1Prop, DType, AccReal>(
        true,
        {BATCH_SIZE, CHANNELS, DH, DW},
        blank_kwargs);
    });
}

TEST(BATCH_NORM, Test2DForward2D_gpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          true,
          {BATCH_SIZE, CHANNELS, DH, DW},
          blank_kwargs);
        TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          true,
          {BATCH_SIZE, CHANNELS, DH, DW},
          blank_kwargs_nocudnn);
      });
  }
}

// blank_kwargs
TEST(BATCH_NORM, Test2DBackwardMixedV1_gpu_cpu) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormV1Prop, DType, AccReal>(
        false, true, inputShape, blank_kwargs, false);
    });
}

TEST(BATCH_NORM, Test2DBackwardMixedV1Complex_gpu_cpu) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormV1Prop, DType, AccReal>(
        false, true, inputShape, blank_kwargs, false);
    });
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({1, 1, 2, 1});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, blank_kwargs, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, blank_kwargs_nocudnn, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, blank_kwargs, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, blank_kwargs_nocudnn, false);
      });
  }
}

// nonfixgamma_kwargs

/*! \brief Check V1 and V2 have same output */
TEST(BATCH_NORM, Test2DBackwardMixedV1V2Complex_cpu_cpu_nfg) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
        false, false, inputShape, nonfixgamma_kwargs, false);
    });
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_nfg) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({1, 1, 2, 1});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, nonfixgamma_kwargs, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_nfg) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, nonfixgamma_kwargs, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn, false);
      });
  }
}

// useglobalstats_kwargs

/*! \brief Check V1 and V2 have same output */
TEST(BATCH_NORM, Test2DBackwardMixedV1V2Complex_cpu_cpu_ugs) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, false, inputShape, useglobalstats_kwargs, false);
      dumpF(&std::cout, bi);
      dumpB(&std::cout, bi);
    });
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_ugs) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({2, 3, 2, 2});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, useglobalstats_kwargs, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_ugs) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, useglobalstats_kwargs, false);
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn, false);
      });
  }
}

#endif  // MXNET_USE_CUDA


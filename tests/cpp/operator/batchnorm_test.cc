/*!
 * Copyright (c) 2017 by Contributors
 * \file batchnorm_test.cc
 * \brief operator unit test utility functions
 * \author Chris Olivier
*/
#include <cstdio>
#include <dmlc/logging.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/batch_norm-inl.h"
#include "../../src/operator/batch_norm_v1-inl.h"
#include "test_op.h"
#include "executor/exec_pass.h"

using namespace mxnet;

#define MXNET_DUMP_C  0
#define DISABLE_VALIDATION 0 /* If performance profiling, may do things
                              * that cause validation to fail */

static constexpr int BATCH_SIZE = 2;
static constexpr int CHANNELS = 3;
static constexpr int DEPTH = 2;
static constexpr int DH = 3;
static constexpr int DW = 2;

//static constexpr int BATCH_SIZE = 1;
//static constexpr int CHANNELS = 1;
//static constexpr int DEPTH = 1;
//static constexpr int DH = 2;
//static constexpr int DW = 2;

static constexpr int TIMING_BATCH_SIZE = 128;
static constexpr int TIMING_CHANNELS = 3;
static constexpr int TIMING_DEPTH = 2;
static constexpr int TIMING_DH = 28;
static constexpr int TIMING_DW = 28;

/*! \brief Validate batch norm test outputs */
template<typename DType>
class BatchNormValidator : public test::op::Validator<DType>
{
  typedef test::op::Validator<DType> Super;
  using Super::compare;

  /*! \brief Only static functions in this class */
  BatchNormValidator() = delete;

  /*! \brief Check batch norm output - 1D */
  static void checkBatchNorm1D(const TBlob *blob) {
    const size_t dim = blob->ndim();
    CHECK_EQ(dim, 3U);

    const size_t num = blob->shape_[0]; // batch size
    const size_t channels = blob->shape_[1];
    const size_t length = blob->shape_[2];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      DType sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < length; ++k) {
          const DType data = test::data_at<DType>(blob, {i, j, k});
          sum += data;
          var += data * data;
          ++itemCount;
        }
      }

      // not channels
      sum /= length * num;
      var /= length * num;

      if(itemCount > 1) {
        // Due to eps, for a small number of entries, the error will be a bit higher for one pass
        const DType kErrorBound = Super::errorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if(!Super::isNear(0, sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(1, var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 2D */
  static void checkBatchNorm2D(const TBlob *blob) {
    const size_t dim = blob->ndim();
    CHECK_EQ(dim, 4U);

    const size_t num = blob->shape_[0]; // batch size
    const size_t channels = blob->shape_[1];
    const size_t height = blob->shape_[2];
    const size_t width = blob->shape_[3];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      DType sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < height; ++k) {
          for (size_t l = 0; l < width; ++l) {
            const DType data = test::data_at<DType>(blob, {i, j, k, l});
            sum += data;
            var += data * data;
            ++itemCount;
          }
        }
      }

      // not channels
      sum /= height * width * num;
      var /= height * width * num;

      if(itemCount > 1) {
        const DType kErrorBound = Super::errorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if(!Super::isNear(0, sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(1, var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 3D */
  static void checkBatchNorm3D(const TBlob *blob) {
    const size_t dim = blob->ndim();
    CHECK_EQ(dim, 5U);
    const size_t num = blob->shape_[0]; // batch size
    const size_t channels = blob->shape_[1];
    const size_t depth = blob->shape_[2];
    const size_t height = blob->shape_[3];
    const size_t width = blob->shape_[4];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      DType sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t d = 0; d < depth; ++d) {
          for (size_t k = 0; k < height; ++k) {
            for (size_t l = 0; l < width; ++l) {
              const DType data = test::data_at<DType>(blob, {i, j, d, k, l});
              sum += data;
              var += data * data;
              ++itemCount;
            }
          }
        }
      }
      // not channels
      sum /= depth * height * width * num;
      var /= depth * height * width * num;

      if(itemCount > 1) {
        const DType kErrorBound = Super::errorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if(!Super::isNear(0, sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(1, var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1";
        }
      }
    }
  }

 public:

  /*! \brief Check batch norm output */
  template<typename BNOperatorProp>
  static void validateForward(const BNOperatorProp& data) {
    const TBlob& outputBlob = data.c_.blob_output_vec_[mxnet::op::batchnorm::kData];
    switch(outputBlob.ndim()) {
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
  static void compare(const test::op::OpInfo<PropType1, DType>& info_1,
               const test::op::OpInfo<PropType2, DType>& info_2) {
    // Input
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kInput, op::batchnorm::kData));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kInput, op::batchnorm::kGamma));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kInput, op::batchnorm::kBeta));
    // Output
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kOutput, op::batchnorm::kOut));
    CHECK_EQ(info_2.prop_->getParam().use_global_stats, info_1.prop_->getParam().use_global_stats);

#if MXNET_USE_CUDNN != 1 /* CUDNN takes a slightly different approach here on first pass */
    // Aux
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kAux, op::batchnorm::kMovingMean));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType>::kAux, op::batchnorm::kMovingVar));
#endif
    if(!info_2.prop_->getParam().use_global_stats) {
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kOutput, op::batchnorm::kMean));
#if !MXNET_USE_CUDNN  /* CUDNN operator stores invstd instead of variance */
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kOutput, op::batchnorm::kVar));
#endif
      // InGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kInGrad, op::batchnorm::kData));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kInGrad, op::batchnorm::kGamma, true));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kInGrad, op::batchnorm::kBeta, true));
      // OutGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kOutGrad, op::batchnorm::kData));
    }
  }

};

/*! \brief BatchNorm-specific test data  */
template <typename DType>
class BNOperatorData : public test::op::BasicOperatorData<DType> {

 public:

  BNOperatorData(const bool isGPU, const TShape& inputShape)
    : test::op::BasicOperatorData<DType>(isGPU, inputShape) {
  }

  virtual void resetForward() override {
    // Init input data
    DType val = 1;
    test::patternFill<DType>(&this->c_.blob_input_vec_[mxnet::op::batchnorm::kData],
                             [&val]{ return val++; });
    test::fill(this->c_.blob_input_vec_[mxnet::op::batchnorm::kGamma], DType(1));  // weights
    test::fill(this->c_.blob_input_vec_[mxnet::op::batchnorm::kBeta], DType(0));   // bias

    // Init the moving data (all mean = 0, all var = 1)
    test::fill(this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingMean], DType(0));
    test::fill(this->c_.blob_aux_states_[mxnet::op::batchnorm::kMovingVar], DType(1));

    for(size_t i = 0, n = this->c_.blob_output_vec_.size(); i < n; ++i) {
      test::fill(this->c_.blob_output_vec_[i], DType(0.1234));
    }
  }

  virtual void resetBackward() override {
    DType val = .001;
    test::patternFill<DType>(&this->c_.blob_out_grad_[mxnet::op::batchnorm::kOut],
                             [&val]{ return val++; });
    test::try_fill(this->c_.blob_out_grad_, mxnet::op::batchnorm::kGamma, DType(0.1));  // weights
    test::try_fill(this->c_.blob_out_grad_, mxnet::op::batchnorm::kBeta,  DType(0.1));  // bias

    test::fill(this->c_.blob_in_grad_[mxnet::op::batchnorm::kData],  DType(0));    // the data
    test::fill(this->c_.blob_in_grad_[mxnet::op::batchnorm::kGamma], DType(0));    // weights
    test::fill(this->c_.blob_in_grad_[mxnet::op::batchnorm::kBeta],  DType(0));    // bias
  }

};

static const test::op::kwargs_t blank_kwargs;
static const test::op::kwargs_t nonfixgamma_kwargs = { {"fix_gamma", "False"} };
static const test::op::kwargs_t useglobalstats_kwargs = { {"use_global_stats", "True"} };

#if !DISABLE_VALIDATION
static bool isUGS(const test::op::kwargs_t& kwargs) {
  for(test::op::kwargs_t::const_iterator i = kwargs.begin(),
        e = kwargs.end(); i != e; ++i) {
    if(!i->first.compare("use_global_stats")) {
      return i->second.compare("True") == 0;
    }
  }
  return false;
}
#endif  // DISABLE_VALIDATION

/*! \brief Test batch norm operator forward pass */
template<typename OperatorProp, typename DType>
static test::op::OpInfo<OperatorProp, DType> testBatchNormOperatorForward(
  bool isGPU,
  const TShape& inputShape,
  const std::vector<std::pair<std::string, std::string> >& kwargs,
  const size_t count = 1) {

#if MXNET_USE_CUDA
  if(isGPU && !test::unitTestsWithCuda) {
    LOG(INFO) << "GPU not found, running test as non-GPU";
  }
#else
  isGPU = false;
#endif

  test::op::OpInfo<OperatorProp, DType> info = test::op::createOpAndInfo<
    OperatorProp,
    BNOperatorData<DType>,
    DType>(isGPU, inputShape, kwargs);

  info.data_->initForward(*info.prop_, info.in_type_);

  if(mxnet::op::Callbacker<Operator> *callbacker =
    dynamic_cast<mxnet::op::Callbacker<Operator> *>(info.data_->op())) {
      callbacker->setCallback(
        [](const std::string& label, const Operator &op, const TBlob &blob) {
          if(test::debugOutput) {
            std::cout << label << ": " << std::endl << blob << std::endl << std::flush;
          }
        }
      );
  }

  info.data_->forward(count);

#if !DISABLE_VALIDATION
  if(!isUGS(kwargs) && count == 1) {
    BatchNormValidator<DType>::validateForward(*info.data_);
  }
#endif

  return info;
}

/*! \brief Test batch norm operator backward pass */
template<typename DType, typename OperatorProp>
static test::op::OpInfo<OperatorProp, DType> runOperatorBackward(
  test::op::OpInfo<OperatorProp, DType> &info,
  const size_t count = 1) {

  info.data_->initBackward(*info.prop_, info.in_type_);

  info.data_->backward(count);
  return info;
}

#define PRT(__obj$, __bvt$, __idx$) \
  test::op::BasicOperatorData<DType>::bvt2String(test::op::BasicOperatorData<DType>::__bvt$) \
    << ": " #__idx$ << ": " << (__obj$).data_->getBlobVect(test::op::BasicOperatorData<DType>::__bvt$)[__idx$]

template<typename StreamType, typename Prop, typename DType>
static StreamType& dumpF(StreamType& os, const size_t x, const test::op::OpInfo<Prop, DType>& prop) {
  if(test::debugOutput) {
    os << std::endl;
    if (x) {
      os << "=============================" << std::endl;
      os << "= " << x << std::endl;
      os << "=============================" << std::endl;
    }

    os << PRT(prop, kInput, op::batchnorm::kData) << std::endl;
    os << PRT(prop, kInput, op::batchnorm::kGamma) << std::endl;
    os << PRT(prop, kInput, op::batchnorm::kBeta) << std::endl;

    os << PRT(prop, kAux, op::batchnorm::kMovingMean) << std::endl;
    os << PRT(prop, kAux, op::batchnorm::kMovingVar) << std::endl;

    os << PRT(prop, kOutput, op::batchnorm::kOut) << std::endl;
    os << PRT(prop, kOutput, op::batchnorm::kMean) << std::endl;
    os << PRT(prop, kOutput, op::batchnorm::kVar) << std::endl;
  }
  return os;
}

template<typename StreamType, typename Prop, typename DType>
static StreamType& dumpB(StreamType& os, const size_t x, const test::op::OpInfo<Prop, DType>& prop) {
  if(test::debugOutput) {
    os << std::endl;
    if (x) {
      os << "=============================" << std::endl;
      os << "= " << x << std::endl;
      os << "=============================" << std::endl;
    }

    os << PRT(prop, kAux, op::batchnorm::kMovingMean) << std::endl;
    os << PRT(prop, kAux, op::batchnorm::kMovingVar) << std::endl;

    os << PRT(prop, kInGrad, op::batchnorm::kData) << std::endl;
    os << PRT(prop, kInGrad, op::batchnorm::kGamma) << std::endl;
    os << PRT(prop, kInGrad, op::batchnorm::kBeta) << std::endl;

    os << PRT(prop, kOutGrad, op::batchnorm::kOut) << std::endl;
  }
  return os;
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType>
static StreamType& dumpF(StreamType& os, const test::op::OpInfoPair<Prop1, Prop2, DType>& bi) {
  return dumpF(dumpF(os, 1, bi.info_1_), 2, bi.info_2_);
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType>
static StreamType& dumpB(StreamType& os, const test::op::OpInfoPair<Prop1, Prop2, DType>& bi) {
  return dumpB(dumpB(os, 1, bi.info_1_), 2, bi.info_2_);
}

template<typename OperatorProp1, typename OperatorProp2, typename DType>
inline test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType> testBackward(
  const bool isGPU1,
  const bool isGPU2,
  const TShape &inputShape,
  const test::op::kwargs_t& kwargs,
  const bool dumpC,
  const size_t count = 1) {

  test::op::OpInfo<OperatorProp1, DType> info_1 =
    testBatchNormOperatorForward<OperatorProp1, DType>(isGPU1, inputShape, kwargs, count);

  test::op::OpInfo<OperatorProp2, DType> info_2 =
    testBatchNormOperatorForward<OperatorProp2, DType>(isGPU2, inputShape, kwargs, count);

  dumpF(std::cout, 1, info_1);
  dumpF(std::cout, 2, info_2);

  // Check that everything is the same after the forward pass
  BatchNormValidator<DType>::compare(info_1, info_2);

  EXPECT_TRUE(test::op::Validator<DType>::compare(*info_1.data_, *info_2.data_,
                                                  test::op::BasicOperatorData<DType>::kInput,
                                                  op::batchnorm::kData));

  info_1.data_->initBackward(*info_1.prop_, info_1.in_type_);
  info_2.data_->initBackward(*info_2.prop_, info_2.in_type_);

  // return backward
  runOperatorBackward<DType>(info_1, count);
  runOperatorBackward<DType>(info_2, count);

  dumpB(std::cout, 1, info_1);
  dumpB(std::cout, 2, info_2);


  // Check that everything is the same after the backward pass
  BatchNormValidator<DType>::compare(info_1, info_2);

  if(dumpC) {
    info_1.data_->dumpC(std::cerr, "BN_Test2DBackward");
  }

  return  { info_1, info_2 };
}

template<typename OperatorProp1, typename OperatorProp2, typename DType>
inline test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType> testBackward(
  const bool isGPU,
  const TShape &inputShape,
  const test::op::kwargs_t kwargs,
  const bool dumpC = false) {

  return testBackward<OperatorProp1, OperatorProp2, DType>(isGPU, isGPU, inputShape, kwargs, dumpC);
}

/*
 * Forward tests
 */
TEST(BATCH_NORM, Test2DForwardV1) {
  test::op::OpInfo<op::BatchNormV1Prop, float> opInfo =
    testBatchNormOperatorForward<op::BatchNormV1Prop, float>(false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
  dumpF(std::cout, 0, opInfo);
}

TEST(BATCH_NORM, Test1DForward) {
  testBatchNormOperatorForward<op::BatchNormProp, float>(false, {BATCH_SIZE, CHANNELS, DW}, blank_kwargs);
}

TEST(BATCH_NORM, Test2DForward) {
  test::op::OpInfo<op::BatchNormProp, float> opInfo =
    testBatchNormOperatorForward<op::BatchNormProp, float>(false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
  dumpF(std::cout, 0, opInfo);
}

TEST(BATCH_NORM, Test3DForward) {
  testBatchNormOperatorForward<op::BatchNormProp, float>(false, {BATCH_SIZE, CHANNELS, DEPTH, DH, DW}, blank_kwargs);
}

template<typename PropType, typename DType>
static void timingTest(const std::string& label,
                       const bool isGPU,
                       const bool stochastic,
                       const int dim = 0,
                       const bool includeBackward = true,
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

    test::op::OpInfo<PropType, DType> info;
    switch (D) {
      case 0:
        info = testBatchNormOperatorForward<PropType, DType>(isGPU,
                                                             {batchSize, channels, width},
                                                             blank_kwargs, count);
        break;
      case 1:
        info = testBatchNormOperatorForward<PropType, DType>(isGPU,
                                                             {batchSize, channels, height, width},
                                                             blank_kwargs, count);
        break;
      case 2:
        info = testBatchNormOperatorForward<PropType, DType>(isGPU,
                                                             {batchSize, channels, depth, height, width},
                                                             blank_kwargs, count);
        break;
      default:
        CHECK(false) << "rangedRand() returned unexpected value";
    }
    if (info.data_.get()) {
      if (includeBackward) {
        runOperatorBackward<DType>(info, count);
      }
      timing += info.data_->timing_;
    }
  } while(0);

  timing.print(std::cout, label);
  std::cout << std::endl << std::flush;
}

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#define GPU_TEST_DIMENSIONS  2  /* Only support 2D */
#else
#define GPU_TEST_DIMENSIONS  0  /* Allow stochastic */
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5

/*! \brief Stress-test random batch size/channels/dimension(s) */
TEST(BATCH_NORM, TestStochasticTiming_2D) {
  timingTest<op::BatchNormProp,   float>("RANDOM: BatchNormProp<cpu>", false, true, GPU_TEST_DIMENSIONS);
#if MXNET_USE_CUDA
  if(test::unitTestsWithCuda) {
    timingTest<op::BatchNormProp,   float>("RANDOM: BatchNormProp<gpu>", true, true, GPU_TEST_DIMENSIONS);
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
  timingTest<op::BatchNormV1Prop, float>("BatchNormV1Prop<cpu> 2D", false, false, 2, true, THISCOUNT);
  timingTest<op::BatchNormProp,   float>("BatchNormProp<cpu> 2D", false, false, 2, true, THISCOUNT);
#if MXNET_USE_CUDA
  if(test::unitTestsWithCuda) {
    timingTest<op::BatchNormV1Prop, float>("BatchNormV1Prop<gpu> 2D", true, false, 2, true, THISCOUNT);
    timingTest<op::BatchNormProp,   float>("BatchNormProp<gpu> 2D", true, false, 2, true, THISCOUNT);
  }
#endif
}

/**
 * Backward tests (generally include forward tests as well)
 */

template<typename DType>
struct BothInfo
{
  test::op::OpInfo<op::BatchNormV1Prop, DType>  info_v1_;
  test::op::OpInfo<op::BatchNormProp, DType>    info_;
};

TEST(BATCH_NORM, TestBackward2D_Simple) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(
      false, inputShape, blank_kwargs);  // Keep it simple
}

TEST(BATCH_NORM, TestBackward2D_SimpleNFG) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(false,
                                                                inputShape,
                                                                nonfixgamma_kwargs);
}

TEST(BATCH_NORM, Test2DBackward_Complex) {
  typedef float DType;
  test::ScopeSet<bool> noDebugOutput(test::debugOutput, false);
  const TShape inputShape({9, 14, 16, 91});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(false, inputShape, blank_kwargs);
}

TEST(BATCH_NORM, Test2DBackward2DPlusLoadAndCompareLogic) {

  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(false, inputShape, blank_kwargs);

  // Expected data state when running forward+backward starting with default values
  // Note: This data structure generated by dumpC()
  static const std::vector< std::vector< std::vector<float> > > ___BN_Test2DBackward_data_shape_1_1_2_1___ =
    {
      { /* kInput */
        { 1.00000f, 2.00000f },
        { 1.00000f },
        { 0.00000f }
      },
      { /* kOutput */
        { -0.99801f, 0.99801f },
        { 1.50000f },
        { 0.25000f }
      },
      { /* kAux */
        { 0.15000f },
        { 0.92500f }
      },
      { /* kInGrad */
        { -0.00398f, 0.00398f },
        { 0.00000f },
        { 1.00200f }
      },
      { /* kOutGrad */
        { 0.00100f, 1.00100f }
      }
    };

  // Test loaded data agsinst calculated data
  test::op::OpInfo<op::BatchNormProp, DType> info_checkLoad =
    test::op::createOpAndInfo<op::BatchNormProp, BNOperatorData<DType>, DType>(false, inputShape, blank_kwargs);
  info_checkLoad.data_->initForward(*info_checkLoad.prop_, info_checkLoad.in_type_);
  info_checkLoad.data_->initBackward(*info_checkLoad.prop_, info_checkLoad.in_type_);
  info_checkLoad.data_->load(___BN_Test2DBackward_data_shape_1_1_2_1___);
  BatchNormValidator<DType>::compare(bi.info_1_, info_checkLoad);
}

template<typename PropType, typename DType>
void compare(const bool isGPU,
             const test::op::OpInfo<PropType, DType>& object,
             const std::vector< std::vector< std::vector<float> > >& values) {

  test::op::OpInfo<PropType, DType> info_checkLoad =
    test::op::createOpAndInfo<PropType, BNOperatorData<DType>, DType>(
      isGPU, object.data_->c_.blob_input_vec_[0].shape_, blank_kwargs);
  info_checkLoad.data_->initForward(*info_checkLoad.prop_, info_checkLoad.in_type_);
  info_checkLoad.data_->initBackward(*info_checkLoad.prop_, info_checkLoad.in_type_);
  info_checkLoad.data_->load(values);
  BatchNormValidator<DType>::compare(object, info_checkLoad);
}

TEST(BATCH_NORM, TestBackward1D_Simple) {

  typedef float DType;
  const TShape inputShape({1, 1, 2});
  test::op::OpInfo<op::BatchNormProp, DType> info =
    testBatchNormOperatorForward<op::BatchNormProp, DType>(false, inputShape, blank_kwargs);
  info.data_->initBackward(*info.prop_, info.in_type_);
  runOperatorBackward(info);

#if MXNET_DUMP_C
  //info.data_->dumpC(std::cerr, "BN_TestBackward1D_Simple");
#endif

  // Expected data state when running forward+backward starting with default values
  // Note: This data structure generated by dumpC()
  static const std::vector< std::vector< std::vector<float> > > ___BN_TestBackward1D_Simple_data_shape_1_1_2___ =
    {
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
        { 1.002f }
      },
      { /* kOutGrad */
        { 0.001f, 1.001f }
      }
    };
  compare(false, info, ___BN_TestBackward1D_Simple_data_shape_1_1_2___);
}

TEST(BATCH_NORM, TestBackward3D) {
  typedef float DType;
  const TShape inputShape({2, 3, 2, 3, 5});
  test::op::OpInfo<op::BatchNormProp, DType> info =
    testBatchNormOperatorForward<op::BatchNormProp, DType>(false, inputShape, blank_kwargs);
  info.data_->initBackward(*info.prop_, info.in_type_);
  runOperatorBackward(info);
#if MXNET_DUMP_C
  info.data_->dumpC(std::cerr, "TestBackward3D");
#endif
}

// nonfixgamma_kwargs
TEST(BATCH_NORM, Test2DBackwardMixed_cpu_cpu_nfg) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(
      false, false, inputShape, nonfixgamma_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

// useglobalstats_kwargs
TEST(BATCH_NORM, Test2DBackwardMixed_cpu_cpu_ugs) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(
      false, false, inputShape, useglobalstats_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

#if MXNET_USE_CUDA

TEST(BATCH_NORM, Test2DForwardV12D_gpu) {
  test::op::OpInfo<op::BatchNormV1Prop, float> opInfo =
    testBatchNormOperatorForward<op::BatchNormV1Prop, float>(true, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
  dumpF(std::cout, 0, opInfo);
}

TEST(BATCH_NORM, Test2DForward2D_gpu) {
  test::op::OpInfo<op::BatchNormProp, float> opInfo =
    testBatchNormOperatorForward<op::BatchNormProp, float>(true, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
  dumpF(std::cout, 0, opInfo);
}

// blank_kwargs
TEST(BATCH_NORM, Test2DBackwardMixedV1_gpu_cpu) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormV1Prop, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormV1Prop, DType>(
      false, true, inputShape, blank_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixedV1Complex_gpu_cpu) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormV1Prop, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormV1Prop, DType>(
      false, true, inputShape, blank_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, blank_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, blank_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

// nonfixgamma_kwargs

/*! \brief Check V1 and V2 have same output */
TEST(BATCH_NORM, Test2DBackwardMixedV1V2Complex_cpu_cpu_nfg) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(
      false, false, inputShape, nonfixgamma_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_nfg) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, nonfixgamma_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_nfg) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, nonfixgamma_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

// useglobalstats_kwargs

/*! \brief Check V1 and V2 have same output */
TEST(BATCH_NORM, Test2DBackwardMixedV1V2Complex_cpu_cpu_ugs) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormV1Prop, op::BatchNormProp, DType>(
      false, false, inputShape, useglobalstats_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_ugs) {
  typedef float DType;
  const TShape inputShape({1, 1, 2, 1});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, useglobalstats_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_ugs) {
  typedef float DType;
  const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
  test::op::OpInfoPair<op::BatchNormProp, op::BatchNormProp, DType> bi =
    testBackward<op::BatchNormProp, op::BatchNormProp, DType>(
      false, true, inputShape, useglobalstats_kwargs, false);
  dumpF(std::cout, bi);
  dumpB(std::cout, bi);
}

#endif  // MXNET_USE_CUDA


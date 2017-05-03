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

#define MXNET_DUMP_C  1
#define DISABLE_VALIDATION 0  // If performance profiling, may do things
                              // that cause validation to fail
static constexpr int BATCH_SIZE = 2;
static constexpr int CHANNELS = 3;
static constexpr int DEPTH = 2;
static constexpr int DH = 3;
static constexpr int DW = 2;

static constexpr int TIMING_BATCH_SIZE = 128;
static constexpr int TIMING_CHANNELS = 3;
static constexpr int TIMING_DEPTH = 2;
static constexpr int TIMING_DH = 28;
static constexpr int TIMING_DW = 28;

/*! \brief Validate batch norm test outputs */
template<typename DType, typename AccReal>
class BatchNormValidator : public test::op::Validator<DType, AccReal>
{
  typedef test::op::Validator<DType, AccReal> Super;
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
      AccReal sum = 0, var = 0;
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
        if(!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(AccReal(1), var, kErrorBound)) {
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
      AccReal sum = 0, var = 0;
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
        if(!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(AccReal(1), var, kErrorBound)) {
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
      AccReal sum = 0, var = 0;
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
        if(!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if(!Super::isNear(AccReal(1), var, kErrorBound)) {
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
  static void compare(const test::op::OpInfo<PropType1, DType, AccReal>& info_1,
                      const test::op::OpInfo<PropType2, DType, AccReal>& info_2) {
    // Input
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput, op::batchnorm::kData));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput, op::batchnorm::kGamma));
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kInput, op::batchnorm::kBeta));
    // Output
    EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                        test::op::BasicOperatorData<DType, AccReal>::kOutput, op::batchnorm::kOut));
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
                          test::op::BasicOperatorData<DType, AccReal>::kOutput, op::batchnorm::kMean));
#if !MXNET_USE_CUDNN  /* CUDNN operator stores invstd instead of variance */
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType>::kOutput, op::batchnorm::kVar));
#endif
      // InGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad, op::batchnorm::kData));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad, op::batchnorm::kGamma));
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kInGrad, op::batchnorm::kBeta));
      // OutGrad
      EXPECT_TRUE(compare(*info_1.data_, *info_2.data_,
                          test::op::BasicOperatorData<DType, AccReal>::kOutGrad, op::batchnorm::kData));
    }
  }

};

/*! \brief BatchNorm-specific test data  */
template <typename DType, typename AccReal>
class BNOperatorData : public test::op::BasicOperatorData<DType, AccReal> {

 public:

  BNOperatorData(const bool isGPU, const TShape& inputShape)
    : test::op::BasicOperatorData<DType, AccReal>(isGPU, inputShape) {
  }

  virtual void resetForward() override {
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
        if(blob.size(0) > 1) {
          blob.dptr<DTypeX>()[1] = DTypeX(3);
        }
      });
    MSHADOW_TYPE_SWITCH(
      this->c_.blob_input_vec_[mxnet::op::batchnorm::kBeta].type_flag_,
      DTypeX, {
        const TBlob& blob = this->c_.blob_input_vec_[mxnet::op::batchnorm::kBeta];
        test::fill(blob, DTypeX(1));
        if(blob.size(0) > 0) {
          blob.dptr<DTypeX>()[0] = DTypeX(3);
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

    for(size_t i = 0, n = this->c_.blob_output_vec_.size(); i < n; ++i) {
      const int dtype = this->c_.blob_output_vec_[i].type_flag_;
      MSHADOW_TYPE_SWITCH(dtype, DTypeX,
                          { test::fill(this->c_.blob_output_vec_[i], DTypeX(0.1234)); });
    }
  }

  virtual void resetBackward() override {
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
  {"use_global_stats", "True"}, {"cudnn_off", "True"}, {"mkl_off", "True"} };
static const test::op::kwargs_t nfs_ugd_kwargs = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}};
static const test::op::kwargs_t nfs_ugd_kwargs_nocudnn = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}, {"cudnn_off", "True"}  };

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
template<typename OperatorProp, typename DType, typename AccReal>
static test::op::OpInfo<OperatorProp, DType, AccReal> TestBatchNormOperatorForward(
  test::op::OpInfo<OperatorProp, DType, AccReal>& opInfo,
  const std::vector<std::pair<std::string, std::string> >& kwargs,
  const size_t count = 1) {

  opInfo.data_->initForward(*opInfo.prop_, opInfo.in_type_);

  opInfo.data_->forward(count);

#if !DISABLE_VALIDATION
  if(!isUGS(kwargs) && count == 1) {
    BatchNormValidator<DType, AccReal>::validateForward(*opInfo.data_);
  }
#endif

  return opInfo;
}

/*! \brief Test batch norm operator forward pass */
template<typename OperatorProp, typename DType, typename AccReal>
static test::op::OpInfo<OperatorProp, DType, AccReal> TestBatchNormOperatorForward(
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

  test::op::OpInfo<OperatorProp, DType, AccReal> info = test::op::createOpAndInfoF<
    OperatorProp, BNOperatorData<DType, AccReal>, DType, AccReal>(isGPU, inputShape, kwargs);

  info.data_->initForward(*info.prop_, info.in_type_);

  info.data_->forward();

#if !DISABLE_VALIDATION
  if(!isUGS(kwargs)) {
    BatchNormValidator<DType, AccReal>::validateForward(*info.data_);
  }
#endif

  return info;
}

/*! \brief Test batch norm operator backward pass */
template<typename DType, typename AccReal, typename OperatorProp>
static test::op::OpInfo<OperatorProp, DType, AccReal> runOperatorBackward(
  test::op::OpInfo<OperatorProp, DType, AccReal> &info,
  const size_t count = 1) {

  info.data_->initBackward(*info.prop_, info.in_type_);

  info.data_->backward(count);
  return info;
}

template<typename StreamType, typename DType, typename AccReal>
static StreamType& PRT(
  StreamType& os,
  const test::op::BasicOperatorData<DType, AccReal>& obj,
  const typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType bvt,
  const size_t idx) {
  os << test::op::BasicOperatorData<DType, AccReal>::bvt2String(bvt) << ": " << idx
     << ": ";
  const TBlob& blob = obj.getBlobVect(bvt)[idx];
  MSHADOW_REAL_TYPE_SWITCH(blob.type_flag_, DTypeX, { test::print_blob<DTypeX>(os, blob); });
  return os;
}

template<typename StreamType, typename Prop, typename DType, typename AccReal>
static StreamType& dumpF(StreamType& os,
                         const test::op::OpInfo<Prop, DType, AccReal>& prop,
                         const size_t x = 0) {
  if(test::debugOutput) {
    os << std::endl;
    if (x) {
      os << "=============================" << std::endl;
      os << "= " << x << std::endl;
      os << "=============================" << std::endl;
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
  return os;
}

template<typename StreamType, typename Prop, typename DType, typename AccReal>
static StreamType& dumpB(StreamType& os,
                         const test::op::OpInfo<Prop, DType, AccReal>& prop,
                         const size_t x = 0) {
  if(test::debugOutput) {
    os << std::endl;
    if (x) {
      os << "=============================" << std::endl;
      os << "= " << x << std::endl;
      os << "=============================" << std::endl;
    }

    typedef typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType BlobVectorType;
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kData);
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kGamma);
    PRT(os, *prop.data_, BlobVectorType::kInGrad, op::batchnorm::kBeta);

    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingMean);
    PRT(os, *prop.data_, BlobVectorType::kAux, op::batchnorm::kMovingVar);

    PRT(os, *prop.data_, BlobVectorType::kOutGrad, op::batchnorm::kOut);
  }
  return os;
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType, typename AccReal>
static StreamType& dumpF(StreamType& os,
                         const test::op::OpInfoPair<Prop1, Prop2, DType, AccReal>& bi) {
  return dumpF(dumpF(os, bi.info_1_, 1), bi.info_2_, 2);
}

template<typename StreamType, typename Prop1, typename Prop2, typename DType, typename AccReal>
static StreamType& dumpB(StreamType& os,
                         const test::op::OpInfoPair<Prop1, Prop2, DType, AccReal>& bi) {
  return dumpB(dumpB(os, bi.info_1_, 1), bi.info_2_, 2);
}

template<typename OperatorProp1, typename OperatorProp2, typename DType, typename AccReal>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType, AccReal> testForwardAndBackward(
  const bool isGPU1,
  const bool isGPU2,
  const TShape &inputShape,
  const test::op::kwargs_t& kwargs,
  const bool dumpC,
  const size_t count = 1) {

  test::op::OpInfo<OperatorProp1, DType, AccReal> info_1 =
    TestBatchNormOperatorForward<OperatorProp1, DType, AccReal>(isGPU1, inputShape,
                                                                kwargs, count);

  test::op::OpInfo<OperatorProp2, DType, AccReal> info_2 =
    TestBatchNormOperatorForward<OperatorProp2, DType, AccReal>(isGPU2, inputShape,
                                                                kwargs, count);

  dumpF(std::cout, info_1, 1);
  dumpF(std::cout, info_2, 2);

  // Check that everything is the same after the forward pass
  BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

  test::op::Validator<DType, AccReal>::compare(*info_1.data_, *info_2.data_,
                                               test::op::BasicOperatorData<DType, AccReal>::kInput,
                                               op::batchnorm::kData);

  info_1.data_->initBackward(*info_1.prop_, info_1.in_type_);
  info_2.data_->initBackward(*info_2.prop_, info_2.in_type_);

  // return backward
  runOperatorBackward(info_1, count);
  runOperatorBackward(info_2, count);

  dumpB(std::cout, info_1, 1);
  dumpB(std::cout, info_2, 2);


  // Check that everything is the same after the backward pass
  BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

  if(dumpC) {
    info_1.data_->dumpC(std::cerr, "BN_Test2DBackward");
  }

  return  { info_1, info_2 };
}

template<typename OperatorProp1, typename OperatorProp2, typename DType, typename AccReal>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, DType, AccReal>
testForwardAndBackward(const bool isGPU,
                       const TShape &inputShape,
                       const test::op::kwargs_t kwargs,
                       const bool dumpC = false) {

  return testForwardAndBackward<OperatorProp1, OperatorProp2, DType, AccReal>(
    isGPU,
    isGPU,
    inputShape,
    kwargs,
    dumpC);
}

template<typename DType, typename AccReal>
static test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>
testBNForwardAndBackward(const bool isGPU,
                         const TShape &inputShape,
                         const test::op::kwargs_t kwargs,
                         const bool dumpC = false) {

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
      auto infoA = testBNForwardAndBackward<DType, AccReal>(false,
                                                            {BATCH_SIZE, CHANNELS, DH, DW},
                                                            blank_kwargs);
      dumpF(std::cout, infoA);
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

TEST(BATCH_NORM, Test2DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        auto opInfoFloatH = TestBatchNormOperatorForward<op::BatchNormProp, DType, AccReal>(
          false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
        dumpF(std::cout, opInfoFloatH);
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

    test::op::OpInfo<PropType, DType, AccReal> info;
    switch (D) {
      case 0:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, width},
          blank_kwargs, count);
        break;
      case 1:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, height, width},
          blank_kwargs, count);
        break;
      case 2:
        info = TestBatchNormOperatorForward<PropType, DType, AccReal>(
          isGPU,
          {batchSize, channels, depth, height, width},
          blank_kwargs, count);
        break;
      default:
        CHECK(false) << "rangedRand() returned unexpected value";
    }
    if (info.data_.get()) {
      if (includeBackward) {
        runOperatorBackward<DType, AccReal>(info, count);
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
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      timingTest<op::BatchNormProp, DType, AccReal>("RANDOM: BatchNormProp<cpu>",
                                                    false, true,
                                                    GPU_TEST_DIMENSIONS); });
#if MXNET_USE_CUDA
  if(test::unitTestsWithCuda) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      mshadow::kFloat32, DType, AccReal,
      {
        timingTest<op::BatchNormProp, DType, AccReal>("RANDOM: BatchNormProp<gpu>",
                                                      true, true,
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
                                                      false, false, 2, true, THISCOUNT);
      timingTest<op::BatchNormProp, DType, AccReal>("BatchNormProp<cpu> 2D",
                                                    false, false, 2, true, THISCOUNT);
#if MXNET_USE_CUDA
      if(test::unitTestsWithCuda) {
        timingTest<op::BatchNormV1Prop, DType, AccReal>("BatchNormV1Prop<gpu> 2D",
                                                        true, false, 2, true, THISCOUNT);
        timingTest<op::BatchNormProp, DType, AccReal>("BatchNormProp<gpu> 2D",
                                                      true, false, 2, true, THISCOUNT);
      }
#endif
    });
}

/**
 * Backward tests (generally include forward tests as well)
 */

template<typename DType, typename AccReal>
struct BothInfo
{
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

template<typename DType, typename AccReal>
static void testEx(const test::op::kwargs_t& kwargs, const size_t count) {

  TShape shapes[2] = {2, 3};
  const TShape inputShape({2, 3});

  test::op::OpInfo<op::BatchNormV1Prop, DType, AccReal> info_1 = test::op::createOpAndInfoF<
    op::BatchNormV1Prop,
    BNOperatorData<DType, AccReal>,
    DType, AccReal>(false, inputShape, kwargs);

  test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_2 = test::op::createOpAndInfoF<
    op::BatchNormProp, BNOperatorData<DType, AccReal>, DType, AccReal>(
    false, inputShape, kwargs);

  info_1.data_->initForward(*info_1.prop_, info_1.in_type_);
  info_2.data_->initForward(*info_1.prop_, info_1.in_type_);
  info_1.data_->initBackward(*info_1.prop_, info_1.in_type_);
  info_2.data_->initBackward(*info_1.prop_, info_1.in_type_);

  TBlob& blob1 = info_1.data_->c_.blob_input_vec_[op::batchnorm::kData];
  test::data_ref<DType>(&blob1, {0, 0}) = -0.05f;
  test::data_ref<DType>(&blob1, {0, 1}) = -0.19f;
  test::data_ref<DType>(&blob1, {0, 2}) = 0.02f;
  test::data_ref<DType>(&blob1, {1, 0}) = -0.12f;
  test::data_ref<DType>(&blob1, {1, 1}) = 0.06f;
  test::data_ref<DType>(&blob1, {1, 2}) = -0.01f;

  TBlob& blob2 = info_2.data_->c_.blob_input_vec_[op::batchnorm::kData];
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

  for(size_t x = 0; x < count; ++x) {
    info_1.data_->forward();
    info_2.data_->forward();

    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

    info_1.data_->backward();
    info_2.data_->backward();

    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);
  }

}


TEST(BATCH_NORM, TestBackward2D_SimpleEx) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      testEx<DType, AccReal>(blank_kwargs, 2);
      testEx<DType, AccReal>(nonfixgamma_kwargs, 2);
      testEx<DType, AccReal>(useglobalstats_kwargs, 2);
      testEx<DType, AccReal>(nfs_ugd_kwargs, 2);
    });
}

TEST(BATCH_NORM, TestBackward2D_SimpleNFG) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(false,
                                                                                       inputShape,
                                                                                       nonfixgamma_kwargs);
    });
}

TEST(BATCH_NORM, Test2DBackward_Complex) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      test::ScopeSet<bool> noDebugOutput(test::debugOutput, false);
      const TShape inputShape({9, 14, 16, 91});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, inputShape, blank_kwargs);
    });
}

TEST(BATCH_NORM, Test2DBackward2DPlusLoadAndCompareLogic) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({1, 1, 2, 1});
      test::op::OpInfoPair<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal> bi =
        testForwardAndBackward<op::BatchNormV1Prop, op::BatchNormProp, DType, AccReal>(
          false, inputShape, blank_kwargs);

#if MXNET_DUMP_C
      bi.info_1_.data_->dumpC(&std::cerr, "Test2DBackward2DPlusLoadAndCompareLogic");
#endif

      // Expected data state when running forward+backward starting with default values
      // Note: This data structure generated by dumpC()
      static const std::vector< std::vector< std::vector<DType> > >
        ___Test2DBackward2DPlusLoadAndCompareLogic_data_shape_1_1_2_1___ =
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
            { -0.00397611f, 0.00397611f },
            { 0.0f },
            { 2.998f }
          },
          { /* kOutGrad */
            { 0.999f, 1.999f }
          }
        };
      // Test loaded data agsinst calculated data
      test::op::OpInfo<op::BatchNormProp, DType, AccReal> info_checkLoad =
        test::op::createOpAndInfoF<op::BatchNormProp, BNOperatorData<DType, AccReal>,
          DType, AccReal>(false, inputShape, blank_kwargs);
      info_checkLoad.data_->initForward(*info_checkLoad.prop_, info_checkLoad.in_type_);
      info_checkLoad.data_->initBackward(*info_checkLoad.prop_, info_checkLoad.in_type_);
      info_checkLoad.data_->load(___Test2DBackward2DPlusLoadAndCompareLogic_data_shape_1_1_2_1___);
      BatchNormValidator<DType, AccReal>::compare(bi.info_1_, info_checkLoad);
    });
}

template<typename PropType, typename DType, typename AccReal>
void compare(const bool isGPU,
             const test::op::OpInfo<PropType, DType, AccReal>& object,
             const std::vector< std::vector< std::vector<DType> > >& values) {

  test::op::OpInfo<PropType, DType, AccReal> info_checkLoad =
    test::op::createOpAndInfoF<PropType, BNOperatorData<DType, AccReal>, DType, AccReal>(
      isGPU, object.data_->c_.blob_input_vec_[0].shape_, blank_kwargs);
  info_checkLoad.data_->initForward(*info_checkLoad.prop_, info_checkLoad.in_type_);
  info_checkLoad.data_->initBackward(*info_checkLoad.prop_, info_checkLoad.in_type_);
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
      info.data_->dumpC(std::cerr, "BN_TestBackward1D_Simple");
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
      info.data_->initBackward(*info.prop_, info.in_type_);
      runOperatorBackward(info);
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
      dumpF(std::cout, bi);
      dumpB(std::cout, bi);
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
      dumpF(std::cout, bi);
      dumpB(std::cout, bi);
    });
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
      dumpF(std::cout, bi);
      dumpB(std::cout, bi);
    });
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_ugs) {
  for (int type :  v2_types) {
  //for (int type :  { 0 }) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        //const TShape inputShape({1, 1, 2, 1});
        //const TShape inputShape({1, 2, 2, 1});
        const TShape inputShape({2, 3, 2, 2});
        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn, false, 1);
//        testForwardAndBackward<op::BatchNormProp, op::BatchNormProp, DType, AccReal>(
//          false, true, inputShape, useglobalstats_kwargs, false);
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


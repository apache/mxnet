/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_DROPOUT_INL_H_
#define MXNET_OPERATOR_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "./operator_common.h"
#include "./mshadow_op.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
}  // namespace dropout

namespace mxnet {
namespace op {

#if defined(USE_MKL) && defined(_OPENMP)
static void bernoulli_generate(int n, double p, int* r) {
  int seed = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  int nthr = omp_get_max_threads();
# pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int my_offset = ithr * avg_amount;
    const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
}
#endif  // USE_MKL && _OPENMP

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
class DropoutOp : public Operator {
 public:
  explicit DropoutOp(DropoutParam param) {
    this->pkeep_ = 1.0f - param.p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2U);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
    if (ctx.is_train) {
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
#if defined(USE_MKL) && defined(_OPENMP)
      DType* outptr = out.dptr_;
      DType* dataptr = data.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);
      int count = mask.shape_[0]*mask.shape_[1];
      bernoulli_generate(count, this->pkeep_, maskptr);
  #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        outptr[i] = dataptr[i] * maskptr[i];
      }
#else
      Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
      mask = tcast<DType>(F<mshadow_op::threshold>(
             prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_));
      Assign(out, req[dropout::kOut], data * mask);
#endif  // USE_MKL && _OPENMP
    } else {
      Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
#if defined(USE_MKL) && defined(_OPENMP)
      DType* ingradptr = gdata.dptr_;
      DType* outgradptr = grad.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);

      int count = mask.shape_[0]*mask.shape_[1];

  #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        ingradptr[i] = outgradptr[i] * maskptr[i];
      }
#else  // USE_MKL && _OPENMP
      Assign(gdata, req[dropout::kData], grad * mask);
#endif  // USE_MKL && _OPENMP
  }

 private:
  real_t pkeep_;
};  // class DropoutOp


template<typename xpu>
Operator *CreateOp(DropoutParam param, int dtype);

#if DMLC_USE_CXX11
class DropoutProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = in_type->at(0);

    if (dtype == -1) {
      LOG(FATAL) << "input type to dropout is not specified.";
      return false;
    }

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DropoutProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Dropout";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout::kOut], out_data[dropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout::kOut], in_grad[dropout::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[dropout::kData], out_data[dropout::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  DropoutParam param_;
};  // class DropoutProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_INL_H_

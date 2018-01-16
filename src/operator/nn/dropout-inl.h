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
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_NN_DROPOUT_INL_H_
#define MXNET_OPERATOR_NN_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../random/sampler.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
enum DropoutOpMode {kTraining, kAlways};
}  // namespace dropout

namespace mxnet {
namespace op {

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", dropout::kTraining)
    .add_enum("always", dropout::kAlways)
    .set_default(dropout::kTraining)
    .describe("Whether to only turn on dropout during training or to also turn on for inference.");
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
class DropoutOp : public Operator {
#if defined(USE_MKL) && defined(_OPENMP)
  static void BernoulliGenerate(common::random::RandGenerator<cpu, DType> gen,
                                int n, double p, int* r) {
    typename RandGenerator<xpu, DType>::Impl genImpl(&gen, 1);
    const int seed = 17 + genImpl.rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel num_threads(nthr)
    {
      const int ithr = omp_get_thread_num();
      const int avg_amount = (n + nthr - 1) / nthr;
      const int my_offset = ithr * avg_amount;
      const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
      if (my_amount > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed + my_offset);
        vslSkipAheadStream(stream, my_offset);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
        vslDeleteStream(&stream);
      }
    }
  }

  // MKL forward pass
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<cpu> *s, RandGenerator<cpu, DType> *pgen,
                                         const double pkeep,
                                         const std::vector<TBlob> &in_data,
                                         const std::vector<TBlob> &out_data) {
    // BernoulliGenerate expects an array int, so for types smaller than int, the mask buffer
    // will be too small, so we can;t use MKL in those cases
    if (sizeof(DType) >= sizeof(int)) {
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
      DType *outptr = out.dptr_;
      DType *dataptr = data.dptr_;
      auto maskptr = reinterpret_cast<int *>(mask.dptr_);
      int count = mask.shape_[0] * mask.shape_[1];
      BernoulliGenerate(*pgen, count, pkeep, maskptr);
      const float pk_1 = 1.0f / pkeep;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = 0; i < count; ++i) {
        outptr[i] = dataptr[i] * maskptr[i] * pk_1;
      }
      return true;
    }
    return false;
  }

  // MKL backward pass
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<cpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    if (sizeof(DType) >= sizeof(int)) {
      Tensor<xpu, 2, DType> grad = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
      DType *ingradptr = gdata.dptr_;
      const DType *outgradptr = grad.dptr_;
      auto maskptr = reinterpret_cast<int *>(mask.dptr_);
      int count = mask.shape_[0] * mask.shape_[1];
      const float pk_1 = 1.0f / pkeep;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = 0; i < count; ++i) {
        ingradptr[i] = outgradptr[i] * maskptr[i] * pk_1;
      }
      return true;
    }
    return false;
  }

#ifdef __CUDACC__
  // GPU never uses MKL
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<gpu> *s, RandGenerator<gpu, DType> *pgen,
                                         const double pkeep,
                                         const std::vector<TBlob> &in_data,
                                         const std::vector<TBlob> &out_data) {
    return false;
  }
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<gpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    return false;
  }
#endif  // __CUDACC__

#else  // #if defined(USE_MKL) && defined(_OPENMP)
  static bool MSHADOW_CINLINE MKLForward(mshadow::Stream<xpu> *s, RandGenerator<xpu, DType> *pgen,
                                const double pkeep,
                                const std::vector<TBlob> &in_data,
                                const std::vector<TBlob> &out_data) {
    return false;
  }
  static bool MSHADOW_CINLINE MKLBackward(mshadow::Stream<xpu> *s, const double pkeep,
                                          const std::vector<TBlob> &in_grad,
                                          const std::vector<TBlob> &out_data,
                                          const std::vector<TBlob> &out_grad) {
    return false;
  }
#endif  // #if defined(USE_MKL) && defined(_OPENMP)

 public:
  /*!
   * \brief Dropout kernel, compute dropout tensor
   */
  struct DropoutKernel {
    /*!
     * \brief Dropout kernel function
     * \param id Thread number (0-based representing count)
     * \param gen Random number generator
     * \param N Total number of items in the output
     * \param step Step between items, related to parallelism
     * \param dropout_out Output dropout values
     * \param mask_out  Output mask (is multiplied to create dropout output, may be 0)
     * \param input_data Input data to perform the dropout on
     * \param pkeep Dropout rate (keep when the generated random number is less than this value)
     */
    MSHADOW_XINLINE static void Map(int id,
                                    RandGenerator<xpu, DType> gen,
                                    const int N,
                                    const int step,
                                    DType *dropout_out,
                                    DType *mask_out,
                                    const DType *input_data,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        mask_out[i] = mshadow_op::threshold::Map<real_t>(rand_num, pkeep) * (1.0f / pkeep);
        dropout_out[i] = input_data[i] * mask_out[i];
      });
    }
  };

  explicit DropoutOp(DropoutParam param) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = static_cast<dropout::DropoutOpMode>(param.mode);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    if (req[dropout::kOut] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob &out = out_data[dropout::kOut];
      if (ctx.is_train || this->mode_ == dropout::kAlways) {
        RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
        CHECK_NOTNULL(pgen);
        if (!MKLForward(s, pgen, this->pkeep_, in_data, out_data)) {
          const TBlob &mask = out_data[dropout::kMask];
          CHECK(req[dropout::kOut] != kAddTo);
          LaunchRNG<DropoutKernel, xpu>(s, pgen, out.Size(),
                                        out.dptr<DType>(),
                                        mask.dptr<DType>(),
                                        in_data[dropout::kData].dptr<DType>(),
                                        this->pkeep_);
        }
      } else {
        const TBlob& data = in_data[dropout::kData];
        if (req[dropout::kOut] == kWriteTo) {
          mxnet_op::copy(s, out, data);
        } else {
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
              s, out.Size(), out.dptr<DType>(), data.dptr<DType>());
          });
        }
      }
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
    if (ctx.is_train || mode_ == dropout::kAlways) {
      if (!MKLBackward(s, this->pkeep_, in_grad, out_data, out_grad)) {
        const TBlob &gdata = in_grad[dropout::kData];
        const TBlob &grad = out_grad[dropout::kOut];
        const TBlob &mask = out_data[dropout::kMask];
        CHECK_EQ(grad.Size(), mask.Size());
        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
            s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<DType>());
        });
      }
    } else {
      const TBlob& gdata = in_grad[dropout::kData];
      const TBlob& grad = out_grad[dropout::kOut];
      if (req[dropout::kData] == kWriteTo) {
        mxnet_op::copy(s, gdata, grad);
      } else {
        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
            s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
        });
      }
    }
  }

 private:
  /*! \brief Dropout rate (keep when the generated random number is less than this value) */
  real_t pkeep_;
  /*! \brief Dropout mode */
  dropout::DropoutOpMode mode_;
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

  std::vector<ResourceRequest> ForwardResource(const std::vector<TShape> &in_shape) const override {
    return { ResourceRequest::kParallelRandom };
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
#endif  // MXNET_OPERATOR_NN_DROPOUT_INL_H_

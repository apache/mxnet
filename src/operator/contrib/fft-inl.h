/*!
 * Copyright (c) 2015 by Contributors
 * \file fft-inl.h
 * \brief
 * \author Chen Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_FFT_INL_H_
#define MXNET_OPERATOR_CONTRIB_FFT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

#if MXNET_USE_CUDA
#include <cufft.h>
#endif

namespace mxnet {
namespace op {
namespace fft {
enum fftOpInputs {kData};
enum fftOpOutputs {kOutComplex};  // seperate the image and real parts at the moment
enum fftOpResource {kTempSpace};  // might be requiered as we need to pad the real matrices
}

struct FFTParam : public dmlc::Parameter<FFTParam> {
  int compute_size;  // the maximum size of sub-batch to be forwarded through FFT in one time
  DMLC_DECLARE_PARAMETER(FFTParam) {
    DMLC_DECLARE_FIELD(compute_size).set_default(128)
    .describe("Maximum size of sub-batch to be forwarded at one time");
  }
};

template<typename xpu, typename DType>
class FFTOp : public Operator {
 public:
  explicit FFTOp(FFTParam p) {
    this->param_ = p;
    init_cufft_ = false;
    dim_ = 0;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);

    // the last dimention should be the dimension of fft vector
    if (!init_cufft_) {
      n_ffts = in_data[fft::kData].shape_.ProdShape(0, in_data[fft::kData].ndim()-1);
      dim_ = in_data[fft::kData].shape_[in_data[fft::kData].ndim()-1];

      stride_ = param_.compute_size*dim_;

      init_cufft_ = true;

      // will handle the (possibly) incomplete group later
      num_compute = n_ffts / param_.compute_size;
    }


    Stream<xpu> *s = ctx.get_stream<xpu>();
    // const TShape& oshape = out_data[fft::kOutComplex].shape_;
    const TShape& ishape = in_data[fft::kData].shape_;
    const TShape& oshape = out_data[fft::kOutComplex].shape_;
    Tensor<xpu, 2, DType> data = in_data[fft::kData].get_with_shape<xpu, 2, DType>(
          Shape2(n_ffts, dim_), s);
    Tensor<xpu, 2, DType> out = out_data[fft::kOutComplex].get_with_shape<xpu, 2, DType>(
          Shape2(n_ffts, dim_*2), s);

    // need temp space to pad the data into complex numbers due to cufft interface
    Tensor<xpu, 1, DType> workspace =
            ctx.requested[fft::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(param_.compute_size*dim_*2), s);
    Tensor<xpu, 2, DType> complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(param_.compute_size, dim_*2), s);
    #if MSHADOW_USE_CUDNN
    // start fft
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param_.compute_size);
    for (size_t idx=0; idx < num_compute; ++idx) {
      complex_data = complex_pad_imag(data.Slice(idx*param_.compute_size,
                                                 idx*param_.compute_size+param_.compute_size));

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(complex_data.dptr_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out.dptr_ + 2*idx*stride_);
      CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
    }
    cufftDestroy(plan);

    // handle the remaining samples
    size_t remain_num = n_ffts - param_.compute_size*num_compute;
    if (remain_num > 0) {
      cufftHandle plan_remain;
      cufftPlanMany(&plan_remain, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0,
                    CUFFT_C2C, remain_num);

      complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                          Shape2(remain_num, dim_*2), s);
      complex_data = complex_pad_imag(data.Slice(
          num_compute*param_.compute_size, num_compute*param_.compute_size+remain_num));

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(complex_data.dptr_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out.dptr_ + 2*num_compute*stride_);
      CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
      cufftDestroy(plan_remain);
    }
    #endif
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    const TShape& ishape = in_grad[fft::kData].shape_;
    const TShape& oshape = out_grad[fft::kOutComplex].shape_;
    Tensor<xpu, 2, DType> gdata = in_grad[fft::kData].get_with_shape<xpu, 2, DType>(
          Shape2(n_ffts, dim_), s);
    Tensor<xpu, 2, DType> grad = out_grad[fft::kOutComplex].get_with_shape<xpu, 2, DType>(
          Shape2(n_ffts, dim_*2), s);
    // need temp space to pad the data into complex numbers due to cufft interface
    Tensor<xpu, 1, DType> workspace =
            ctx.requested[fft::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(param_.compute_size*dim_*2), s);
    Tensor<xpu, 2, DType> complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(param_.compute_size, dim_*2), s);

    // by default, we think forward is firstly conducted
    // In this solution, out_grad must comes from a fft of real signal,
    // so that it is Hermitian symmetric, giving a real output
    // but if it is not, remember that we have implemented complex_take_real, and use this
    #if MSHADOW_USE_CUDNN
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param_.compute_size);
    for (size_t idx = 0; idx < num_compute; ++idx) {
      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(grad.dptr_ + 2*idx*stride_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(complex_data.dptr_);
      CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);

      Assign(gdata.Slice(idx*param_.compute_size, (idx+1)*param_.compute_size),
             req[fft::kData], complex_toreal(complex_data));
    }
    cufftDestroy(plan);

    // handle the remaining samples
    size_t remain_num = n_ffts - param_.compute_size*num_compute;
    if (remain_num > 0) {
      cufftHandle plan_remain;
      cufftPlanMany(&plan_remain, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0,
                    CUFFT_C2C, remain_num);
      complex_data = Tensor<xpu, 2, DType>(workspace.dptr_,
                                              Shape2(remain_num, dim_*2), s);

      cufftComplex* in_tmp = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(grad.dptr_ + 2*num_compute*stride_));
      cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(complex_data.dptr_);
      CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);

      Assign(gdata.Slice(param_.compute_size*num_compute,
                         param_.compute_size*num_compute+remain_num),
             req[fft::kData], complex_toreal(complex_data));
      cufftDestroy(plan_remain);
    }
    #endif
    // for bp, we should not divide it
    // but for comparison with np.fft.ifft, we should do it.
    // gdata /= dim_;
  }

 private:
  FFTParam param_;
  int dim_, stride_, num_compute, n_ffts;
  bool init_cufft_;
};  // class FFTOp

// Declare Factory Function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FFTParam param, int dtype);

#if DMLC_USE_CXX11
class FFTProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

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
    CHECK_EQ(in_shape->size(), 1) <<"Input:[data]";
    const TShape &dshape = (*in_shape)[fft::kData];
    // require data to be known
    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    if (dshape.ndim() == 4) {
      out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]*2));
    } else if (dshape.ndim() == 2) {
      out_shape->push_back(Shape2(dshape[0], dshape[1]*2));
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    FFTProp* fft_sym = new FFTProp();
    fft_sym->param_ = this->param_;
    return fft_sym;
  }

  std::string TypeString() const override {
    return "_contrib_fft";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fft::kOutComplex], in_data[fft::kData]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fft::kData], in_grad[fft::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  FFTParam param_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_FFT_INL_H_

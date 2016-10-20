/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_convolution-inl.h
 * \brief
 * \author Chen, Xiaoming
*/
#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

#include <dmlc/timer.h>
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "mkl_service.h"
#include "../convolution-inl.h"
#include "./mkldnn_memory-inl.h"
#include "./mkldnn_cppwrapper.h"
namespace mxnet {
namespace op {
#if MXNET_USE_MKLDNN == 1
static int getMKLBuildDate() {
  static int build = 0;
  if (build == 0) {
    MKLVersion v;
    mkl_get_version(&v);
    build = atoi(v.Build);
  }
  return build;
}

template <typename DType> class MKLDNNConvolutionOp : public Operator {
 public:
  explicit MKLDNNConvolutionOp(ConvolutionParam param)
      : param_(param),
        fwd_in_data_(new MKLData<DType>()),
        fwd_out_data_(new MKLData<DType>()),
        fwd_filter_data_(new MKLData<DType>()),
        fwd_bias_data_(new MKLData<DType>()),
        convolutionFwd_(NULL),
        bwdd_in_diff_(new MKLData<DType>()),
        bwdd_out_diff_(new MKLData<DType>()),
        bwdd_filter_data_(new MKLData<DType>()),
        bwdf_in_data_(new MKLData<DType>()),
        bwdf_out_diff_(new MKLData<DType>()),
        bwdf_filter_diff_(new MKLData<DType>()),
        bwdb_out_diff_(new MKLData<DType>()),
        bwdb_bias_diff_(new MKLData<DType>()),
        convolutionBwdData_(NULL),
        convolutionBwdFilter_(NULL),
        convolutionBwdBias_(NULL) {
    init_mkl_ = false;
    algo_ = dnnAlgorithmConvolutionDirect;
  }

  ~MKLDNNConvolutionOp() {
    dnnDelete<DType>(convolutionBwdBias_);
    dnnDelete<DType>(convolutionBwdFilter_);
    dnnDelete<DType>(convolutionBwdData_);
    dnnDelete<DType>(convolutionFwd_);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    CHECK_EQ(out_data.size(), 1);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mkldnn";
    }

    Tensor<cpu, 4, DType> data = in_data[conv::kData].get<cpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group, param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] *
                   param_.kernel[1]);
    Tensor<cpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<cpu, 3, DType>(wmat_shape, s);
    Tensor<cpu, 1, DType> bias;
    if (!param_.no_bias) {
      bias = in_data[conv::kBias].get<cpu, 1, DType>(s);
    }
    Tensor<cpu, 4, DType> out = out_data[conv::kOut].get<cpu, 4, DType>(s);

    void *res_convolutionFwd[dnnResourceNumber];
    if (!init_mkl_) {
      Init(s, in_data, out_data);
    }

    res_convolutionFwd[dnnResourceSrc] =
        fwd_in_data_->get_converted_prv(data.dptr_, false);

    res_convolutionFwd[dnnResourceFilter] =
        fwd_filter_data_->get_converted_prv(wmat.dptr_, false);
    if (!param_.no_bias) {
      res_convolutionFwd[dnnResourceBias] =
          fwd_bias_data_->get_converted_prv(bias.dptr_, false);
    } else {
      res_convolutionFwd[dnnResourceBias] = NULL;
    }
    res_convolutionFwd[dnnResourceDst] =
        reinterpret_cast<void *>(fwd_out_data_->set_output_ptr(out.dptr_));
    dnnError_t status = dnnExecute<DType>(convolutionFwd_, res_convolutionFwd);
    CHECK_EQ(status, E_SUCCESS) << "execute forward fail with status" << status;
    fwd_out_data_->get_output_ptr(out.dptr_);
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
    dnnError_t status = E_SUCCESS;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    if (!init_mkl_) {
      Init(s, in_data, out_data);
    }

    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mkldnn";
    }
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);

    DType *grad_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *gwmat_ptr = NULL;
    DType *data_ptr = NULL;
    DType *gdata_ptr = NULL;

    Tensor<cpu, 4, DType> data = in_data[conv::kData].get<cpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group, param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] *
                   param_.kernel[1]);
    Tensor<cpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<cpu, 3, DType>(wmat_shape, s);
    Tensor<cpu, 4, DType> grad = out_grad[conv::kOut].get<cpu, 4, DType>(s);
    Tensor<cpu, 4, DType> gdata = in_grad[conv::kData].get<cpu, 4, DType>(s);
    Tensor<cpu, 3, DType> gwmat =
        in_grad[conv::kWeight].get_with_shape<cpu, 3, DType>(wmat_shape, s);

    grad_ptr = grad.dptr_;
    wmat_ptr = wmat.dptr_;
    gwmat_ptr = gwmat.dptr_;
    data_ptr = data.dptr_;
    gdata_ptr = gdata.dptr_;

    // avoid input layer's gradient calculation for MKLDNN
    if (!((gdata.shape_[1] == 3) || (gdata.shape_[1] == 1))) {
      void *res_convolutionBwdData[dnnResourceNumber];
      res_convolutionBwdData[dnnResourceDiffDst] =
          bwdd_out_diff_->get_converted_prv(grad_ptr, false);
      res_convolutionBwdData[dnnResourceFilter] =
          bwdd_filter_data_->get_converted_prv(wmat_ptr, false);
      res_convolutionBwdData[dnnResourceDiffSrc] =
          reinterpret_cast<void *>(bwdd_in_diff_->set_output_ptr(gdata_ptr));
      status = dnnExecute<DType>(convolutionBwdData_, res_convolutionBwdData);
      CHECK_EQ(status, E_SUCCESS) << "Backward Data conv failed with status "
                                  << status;
      bwdd_in_diff_->get_output_ptr(gdata_ptr);
    }

    void *res_convolutionBwdFilter[dnnResourceNumber];
    if (dnnLayoutCompare<DType>(bwdf_out_diff_->layout_int,
                                bwdd_out_diff_->layout_int)) {
      res_convolutionBwdFilter[dnnResourceDiffDst] =
          bwdd_out_diff_->get_converted_prv(grad_ptr, true);
    } else {
      res_convolutionBwdFilter[dnnResourceDiffDst] =
          bwdf_out_diff_->get_converted_prv(grad_ptr, false);
    }
    if (dnnLayoutCompare<DType>(fwd_in_data_->layout_int,
                                bwdf_in_data_->layout_int)) {
      res_convolutionBwdFilter[dnnResourceSrc] =
          fwd_in_data_->get_converted_prv(data_ptr, true);
    } else {
      res_convolutionBwdFilter[dnnResourceSrc] =
          bwdf_in_data_->get_converted_prv(data_ptr, false);
    }
    res_convolutionBwdFilter[dnnResourceDiffFilter] =
        reinterpret_cast<void *>(bwdf_filter_diff_->set_output_ptr(gwmat_ptr));
    status = dnnExecute<DType>(convolutionBwdFilter_, res_convolutionBwdFilter);
    CHECK_EQ(status, E_SUCCESS) << "Backward Filter conv failed with status "
                                << status;
    bwdf_filter_diff_->get_output_ptr(gwmat_ptr);

    if (!param_.no_bias) {
      Tensor<cpu, 1, DType> gbias = in_grad[conv::kBias].get<cpu, 1, DType>(s);
      void *res_convolutionBwdBias[dnnResourceNumber];
      if (dnnLayoutCompare<DType>(bwdb_out_diff_->layout_int,
                                  bwdd_out_diff_->layout_int)) {
        res_convolutionBwdBias[dnnResourceDiffDst] =
            bwdd_out_diff_->get_converted_prv(grad_ptr, true);
      } else if (dnnLayoutCompare<DType>(bwdb_out_diff_->layout_int,
                                         bwdf_out_diff_->layout_int)) {
        res_convolutionBwdBias[dnnResourceDiffDst] =
            bwdf_out_diff_->get_converted_prv(grad_ptr, true);
      } else {
        res_convolutionBwdBias[dnnResourceDiffDst] =
            bwdb_out_diff_->get_converted_prv(grad_ptr, false);
      }
      res_convolutionBwdBias[dnnResourceDiffBias] = reinterpret_cast<void *>(
          bwdb_bias_diff_->set_output_ptr(gbias.dptr_));
      status = dnnExecute<DType>(convolutionBwdBias_, res_convolutionBwdBias);
      CHECK_EQ(status, E_SUCCESS) << "Backward Bias conv failed with status "
                                  << status;
      bwdb_bias_diff_->get_output_ptr(gbias.dptr_);
    }
  }

 private:
  inline void Init(mshadow::Stream<cpu> *s, const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    if (!init_mkl_) {
      init_mkl_ = true;
      dnnError_t status = E_SUCCESS;

      Tensor<cpu, 4, DType> data = in_data[conv::kData].get<cpu, 4, DType>(s);
      Tensor<cpu, 4, DType> out = out_data[conv::kOut].get<cpu, 4, DType>(s);

      const size_t batch_size = data.size(0);
      const size_t input_channels = data.size(1);
      const size_t input_height = data.size(2);
      const size_t input_width = data.size(3);
      const size_t output_channels = param_.num_filter;
      const size_t output_height = out.size(2);
      const size_t output_width = out.size(3);
      const size_t group_number = param_.num_group;
      const size_t kernel_height = param_.kernel[0];
      const size_t kernel_width = param_.kernel[1];
      size_t dimension = 4;

      const size_t idata_sizes[4] = { input_width, input_height, input_channels,
                                      batch_size };
      const size_t idata_strides[4] = {
        1, input_width, input_width * input_height,
        input_width * input_height * input_channels};

      size_t g_mkl2017 = group_number;
      size_t f_dimension = dimension + (group_number != 1);
      if (getMKLBuildDate() < 20160701) {
        g_mkl2017 = 1;
        f_dimension = dimension;
      }
      const size_t fdata_sizes[5] = {kernel_width, kernel_height,
                                      input_channels / group_number,
                                      output_channels / g_mkl2017, g_mkl2017};
      const size_t fdata_strides[5] = {
        1, kernel_width, kernel_width * kernel_height,
        kernel_width * kernel_height * input_channels / group_number,
        kernel_width * kernel_height * input_channels / group_number *
            output_channels / group_number
      };

      const size_t bias_sizes[1] = {output_channels};
      const size_t bias_strides[1] = {1};

      const size_t odata_sizes[4] = {output_width, output_height,
                                      output_channels, batch_size};
      const size_t odata_strides[4] = {
        1, output_width, output_width * output_height,
        output_width * output_height * output_channels};

      size_t convolutionStrides[2] = { param_.stride[1], param_.stride[0] };
      int inputOffset[2] = {static_cast<int>(-param_.pad[1]), static_cast<int>(-param_.pad[0])};

      // forward setup
      if (!param_.no_bias) {
        status = dnnGroupsConvolutionCreateForwardBias<DType>(
            &convolutionFwd_, NULL, algo_, group_number, dimension, idata_sizes,
            odata_sizes, fdata_sizes, convolutionStrides, inputOffset,
            dnnBorderZeros);
      } else {
        status = dnnGroupsConvolutionCreateForward<DType>(
            &convolutionFwd_, NULL, algo_, group_number, dimension, idata_sizes,
            odata_sizes, fdata_sizes, convolutionStrides, inputOffset,
            dnnBorderZeros);
      }
      CHECK_EQ(status, E_SUCCESS) << "status" << status;

      fwd_in_data_->create_layouts(convolutionFwd_, dnnResourceSrc, dimension,
                                   idata_sizes, idata_strides);
      fwd_out_data_->create_layouts(convolutionFwd_, dnnResourceDst, dimension,
                                    odata_sizes, odata_strides);
      fwd_filter_data_->create_layouts(convolutionFwd_, dnnResourceFilter,
                                       f_dimension, fdata_sizes, fdata_strides);
      if (!param_.no_bias) {
        fwd_bias_data_->create_layouts(convolutionFwd_, dnnResourceBias, 1,
                                       bias_sizes, bias_strides);
      }

      // backward data
      status = dnnGroupsConvolutionCreateBackwardData<DType>(
          &convolutionBwdData_, NULL, algo_, group_number, dimension,
          idata_sizes, odata_sizes, fdata_sizes, convolutionStrides,
          inputOffset, dnnBorderZeros);
      CHECK_EQ(status, E_SUCCESS)
          << "Failed dnnConvolutionCreateBackwardData with status " << status
          << "\n";

      bwdd_in_diff_->create_layouts(convolutionBwdData_, dnnResourceDiffSrc,
                                    dimension, idata_sizes, idata_strides);
      bwdd_out_diff_->create_layouts(convolutionBwdData_, dnnResourceDiffDst,
                                     dimension, odata_sizes, odata_strides);
      bwdd_filter_data_->create_layouts(convolutionBwdData_, dnnResourceFilter,
                                        f_dimension, fdata_sizes,
                                        fdata_strides);

      // backward filter
      status = dnnGroupsConvolutionCreateBackwardFilter<DType>(
          &convolutionBwdFilter_, NULL, algo_, group_number, dimension,
          idata_sizes, odata_sizes, fdata_sizes, convolutionStrides,
          inputOffset, dnnBorderZeros);
      CHECK_EQ(status, E_SUCCESS)
          << "Failed dnnConvolutionCreateBackwardFilter with status " << status
          << "\n";

      bwdf_in_data_->create_layouts(convolutionBwdFilter_, dnnResourceSrc,
                                    dimension, idata_sizes, idata_strides);
      bwdf_out_diff_->create_layouts(convolutionBwdFilter_, dnnResourceDiffDst,
                                     dimension, odata_sizes, odata_strides);
      bwdf_filter_diff_->create_layouts(convolutionBwdFilter_,
                                        dnnResourceDiffFilter, f_dimension,
                                        fdata_sizes, fdata_strides);

      // backward bias
      if (!param_.no_bias) {
        status = dnnGroupsConvolutionCreateBackwardBias<DType>(
            &convolutionBwdBias_, NULL, algo_, group_number, dimension,
            odata_sizes);
        CHECK_EQ(status, E_SUCCESS)
            << "Failed dnnConvolutionCreateBackwardBias with status " << status
            << "\n";

        bwdb_out_diff_->create_layouts(convolutionBwdBias_, dnnResourceDiffDst,
                                       dimension, odata_sizes, odata_strides);
        bwdb_bias_diff_->create_layouts(convolutionBwdBias_,
                                        dnnResourceDiffBias, 1, bias_sizes,
                                        bias_strides);
      }
    }
  }

  bool init_mkl_;
  dnnAlgorithm_t algo_;
  ConvolutionParam param_;
  std::shared_ptr<MKLData<DType>> fwd_in_data_, fwd_out_data_, fwd_filter_data_,
      fwd_bias_data_;
  dnnPrimitive_t convolutionFwd_;
  std::shared_ptr<MKLData<DType>> bwdd_in_diff_, bwdd_out_diff_,
      bwdd_filter_data_;
  std::shared_ptr<MKLData<DType>> bwdf_in_data_, bwdf_out_diff_,
      bwdf_filter_diff_;
  std::shared_ptr<MKLData<DType>> bwdb_out_diff_, bwdb_bias_diff_;
  dnnPrimitive_t convolutionBwdData_, convolutionBwdFilter_,
      convolutionBwdBias_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

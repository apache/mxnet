/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_convolution-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKLDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_CONVOLUTION_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "mkl_conv-common-inl.h"
#include "mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLDNNConvolutionOp : public Operator, public MKLDNNLayer<DType>,
  public MKLConvCommon<xpu, DType> {
 public:
  std::string getName() {
    std::string name = "MKLDNNConvolutionOp";
    return name;
  }
  explicit MKLDNNConvolutionOp(ConvolutionParam p)
    : MKLDNNLayer<DType>()
    , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
    , convFwd_pd(NULL)
    , convBwdData_pd(NULL), convBwdWeights_pd(NULL) {
    this->param_ = p;
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    b_init_conv = false;
  }

  virtual ~MKLDNNConvolutionOp() {
  }
  void init_properties(const mshadow::Tensor<xpu, 4, DType> &data,
    const mshadow::Tensor<xpu, 4, DType> &out) {
    this->stride_w_ = param_.stride[1];
    this->stride_h_ = param_.stride[0];
    this->width_ = data.shape_[3];
    this->height_ = data.shape_[2];
    this->pad_w_ = param_.pad[1];
    this->pad_h_ = param_.pad[0];
    this->kernel_w_ = param_.kernel[1];
    this->kernel_h_ = param_.kernel[0];
    this->channels_ = data.shape_[1];
    this->num_ = data.shape_[0];
    this->group_ = param_.num_group;
    this->width_out_ = out.shape_[3];
    this->height_out_ = out.shape_[2];
    this->channel_output_ = out.shape_[1];
  }
 private:
  void InitForward(const OpContext &ctx) {
      auto propagation =
        (!ctx.is_train) ? prop_kind::forward_scoring : prop_kind::forward_training;

      int32_t g = std::max(this->group_, 1);
      int32_t n = this->num_;
      int32_t iw = this->width_;
      int32_t ih = this->height_;
      int32_t ic = this->channels_;

      int32_t ow = this->width_out_;
      int32_t oh = this->height_out_;
      int32_t oc = this->channel_output_;

      int32_t kw = this->kernel_w_;
      int32_t kh = this->kernel_h_;
      memory::dims convolutionStrides{ static_cast<int>(this->stride_h_),
        static_cast<int>(this->stride_w_) };
      memory::dims padding{ this->pad_h_, this->pad_w_ };

      memory::data_type mpcsn = memory::data_type::f32;
      memory::format mfmt_any = memory::format::any;
      mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

      memory::dims bottom_tz = { n, ic, ih, iw };
      memory::dims bias_tz = { oc };
      memory::dims top_tz = { n, oc, oh, ow };
      memory::dims weights_tz =
        (g != 1) ? memory::dims{ g, oc / g, ic / g, kh, kw } : memory::dims{ oc, ic, kh, kw };

      memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt_any);
      memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt_any);
      memory::desc init_top_md({ top_tz }, mpcsn, mfmt_any);
      memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt_any);

      // ---- Initialize convolution primitive descriptor
      std::shared_ptr<convolution_forward::desc> convFwd_desc;
      if (!this->param_.no_bias) {
        convFwd_desc.reset(
          new convolution_forward::desc(propagation, algorithm::convolution_direct
          , init_bottom_md, init_weights_md, init_bias_md, init_top_md
          , convolutionStrides, padding, padding, padding_kind::zero));
      } else {
        convFwd_desc.reset(
          new convolution_forward::desc(propagation, algorithm::convolution_direct
          , init_bottom_md, init_weights_md, init_top_md
          , convolutionStrides, padding, padding, padding_kind::zero));
      }
      convFwd_pd.reset(new convolution_forward::primitive_desc(*convFwd_desc, cpu_engine));
      CHECK(convFwd_pd);
      // ---- Create priv memory primitive descriptors stored as class members -------------
      typedef typename memory::primitive_desc MemPD;
      std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(
        new MemPD(convFwd_pd->src_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(
        new MemPD(convFwd_pd->dst_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(
        new MemPD(convFwd_pd->weights_primitive_desc()));

      // ---- Create usr memory primitive descriptors -------------
      memory::format mfmt_nchw = memory::format::nchw;
      memory::format weights_mfmt = (g != 1) ? memory::format::goihw : memory::format::oihw;

      std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
        new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
      std::shared_ptr<MemPD> usr_bias_data_memory_pd(
        new MemPD({ { bias_tz }, mpcsn, memory::format::x }, cpu_engine));
      std::shared_ptr<MemPD> usr_top_data_memory_pd(
        new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
      std::shared_ptr<MemPD> usr_weights_data_memory_pd(
        new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));


      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_bottom_data.reset(
        new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
      fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
      fwd_top_data.reset(
        new MKLDNNData<DType>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
      fwd_top_data->name = "fwd_top_data      @ " + this->getName();
      fwd_weights_data.reset(
        new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
      fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
      if (!this->param_.no_bias) {
        std::shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(
          new MemPD(convFwd_pd->bias_primitive_desc()));
        fwd_bias_data.reset(
          new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));
        fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
      }
  }
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[conv::kOut], kWriteTo);
        // size_t expected = this->param_.no_bias ? 2 : 3;
        // CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4, DType> data =
            mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kData], s);
        Tensor<xpu, 4, DType> out =
            mkl_experimental_direct_get<xpu, 4, DType>(out_data[conv::kOut], s);
        Tensor<xpu, 4, DType> wmat =
            mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kWeight], s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(wmat.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        DType *data_ptr = data.dptr_;
        DType *wmat_ptr = wmat.dptr_;
        DType *out_ptr = out.dptr_;
      if (convFwd_pd == NULL) {
        if (!b_init_conv) {
          this->init_properties(data, out);
          this->b_init_conv = true;
        }

        InitForward(ctx);
          // ---  init primitive and prv_memory descriptors ---------
        fwd_bottom_data_primitive =
          fwd_bottom_data->get_converted_prv(data_ptr, false, in_data[conv::kData]);
        fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(wmat_ptr, true,
          in_data[conv::kWeight]);
        if (!this->param_.no_bias) {
          Tensor<xpu, 1, DType> bias = mkl_experimental_direct_get<xpu, 1, DType>(in_data[conv::kBias], s);
          fwd_bias_data_primitive =
            fwd_bias_data->get_converted_prv(bias.dptr_, true, in_data[conv::kBias]);
        }
        fwd_top_data_memory = fwd_top_data->create_output_memory(out_ptr, out_data[conv::kOut],
          fwd_top_data);
        if (!this->param_.no_bias) {
          convFwd.reset(new convolution_forward(*convFwd_pd
            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
            , *fwd_bias_data_primitive, *fwd_top_data_memory));
        } else {
          convFwd.reset(new convolution_forward(*convFwd_pd
            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
            , *fwd_top_data_memory));
        }
      } else {
          fwd_bottom_data->sync_converted_prv(data_ptr, false, in_data[conv::kData]);
          fwd_weights_data->sync_converted_prv(wmat_ptr, true, in_data[conv::kWeight]);
          if (!this->param_.no_bias) {
              Tensor<xpu, 1, DType> bias = mkl_experimental_direct_get<xpu, 1, DType>(in_data[conv::kBias], s);
              fwd_bias_data->sync_converted_prv(bias.dptr_, true, in_data[conv::kBias]);
          }
          fwd_top_data->sync_output_memory(out_data[conv::kOut],
            fwd_top_data);
      }
      convFwd.submit();
  }
  void InitConvolutionBwd(const OpContext &ctx,
    const std::vector<TBlob> &out_grad,
    const std::vector<TBlob> &in_data,
    const std::vector<TBlob> &in_grad) {
    int32_t g = std::max(this->group_, 1);
    int32_t n = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    int32_t ow = this->width_out_;
    int32_t oh = this->height_out_;
    int32_t oc = this->channel_output_;

    int32_t kw = this->kernel_w_;
    int32_t kh = this->kernel_h_;
    memory::dims convolutionStrides{ this->stride_h_, this->stride_w_ };
    memory::dims padding{ this->pad_h_, this->pad_w_ };

    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;

    memory::dims bottom_tz = { n, ic, ih, iw };
    memory::dims bias_tz = { oc };
    memory::dims top_tz = { n, oc, oh, ow };
    memory::dims weights_tz =
      (g != 1) ? memory::dims{ g, oc / g, ic / g, kh, kw } : memory::dims{ oc, ic, kh, kw };
    memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt_any);
    memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt_any);
    memory::desc init_top_md({ top_tz }, mpcsn, mfmt_any);
    memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt_any);

    void * top_diff_data =
      const_cast<DType*>(mkl_prv_data<DType>(out_grad[0]));
      std::shared_ptr<MKLDNNMemoryDescriptor<DType> > mem_descr
        = get_mkldnn_prv_descriptor<DType>(out_grad[0]);
    // ---- Initialize convolution primitive descriptor -------------
    std::shared_ptr<convolution_backward_data::desc> convBwdData_desc;
    std::shared_ptr<convolution_backward_weights::desc> convBwdWeights_desc;
    if (!this->param_.no_bias) {
      convBwdWeights_desc.reset(
        new convolution_backward_weights::desc(algorithm::convolution_direct
        , init_bottom_md, init_weights_md, init_bias_md, init_top_md
        , convolutionStrides, padding, padding, padding_kind::zero));
    } else {
      convBwdWeights_desc.reset(
        new convolution_backward_weights::desc(algorithm::convolution_direct
        , init_bottom_md, init_weights_md, init_top_md
        , convolutionStrides, padding, padding, padding_kind::zero));
    }
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    convBwdData_desc.reset(
      new convolution_backward_data::desc(algorithm::convolution_direct
      , init_bottom_md, init_weights_md, init_top_md
      , convolutionStrides, padding, padding, padding_kind::zero));
    convBwdData_pd.reset(
      new convolution_backward_data::primitive_desc(*convBwdData_desc,
      cpu_engine, *convFwd_pd));

    convBwdWeights_pd.reset(
      new convolution_backward_weights::primitive_desc(*convBwdWeights_desc,
      cpu_engine, *convFwd_pd));


    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD;

    std::shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(
      new MemPD(convBwdData_pd->diff_src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(
      new MemPD(convBwdData_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(
      new MemPD(convBwdData_pd->weights_primitive_desc()));

    std::shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(
      new MemPD(convBwdWeights_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(
      new MemPD(convBwdWeights_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(
      new MemPD(convBwdWeights_pd->diff_weights_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format weights_mfmt = (g != 1) ? memory::format::goihw : memory::format::oihw;

    // ???!!! can we use usr memory primitive descrittors for backward??
    std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
      new MemPD({ { bottom_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_bias_data_memory_pd(
      new MemPD({ { bias_tz }, mpcsn, memory::format::x }, cpu_engine));
    std::shared_ptr<MemPD> usr_top_data_memory_pd(
      new MemPD({ { top_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_weights_data_memory_pd(
      new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bwdd_bottom_diff.reset(
      new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
    bwdd_bottom_diff->name = "bwdd_bottom_diff   @ " + this->getName();
    bwdw_bottom_data.reset(
      new MKLDNNData<DType>(usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd));
    bwdw_bottom_data->name = "bwdw_bottom_data   @ " + this->getName();

    bwdd_top_diff.reset(
      new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd));
    bwdd_top_diff->name = "bwdd_top_diff      @ " + this->getName();
    bwdw_top_diff.reset(
      new MKLDNNData<DType>(usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd));
    bwdw_top_diff->name = "bwdw_top_diff      @ " + this->getName();
    bwdd_weights_data.reset(
      new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd));
    bwdd_weights_data->name = "bwdd_weights_data  @ " + this->getName();
    bwdw_weights_diff.reset(
      new MKLDNNData<DType>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
    bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->getName();
    if (!this->param_.no_bias) {
      std::shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(
        new MemPD(convBwdWeights_pd->diff_bias_primitive_desc()));
      bwdw_bias_diff.reset(
        new MKLDNNData<DType>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd));
      bwdw_bias_diff->name = "bwdw_bias_diff     @ " + this->getName();
    }
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    }
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kData], s);
    Shape<3> wmat_shape =
      Shape3(param_.num_group,
        param_.num_filter / param_.num_group,
        data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
        in_data[conv::kWeight], wmat_shape, s);
    Tensor<xpu, 4, DType> grad =
      mkl_experimental_direct_get<xpu, 4, DType>(out_grad[conv::kOut], s);
    Tensor<xpu, 4, DType> gdata =
      mkl_experimental_direct_get<xpu, 4, DType>(in_grad[conv::kData], s);
    Tensor<xpu, 3, DType> gwmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
        in_grad[conv::kWeight], wmat_shape, s);
    
    if (!b_init_conv) {
      this->init_properties(data, grad);
      b_init_conv = true;
    }
    if (convBwdData_pd == NULL) {
      this->InitConvolutionBwd(ctx, out_grad, in_data, in_grad);
    }


    // ---  init primitive and prv_memory descriptors ---------
    if (req[0]) {
      Storage::Handle addtoWorkspace;
      if (req[0] == kAddTo) {
          // wait mkl support addto mode
          this->AddToModeAllocAndStoreBuffer(gdata.dptr_, in_grad[conv::kData].Size(),
            &addtoWorkspace);
      }
      if (convBwdData.aprimitive != NULL) {
        bwdd_top_diff->sync_converted_prv(grad.dptr_, false, out_grad[conv::kOut]);
        bwdd_weights_data->sync_converted_prv(wmat.dptr_, false, in_data[conv::kWeight]);
        bwdd_bottom_diff->sync_output_memory(in_grad[conv::kData], bwdd_bottom_diff);
      } else {
        bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(grad.dptr_, false,
        out_grad[conv::kOut]);
      bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(wmat.dptr_, false,
        in_data[conv::kWeight]);
      bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(gdata.dptr_,
        in_grad[conv::kData], bwdd_bottom_diff);

      convBwdData.reset(new convolution_backward_data(*convBwdData_pd
        , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
        , *bwdd_bottom_diff_memory));
      }
      convBwdData.submit();
      if (req[0] == kAddTo) {
        if (bwdd_bottom_diff->conversion_needed()) {
          bwdd_bottom_diff->convert_from_prv(gdata.dptr_);
        }
        this->AddToModeAddAndReleaseBuffer(&addtoWorkspace, gdata.dptr_,
          in_grad[conv::kData].Size());
      }
    }
    if (req[1]) {
      Storage::Handle addtoWorkspace;
      if (req[1] == kAddTo) {
        // wait mkl support addto mode
        this->AddToModeAllocAndStoreBuffer(gwmat.dptr_, in_grad[conv::kWeight].Size(),
          &addtoWorkspace);
      }
      if (convBwdWeights.aprimitive == NULL) {
          bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(grad.dptr_, false,
            out_grad[conv::kOut]);
          bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(data.dptr_, false,
            in_data[conv::kData]);
          
          bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(gwmat.dptr_,
            in_grad[conv::kWeight], bwdw_weights_diff);
          if (!this->param_.no_bias) {
            Tensor<xpu, 1, DType> gbias = mkl_experimental_direct_get<xpu, 1, DType>(in_grad[conv::kBias], s);
            bwdw_bias_diff_memory = bwdw_bias_diff->create_output_memory(gbias.dptr_,
              in_grad[conv::kBias], bwdw_bias_diff);

            convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
              , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
              , *bwdw_weights_diff_memory, *bwdw_bias_diff_memory));

          } else {
            convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
              , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
              , *bwdw_weights_diff_memory));
          }
      } else {
        bwdw_top_diff->sync_converted_prv(grad.dptr_, false, out_grad[conv::kOut]);
        bwdw_bottom_data->sync_converted_prv(data.dptr_, false, in_data[conv::kData]);
        bwdw_weights_diff->sync_output_memory(in_grad[conv::kWeight], bwdw_weights_diff);
        if (!this->param_.no_bias) 
          bwdw_bias_diff->sync_output_memory(in_grad[conv::kBias], bwdw_bias_diff);
      }
      convBwdWeights.submit();
      if (req[1] == kAddTo) {
        if (bwdw_weights_diff->conversion_needed()) {
          bwdw_weights_diff->convert_from_prv(gwmat.dptr_);
        }
        this->AddToModeAddAndReleaseBuffer(&addtoWorkspace, gwmat.dptr_,
          in_grad[conv::kWeight].Size());
      }
    }
}

 private:
  std::shared_ptr<MKLDNNData<DType> > fwd_bottom_data, fwd_top_data,
    fwd_weights_data, fwd_bias_data,
    bwdd_weights_data, bwdw_bottom_data;
  std::shared_ptr<MKLDNNData<DType> > bwdd_bottom_diff, bwdd_top_diff,
    bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;
  std::shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
  MKLDNNPrimitive<DType> convFwd;
  std::shared_ptr<convolution_backward_data::primitive_desc> convBwdData_pd;
  std::shared_ptr<convolution_backward_weights::primitive_desc> convBwdWeights_pd;
  MKLDNNPrimitive<DType> convBwdData, convBwdWeights;
  ConvolutionParam param_;
  bool b_init_conv;
  memory::dims input_tz;
  memory::dims bias_tz;
  memory::dims output_tz;
  memory::dims weights_tz;
  std::shared_ptr<memory> fwd_bottom_data_primitive,
        fwd_weights_data_primitive, fwd_bias_data_primitive;
      std::shared_ptr<memory> fwd_top_data_memory;
      std::shared_ptr<memory> bwdd_top_diff_primitive, bwdd_weights_data_primitive,
        bwdd_diff_src_primitive;
      std::shared_ptr<memory> bwdd_bottom_diff_memory;
      std::shared_ptr<memory> bwdw_bottom_data_primitive, bwdw_top_diff_primitive;
      std::shared_ptr<memory> bwdw_weights_diff_memory, bwdw_bias_diff_memory;
};  // class MKLDNNConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKLDNN_CONVOLUTION_INL_H_

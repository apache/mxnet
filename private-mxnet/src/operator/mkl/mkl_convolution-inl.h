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
* \file mkl_convolution-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#if MXNET_USE_MKLDNN == 1
#include "mkl_conv-common-inl.h"
#include "mkldnn_base-inl.h"
#endif
#if MXNET_USE_MKL2017 == 1
#include <mxnet/storage.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <utility>
#include "../operator_common.h"
#include "../convolution-inl.h"
#include "./mkl_util-inl.h"
#endif

namespace mxnet {
namespace op {


#if MXNET_USE_MKLDNN == 1
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
      // if (std::is_same<DType, double>::value)   NOT_IMPLEMENTED;
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
      
      if (convFwd_pd == NULL) {
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
        Tensor<xpu, 1, DType> bias;
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(wmat.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        DType *data_ptr = data.dptr_;
        DType *wmat_ptr = wmat.dptr_;
        DType *out_ptr = out.dptr_;
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
            bias = mkl_experimental_direct_get<xpu, 1, DType>(in_data[conv::kBias], s);
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
          fwd_bottom_data->sync_converted_prv(false, in_data[conv::kData]);
          fwd_weights_data->sync_converted_prv(true, in_data[conv::kWeight]);
          if (!this->param_.no_bias) 
              fwd_bias_data->sync_converted_prv(true, in_data[conv::kBias]);
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
    Tensor<xpu, 1, DType> gbias;
    if (!this->param_.no_bias) {
        gbias = mkl_experimental_direct_get<xpu, 1, DType>(in_grad[conv::kBias], s);
    }
    if (!b_init_conv) {
      this->init_properties(data, grad);
      b_init_conv = true;
    }
    if (convBwdData_pd == NULL) {
      this->InitConvolutionBwd(ctx, out_grad, in_data, in_grad);
    }

    std::shared_ptr<memory> bwdd_top_diff_primitive, bwdd_weights_data_primitive,
      bwdd_diff_src_primitive;
    std::shared_ptr<memory> bwdd_bottom_diff_memory;

    // ---  init primitive and prv_memory descriptors ---------
    if (req[0]) {
      bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(grad.dptr_, true,
        out_grad[conv::kOut]);
      bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(wmat.dptr_, false,
        in_data[conv::kWeight]);
      Storage::Handle addtoWorkspace;
      if (req[0] == kAddTo) {
          // wait mkl support addto mode
          this->AddToModeAllocAndStoreBuffer(gdata.dptr_, in_grad[conv::kData].Size(),
            &addtoWorkspace);
      }
      bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(gdata.dptr_,
        in_grad[conv::kData], bwdd_bottom_diff);

      convBwdData.reset(new convolution_backward_data(*convBwdData_pd
        , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
        , *bwdd_bottom_diff_memory));

      convBwdData.submit();
      if (req[0] == kAddTo) {
        if (bwdd_bottom_diff->conversion_needed()) {
          bwdd_bottom_diff->convert_from_prv(gdata.dptr_);
        }
        this->AddToModeAddAndReleaseBuffer(&addtoWorkspace, gdata.dptr_,
          in_grad[conv::kData].Size());
      }
    }
    std::shared_ptr<memory> bwdw_bottom_data_primitive, bwdw_top_diff_primitive;
    std::shared_ptr<memory> bwdw_weights_diff_memory, bwdw_bias_diff_memory;
    if (req[1]) {
      bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(grad.dptr_, true,
        out_grad[conv::kOut]);
      MKLDNNMemoryDescriptor<DType>* fwd_bottom_data_desc = fwd_bottom_data.get();
      bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(data.dptr_, false,
        in_data[conv::kData]);
      Storage::Handle addtoWorkspace;
      if (req[1] == kAddTo) {
        // wait mkl support addto mode
        this->AddToModeAllocAndStoreBuffer(gwmat.dptr_, in_grad[conv::kWeight].Size(),
          &addtoWorkspace);
      }
      bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(gwmat.dptr_,
        in_grad[conv::kWeight], bwdw_weights_diff);
      if (!this->param_.no_bias) {
        Tensor<xpu, 1, DType> gbias =
          mkl_experimental_direct_get<xpu, 1, DType>(in_grad[conv::kBias], s);
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
};  // class MKLConvolutionOp
#endif

#if MXNET_USE_MKL2017 == 1
template<typename xpu, typename DType>
class MKLConvolutionOp : public Operator {
 public:
  static std::string getName() {
    return "MKLConvolutionOp";
  }
  void SetupBuffer() {
    convolutionBwdBias = static_cast<dnnPrimitive_t>(NULL);
    convolutionBwdFilter = static_cast<dnnPrimitive_t>(NULL);
    convolutionBwdData = static_cast<dnnPrimitive_t>(NULL);
    convolutionFwd = static_cast<dnnPrimitive_t>(NULL);
    fwd_bottom_data = MKLData<DType>::create();
    fwd_top_data = MKLData<DType>::create();
    fwd_filter_data = MKLData<DType>::create();
    fwd_bias_data = MKLData<DType>::create();
    bwdd_top_diff = MKLData<DType>::create();
    bwdd_bottom_diff = MKLData<DType>::create();
    bwdd_filter_data = MKLData<DType>::create();
    bwdf_top_diff = MKLData<DType>::create();
    bwdf_filter_diff = MKLData<DType>::create();
    bwdf_bottom_data = MKLData<DType>::create();
    bwdb_top_diff = MKLData<DType>::create();
    bwdb_bias_diff = MKLData<DType>::create();
    // Names are for debugging purposes only.
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
    fwd_top_data->name = "fwd_top_data      @ " + this->getName();
    fwd_filter_data->name = "fwd_filter_data   @ " + this->getName();
    fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
    bwdd_top_diff->name = "bwdd_top_diff     @ " + this->getName();
    bwdd_bottom_diff->name = "bwdd_bottom_diff  @ " + this->getName();
    bwdd_filter_data->name = "bwdd_filter_data  @ " + this->getName();
    bwdf_top_diff->name = "bwdf_top_diff     @ " + this->getName();
    bwdf_bottom_data->name = "bwdf_bottom_data  @ " + this->getName();
    bwdf_filter_diff->name = "bwdf_filter_diff  @ " + this->getName();
    bwdb_top_diff->name = "bwdb_top_diff     @ " + this->getName();
    bwdb_bias_diff->name = "bwdb_bias_diff    @ " + this->getName();
  }

  explicit MKLConvolutionOp(ConvolutionParam p):
                            convolutionFwd(NULL),
                            convolutionBwdData(static_cast<dnnPrimitive_t>(NULL)),
                            convolutionBwdFilter(static_cast<dnnPrimitive_t>(NULL)),
                            convolutionBwdBias(static_cast<dnnPrimitive_t>(NULL)) {
    this->param_ = p;
    init_mkldnn_ = false;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    SetupBuffer();
  }
  void ReleaseBuffer() {
    if (convolutionFwd != NULL) {
     dnnDelete<DType>(convolutionFwd);
     convolutionFwd = NULL;
    }
    if (convolutionBwdData != NULL) {
     dnnDelete<DType>(convolutionBwdData);
     convolutionBwdData = NULL;
    }
    if (convolutionBwdFilter != NULL) {
     dnnDelete<DType>(convolutionBwdFilter);
     convolutionBwdFilter = NULL;
    }
    if (!param_.no_bias && convolutionBwdBias != NULL) {
     dnnDelete<DType>(convolutionBwdBias);
     convolutionBwdBias = NULL;
    }
  }
  virtual ~MKLConvolutionOp() {
    ReleaseBuffer();
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    this->width_ = data.shape_[3];
    this->height_ = data.shape_[2];
    this->channels_ = data.shape_[1];
    this->num_ = data.shape_[0];
    this->group_ = param_.num_group;
    this->width_out_ = out.shape_[3];
    this->height_out_ = out.shape_[2];
    int channel_out_ = out.shape_[1];
    this->num_output_ = channel_out_;
    kernel_w_ = param_.kernel[1];
    kernel_h_ = param_.kernel[0];
    stride_w_ = param_.stride[1];
    stride_h_ = param_.stride[0];
    pad_w_ = param_.pad[1];
    pad_h_ = param_.pad[0];
    int status;
    size_t n, g;
    size_t iw, ih, ic;
    size_t ow, oh, oc;
    size_t kw, kh;
    size_t dimension = 4;
    g = std::max(this->group_, 1);
    n = this->num_;
    iw = this->width_;
    ih = this->height_;
    ic = this->channels_;
    ow = this->width_out_;
    oh = this->height_out_;
    oc = this->num_output_;
    kw = this->kernel_w_;
    kh = this->kernel_h_;
    oc = this->num_output_;
    size_t bdata_sizes[4] = { iw, ih, ic, n };
    size_t bdata_strides[4] = { 1, iw, iw*ih, iw*ih*ic };
    /* starting with MKL 2017 Gold in case of groups filter layout
    * becomes 5D, i.e. groups become a separate dimension */
    size_t g_mkl2017 = g;
    size_t f_dimension = dimension + (g != 1);
    if (getMKLBuildDate() < 20160701) {
     g_mkl2017 = 1;
     f_dimension = dimension;
    }
    size_t fdata_sizes[5] = { kw, kh, ic / g, oc / g_mkl2017, g_mkl2017 };
    size_t fdata_strides[5] = { 1, kw, kw*kh, kw*kh*ic / g, kw*kh*ic / g*oc / g };
    size_t bias_sizes[1] = { oc };
    size_t bias_strides[1] = { 1 };
    size_t tdata_sizes[4] = { ow, oh, oc, n };
    size_t tdata_strides[4] = { 1, ow, ow*oh, ow*oh*oc };
    size_t convolutionStrides[2] = { this->stride_w_, this->stride_h_ };
    int    inputOffset[2] = { -this->pad_w_, -this->pad_h_ };
    // Names are for debugging purposes only.
    /*** convolution section ***/
    if (!param_.no_bias) {
      status = dnnGroupsConvolutionCreateForwardBias<DType>(&convolutionFwd,
                                                            NULL,
                                                            dnnAlgorithmConvolutionDirect,
                                                            g,
                                                            dimension,
                                                            bdata_sizes,
                                                            tdata_sizes,
                                                            fdata_sizes,
                                                            convolutionStrides,
                                                            inputOffset,
                                                            dnnBorderZeros);
    } else {
      status = dnnGroupsConvolutionCreateForward<DType>(&convolutionFwd,
                                                        NULL,
                                                        dnnAlgorithmConvolutionDirect,
                                                        g,
                                                        dimension,
                                                        bdata_sizes,
                                                        tdata_sizes,
                                                        fdata_sizes,
                                                        convolutionStrides,
                                                        inputOffset,
                                                        dnnBorderZeros);
    }
    CHECK_EQ(status, 0)
     << "Failed dnnCreateConvolution<DType>(dnnForward) with status "
     << status << "\n";
    fwd_bottom_data->create_layouts(convolutionFwd, dnnResourceSrc, dimension,
                                    bdata_sizes, bdata_strides);
    fwd_top_data->create_layouts(convolutionFwd, dnnResourceDst, dimension,
                                 tdata_sizes, tdata_strides);
    fwd_filter_data->create_layouts(convolutionFwd, dnnResourceFilter,
                                    f_dimension, fdata_sizes, fdata_strides);
    if (!param_.no_bias)
      fwd_bias_data->create_layouts(convolutionFwd, dnnResourceBias, 1,
                                    bias_sizes, bias_strides);
    /*
    * Backward by data layer setup
    */
    status = dnnGroupsConvolutionCreateBackwardData<DType>(&convolutionBwdData,
                                                           NULL,
                                                           dnnAlgorithmConvolutionDirect,
                                                           g,
                                                           dimension,
                                                           bdata_sizes,
                                                           tdata_sizes,
                                                           fdata_sizes,
                                                           convolutionStrides,
                                                           inputOffset,
                                                           dnnBorderZeros);
    CHECK_EQ(status, 0)
     << "Failed dnnConvolutionCreateBackwardData with status "
     << status << "\n";
    bwdd_bottom_diff->create_layouts(convolutionBwdData, dnnResourceDiffSrc,
                                     dimension, bdata_sizes, bdata_strides);
    bwdd_top_diff->create_layouts(convolutionBwdData, dnnResourceDiffDst,
                                  dimension, tdata_sizes, tdata_strides);
    bwdd_filter_data->create_layouts(convolutionBwdData, dnnResourceFilter,
                                     f_dimension, fdata_sizes, fdata_strides);
    /*
    * Backward by filter layer setup
    */
    status = dnnGroupsConvolutionCreateBackwardFilter<DType>(&convolutionBwdFilter,
                                                             NULL,
                                                             dnnAlgorithmConvolutionDirect,
                                                             g,
                                                             dimension,
                                                             bdata_sizes,
                                                             tdata_sizes,
                                                             fdata_sizes,
                                                             convolutionStrides,
                                                             inputOffset,
                                                             dnnBorderZeros);
    CHECK_EQ(status, 0)
     << "Failed dnnConvolutionCreateBackwardFilter with status "
     << status << "\n";
    bwdf_bottom_data->create_layouts(convolutionBwdFilter, dnnResourceSrc,
                                     dimension, bdata_sizes, bdata_strides);
    bwdf_top_diff->create_layouts(convolutionBwdFilter, dnnResourceDiffDst,
                                  dimension, tdata_sizes, tdata_strides);
    bwdf_filter_diff->create_layouts(convolutionBwdFilter, dnnResourceDiffFilter,
                                     f_dimension, fdata_sizes, fdata_strides);
    /*
    * Backward by bias layer setup
    */
    if (!param_.no_bias) {
      status = dnnGroupsConvolutionCreateBackwardBias<DType>(&convolutionBwdBias,
                                                             NULL,
                                                             dnnAlgorithmConvolutionDirect,
                                                             g,
                                                             dimension,
                                                             tdata_sizes);
     CHECK_EQ(status, 0)
      << "Failed dnnConvolutionCreateBackwardBias with status "
      << status << "\n";
     bwdb_top_diff->create_layouts(convolutionBwdBias, dnnResourceDiffDst,
                                   dimension, tdata_sizes, tdata_strides);
     bwdb_bias_diff->create_layouts(convolutionBwdBias, dnnResourceDiffBias, 1,
                                    bias_sizes, bias_strides);
    }
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    DType *data_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *out_ptr = NULL;
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kData], s);
    Tensor<xpu, 4, DType> out =
      mkl_experimental_direct_get<xpu, 4, DType>(out_data[conv::kOut], s);
    Tensor<xpu, 4, DType> wmat =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[conv::kWeight], s);
    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    data_ptr = data.dptr_;
    wmat_ptr = wmat.dptr_;
    out_ptr = out.dptr_;
    int status;
    void *res_convolutionFwd[dnnResourceNumber];
    res_convolutionFwd[dnnResourceSrc] =
      fwd_bottom_data->get_converted_prv(data_ptr, false, in_data[conv::kData]);
    res_convolutionFwd[dnnResourceFilter] =
      fwd_filter_data->get_converted_prv(wmat_ptr, true, in_data[conv::kWeight]);
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias =
        mkl_experimental_direct_get<xpu, 1, DType>(in_data[conv::kBias], s);
      res_convolutionFwd[dnnResourceBias] =
        fwd_bias_data->get_converted_prv(bias.dptr_, true, in_data[conv::kBias]);
    }

    res_convolutionFwd[dnnResourceDst] = fwd_top_data->get_output_ptr(out_ptr,
      fwd_top_data, out_data[conv::kOut]);
    status = dnnExecute<DType>(convolutionFwd, res_convolutionFwd);
    CHECK_EQ(status, 0) << "Forward convolution failed with status " << status;
#if MKL_EXPERIMENTAL == 0
    if (fwd_top_data->conversion_needed()) {
        fwd_top_data->convert_from_prv(out_ptr);
    }
#endif
  }
  void AddToModeAllocAndStoreBuffer(void *src, int blob_size, Storage::Handle *pws) {
    int blob_byte_size = blob_size * sizeof(DType);
    *pws = Storage::Get()->Alloc(blob_byte_size, Context::CPU());
    memcpy(pws->dptr, src, blob_byte_size);
  }
  void AddToModeAddAndReleaseBuffer(Storage::Handle *pws, void *dst_, int blob_size) {
    DType *dst = reinterpret_cast<DType*>(dst_);
    DType *src = reinterpret_cast<DType*>(pws->dptr);
#pragma omp parallel for
    for (int i = 0; i < blob_size; i++) {
      dst[i] += src[i];
    }
    if (pws->dptr)
      Storage::Get()->Free(*pws);
    pws->dptr = NULL;
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

    if (!init_mkldnn_) {
      init_mkldnn_ = true;
      LayerSetUp(data, grad);
    }
    int status;
    if (req[0]) {
      void *res_convolutionBwdData[dnnResourceNumber];
      res_convolutionBwdData[dnnResourceDiffDst] =
        bwdd_top_diff->get_converted_prv(grad.dptr_, true, out_grad[conv::kOut]);

      res_convolutionBwdData[dnnResourceFilter] =
        bwdd_filter_data->get_converted_prv(wmat.dptr_, false, in_data[conv::kWeight]);
     Storage::Handle addtoWorkspace;
     if (req[0] == kAddTo) {
       // wait mkl support addto mode
       AddToModeAllocAndStoreBuffer(gdata.dptr_, in_grad[conv::kData].Size(), &addtoWorkspace);
     }

     res_convolutionBwdData[dnnResourceDiffSrc] = bwdd_bottom_diff->get_output_ptr(gdata.dptr_,
       bwdd_bottom_diff, in_grad[conv::kData]);
     status = dnnExecute<DType>(convolutionBwdData, res_convolutionBwdData);
     CHECK_EQ(status, 0) << "Backward Data conv failed with status " << status;
#if MKL_EXPERIMENTAL == 0
     if (bwdd_bottom_diff->conversion_needed()) {
       bwdd_bottom_diff->convert_from_prv(gdata.dptr_);
     }
#endif
     if (req[0] == kAddTo) {
       if (bwdd_bottom_diff->conversion_needed()) {
         bwdd_bottom_diff->convert_from_prv(gdata.dptr_);
       }
      AddToModeAddAndReleaseBuffer(&addtoWorkspace, gdata.dptr_, in_grad[conv::kData].Size());
     }
    }
    if (req[1]) {
      void *res_convolutionBwdFilter[dnnResourceNumber];

      res_convolutionBwdFilter[dnnResourceDiffDst] =
        bwdf_top_diff->get_converted_prv(grad.dptr_, true, out_grad[conv::kOut]);

      res_convolutionBwdFilter[dnnResourceSrc] =
        bwdf_bottom_data->get_converted_prv(data.dptr_, false,
          in_data[conv::kData]);
     Storage::Handle addtoWorkspace;
     if (req[1] == kAddTo) {
       // wait mkl support addto mode
       AddToModeAllocAndStoreBuffer(gwmat.dptr_, in_grad[conv::kWeight].Size(), &addtoWorkspace);
     }

     res_convolutionBwdFilter[dnnResourceDiffFilter] = bwdf_filter_diff->get_output_ptr(
       gwmat.dptr_, bwdf_filter_diff, in_grad[conv::kWeight]);
     status = dnnExecute<DType>(convolutionBwdFilter, res_convolutionBwdFilter);
     CHECK_EQ(status, 0) << "Backward Filter conv failed with status " << status;
#if MKL_EXPERIMENTAL == 0
     if (bwdf_filter_diff->conversion_needed()) {
       bwdf_filter_diff->convert_from_prv(gwmat.dptr_);
     }
#endif
     if (req[1] == kAddTo) {
       if (bwdf_filter_diff->conversion_needed()) {
         bwdf_filter_diff->convert_from_prv(gwmat.dptr_);
       }
       AddToModeAddAndReleaseBuffer(&addtoWorkspace, gwmat.dptr_, in_grad[conv::kWeight].Size());
     }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias =
        mkl_experimental_direct_get<xpu, 1, DType>(in_grad[conv::kBias], s);
      void *res_convolutionBwdBias[dnnResourceNumber];
      res_convolutionBwdBias[dnnResourceDiffDst] =
        bwdb_top_diff->get_converted_prv(grad.dptr_, true, out_grad[conv::kOut]);

      res_convolutionBwdBias[dnnResourceDiffBias] = bwdb_bias_diff->get_output_ptr(gbias.dptr_,
        bwdb_bias_diff, in_grad[conv::kBias]);
      status = dnnExecute<DType>(convolutionBwdBias, res_convolutionBwdBias);
      CHECK_EQ(status, 0) << "Backward Bias failed with status " << status;
#if MKL_EXPERIMENTAL == 0
      if (bwdb_bias_diff->conversion_needed()) {
        bwdb_bias_diff->convert_from_prv(gbias.dptr_);
      }
#endif
    }
  }

 private:
  ConvolutionParam param_;
  size_t width_,
         height_,
         width_out_,
         height_out_,
         kernel_w_,
         kernel_h_,
         stride_w_,
         stride_h_;
  int group_,
      num_,
      num_output_;
  size_t channels_;
  int pad_w_,
      pad_h_;
  bool init_mkldnn_;
  dnnPrimitive_t convolutionFwd;
  dnnPrimitive_t convolutionBwdData;
  dnnPrimitive_t convolutionBwdFilter;
  dnnPrimitive_t convolutionBwdBias;
  /* Fwd step */
  std::shared_ptr<MKLData<DType> > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                   fwd_bias_data;
  /* Bwd data step */
  std::shared_ptr<MKLData<DType> > bwdd_top_diff, bwdd_bottom_diff;
  std::shared_ptr<MKLData<DType> > bwdd_filter_data;
  /* Bwd filter step */
  std::shared_ptr<MKLData<DType> > bwdf_top_diff, bwdf_filter_diff;
  std::shared_ptr<MKLData<DType> > bwdf_bottom_data;
  std::shared_ptr<MKLData<DType> > bwdf_filter_diff_iter, bwdf2fwd_filter_diff,
                                   bwdb_bias_diff_iter;
  /* Bwd bias step */
  std::shared_ptr<MKLData<DType> > bwdb_top_diff, bwdb_bias_diff;
};  // class ConvolutionOp
#endif
}  // namespace op
}  // namespace mxnet



#endif  // MXNET_OPERATOR_MKL_MKL_CONVOLUTION_INL_H_

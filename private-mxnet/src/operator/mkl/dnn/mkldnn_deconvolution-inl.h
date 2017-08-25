/*******************************************************************************
* Copyright 2017 Intel Corporation
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
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_DECONVOLUTION_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../mkl_conv-common-inl.h"
#include "mkldnn_base-inl.h"


namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class MKLDNNDeConvolutionOp : public Operator, public MKLDNNLayer<DType>,
  public MKLConvCommon<xpu, DType> {
 private:
  static int s_id_gen;
  int m_id;

 public:
  std::string getName() {
    std::string name = "MKLDNNDeConvolutionOp_";
    name = name + std::to_string(m_id);
    return name;
  }
  explicit MKLDNNDeConvolutionOp(DeconvolutionParam p)
    : MKLDNNLayer<DType>() {
    this->param_ = p;
    this->init_mkldnn_ = false;
  }

  virtual ~MKLDNNDeConvolutionOp() {
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
  void InitDeconvolution(const OpContext &ctx) {
    typedef typename memory::primitive_desc MemPD;
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

    // ---- Initialize memory descriptors (fromat = any) to create convolution descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;
    mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

    input_tz = { n, ic, ih, iw };
    bias_tz = { oc };
    output_tz = { n, oc, oh, ow };
    weights_tz = (g != 1) ?
      memory::dims{ g, oc / g, ic / g, kh, kw } : memory::dims{ oc, ic, kh, kw };

    // ---- Memory descriptors for initializing of convolution primitive descriptor
    memory::desc init_input_md({ input_tz }, mpcsn, mfmt_any);
    memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt_any);
    memory::desc init_output_md({ output_tz }, mpcsn, mfmt_any);
    memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt_any);
    // ---- Create usr memory primitive descriptors
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format weights_mfmt = (g != 1) ? memory::format::goihw : memory::format::oihw;
    std::shared_ptr<MemPD> usr_input_mpd(
      new MemPD({ { input_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_output_mpd(
      new MemPD({ { output_tz }, mpcsn, mfmt_nchw }, cpu_engine));
    std::shared_ptr<MemPD> usr_weights_mpd(
      new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));
    std::shared_ptr<MemPD> usr_bias_mpd(
      new MemPD({ { bias_tz }, mpcsn, memory::format::x }, cpu_engine));

    // ---- Decov Backward Data
    std::shared_ptr<convolution_forward::desc> deconvBwd_desc;
    deconvBwd_desc.reset(new convolution_forward::desc(prop_kind::forward_training
      , algorithm::convolution_direct
      , init_output_md , init_weights_md, init_input_md
      , convolutionStrides, padding, padding, padding_kind::zero));
    deconvBwd_pd.reset(new convolution_forward::primitive_desc(*deconvBwd_desc, cpu_engine));
    CHECK(deconvBwd_pd);
    std::shared_ptr<MemPD> bwdd_prv_output_mpd(new MemPD(deconvBwd_pd->dst_primitive_desc()));
    std::shared_ptr<MemPD> bwdd_prv_input_mpd(new MemPD(deconvBwd_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> bwdd_prv_weights_mpd(new MemPD(deconvBwd_pd->weights_primitive_desc()));

    // ---  init primitive and prv_memory descriptors ---------
    bwdd_top_diff.reset(new MKLDNNData<DType>(usr_output_mpd, bwdd_prv_input_mpd));
    bwdd_top_diff->name = "bwdd_top_diff   @ " + this->getName();
    bwdd_bottom_diff.reset(new MKLDNNData<DType>(usr_input_mpd, bwdd_prv_output_mpd));
    bwdd_bottom_diff->name = "bwdd_bottom_diff      @ " + this->getName();
    bwdd_filter_data.reset(new MKLDNNData<DType>(usr_weights_mpd, bwdd_prv_weights_mpd));
    bwdd_filter_data->name = "bwdd_filter_data  @ " + this->getName();
    // ---- Decov Forward Data
    std::shared_ptr<convolution_backward_data::desc> deconvFwd_desc;
    deconvFwd_desc.reset(new convolution_backward_data::desc(algorithm::convolution_direct
      , init_output_md,  init_weights_md, init_input_md
      , convolutionStrides, padding, padding, padding_kind::zero));
    deconvFwd_pd.reset(new convolution_backward_data::primitive_desc(
      *deconvFwd_desc, cpu_engine, *deconvBwd_pd));
    CHECK(deconvFwd_pd);
    std::shared_ptr<MemPD> fwd_prv_output_mpd(new MemPD(deconvFwd_pd->diff_src_primitive_desc()));
    std::shared_ptr<MemPD> fwd_prv_input_mpd(new MemPD(deconvFwd_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> fwd_prv_weights_mpd(new MemPD(deconvFwd_pd->weights_primitive_desc()));

    // ---  init primitive and prv_memory descriptors ---------
    fwd_bottom_data.reset(new MKLDNNData<DType>(usr_input_mpd, fwd_prv_input_mpd));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
    fwd_top_data.reset(new MKLDNNData<DType>(usr_output_mpd, fwd_prv_output_mpd));
    fwd_top_data->name = "fwd_top_data      @ " + this->getName();
    fwd_filter_data.reset(new MKLDNNData<DType>(usr_weights_mpd, fwd_prv_weights_mpd));
    fwd_filter_data->name = "fwd_filter_data  @ " + this->getName();

    // ---- Decov Backward Weight
    std::shared_ptr<convolution_backward_weights::desc> deconvBwdWeight_desc;
    if (!this->param_.no_bias) {
      deconvBwdWeight_desc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct
        , init_output_md, init_weights_md, init_bias_md, init_input_md
        , convolutionStrides, padding, padding, padding_kind::zero));
    } else {
      deconvBwdWeight_desc.reset(new convolution_backward_weights::desc(
        algorithm::convolution_direct
        , init_output_md, init_weights_md, init_input_md
        , convolutionStrides, padding, padding, padding_kind::zero));
    }
    deconvBwdWeight_pd.reset(new convolution_backward_weights::primitive_desc(
      *deconvBwdWeight_desc, cpu_engine, *deconvBwd_pd));
    CHECK(deconvBwdWeight_pd);

    std::shared_ptr<MemPD> bwdf_prv_diff_dst_mpd(
      new MemPD(deconvBwdWeight_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> bwdf_prv_src_mpd(
      new MemPD(deconvBwdWeight_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> bwdf_prv_diff_weights_md(
      new MemPD(deconvBwdWeight_pd->diff_weights_primitive_desc()));
    bwdf_top_diff.reset(new MKLDNNData<DType>(usr_input_mpd, bwdf_prv_diff_dst_mpd));
    bwdf_top_diff->name = "bwdf_top_diff      @ " + this->getName();

    bwdf_bottom_data.reset(new MKLDNNData<DType>(usr_output_mpd, bwdf_prv_src_mpd));
    bwdf_bottom_data->name = "bwdf_bottom_data   @ " + this->getName();

    bwdf_filter_diff.reset(new MKLDNNData<DType>(usr_weights_mpd,
      bwdf_prv_diff_weights_md));
    bwdf_filter_diff->name = "bwdf_filter_diff   @ " + this->getName();
    if (!this->param_.no_bias) {
      std::shared_ptr<MemPD> bwdf_prv_diff_bias_mpd(
        new MemPD(deconvBwdWeight_pd->diff_bias_primitive_desc()));
      // Backward by data layer setup
      bwdb_bias_diff.reset(new MKLDNNData<DType>(usr_bias_mpd, bwdf_prv_diff_bias_mpd));
      bwdb_bias_diff->name = "bwdb_bias_diff   @ " + this->getName();
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
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> out =
      mkl_experimental_direct_get<xpu, 4, DType>(out_data[deconv::kOut], s);
    Tensor<xpu, 4, DType> wmat =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[deconv::kWeight], s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(wmat.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (deconvFwd_pd == NULL) {
      this->init_properties(data, out);
      InitDeconvolution(ctx);
    }
    // Diff Dst => dy => data
    // Diff Src => dx => out
    std::shared_ptr<memory> fwd_data_primitive, fwd_weights_primitive, fwd_out_memory;
    fwd_data_primitive = fwd_bottom_data->get_converted_prv(data.dptr_, true,
      in_data[deconv::kData]);
    fwd_weights_primitive = fwd_filter_data->get_converted_prv(wmat.dptr_, false,
      in_data[deconv::kWeight]);
    fwd_out_memory = fwd_top_data->create_output_memory(out.dptr_,
      out_data[deconv::kOut], fwd_top_data);
    deconvFwd.reset(new convolution_backward_data(*deconvFwd_pd
      , *fwd_data_primitive, *fwd_weights_primitive
      , *fwd_out_memory));
    deconvFwd.submit();
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[deconv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 4, DType> out_cpu = out_data[deconv::kOut].get<xpu, 4, DType>(s);
      out_cpu += broadcast<1>(bias, out_cpu.shape_);
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
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data =
      mkl_experimental_direct_get<xpu, 4, DType>(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> grad =
      mkl_experimental_direct_get<xpu, 4, DType>(out_grad[deconv::kOut], s);
    Tensor<xpu, 4, DType> gdata =
      mkl_experimental_direct_get<xpu, 4, DType>(in_grad[deconv::kData], s);
    Shape<3> wmat_shape =
      Shape3(param_.num_group,
        data.shape_[1] / param_.num_group,
        param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);

    Tensor<xpu, 3, DType> wmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
        in_data[deconv::kWeight], wmat_shape, s);
    Tensor<xpu, 3, DType> gwmat =
      mkl_experimental_direct_get_with_shape<xpu, 3, DType>(
        in_grad[deconv::kWeight], wmat_shape, s);
    std::shared_ptr<memory> bwdf_src_primitive, bwdf_diff_dst_primitive;
    std::shared_ptr<memory> bwdf_diff_weights_memory, bwdd_diff_bias_memory;
    if (req[1]) {
      bwdf_diff_dst_primitive = bwdf_top_diff->get_converted_prv(grad.dptr_, true,
        out_grad[deconv::kOut]);
      bwdf_src_primitive = bwdf_bottom_data->get_converted_prv(data.dptr_, false,
        in_data[deconv::kData]);
      Storage::Handle addtoWorkspace;
      if (req[1] == kAddTo) {
        // wait mkl support addto mode
        this->AddToModeAllocAndStoreBuffer(gwmat.dptr_, in_grad[deconv::kWeight].Size(),
          &addtoWorkspace);
      }
      bwdf_diff_weights_memory = bwdf_filter_diff->create_output_memory(gwmat.dptr_,
        in_grad[deconv::kWeight], bwdf_filter_diff);

      if (!this->param_.no_bias) {
        Tensor<xpu, 1, DType> gbias =
          mkl_experimental_direct_get<xpu, 1, DType>(in_grad[deconv::kBias], s);
        bwdd_diff_bias_memory = bwdb_bias_diff->create_output_memory(gbias.dptr_,
          in_grad[deconv::kBias], bwdb_bias_diff);
        deconvBwdWeight.reset(new convolution_backward_weights(*deconvBwdWeight_pd
          , *bwdf_diff_dst_primitive, *bwdf_src_primitive, *bwdf_diff_weights_memory
          , *bwdd_diff_bias_memory));
      } else {
        deconvBwdWeight.reset(new convolution_backward_weights(*deconvBwdWeight_pd
          , *bwdf_diff_dst_primitive, *bwdf_src_primitive, *bwdf_diff_weights_memory));
      }
      deconvBwdWeight.submit();
      if (req[1] == kAddTo) {
        if (bwdf_filter_diff->conversion_needed()) {
          bwdf_filter_diff->convert_from_prv(gwmat.dptr_);
        }
        this->AddToModeAddAndReleaseBuffer(&addtoWorkspace, gwmat.dptr_,
          in_grad[deconv::kWeight].Size());
      }
    }
    if (req[deconv::kData] != kNullOp) {
      std::shared_ptr<memory> grad_primitive, weights_primitive;
      std::shared_ptr<memory> gdata_output_memory;
      grad_primitive = bwdd_top_diff->get_converted_prv(grad.dptr_, false,
        out_grad[deconv::kOut]);
      weights_primitive = bwdd_filter_data->get_converted_prv(wmat.dptr_, true,
        in_data[deconv::kWeight]);
      gdata_output_memory = bwdd_bottom_diff->create_output_memory(gdata.dptr_,
        in_grad[deconv::kData], bwdd_bottom_diff);
      deconvBwd.reset(new convolution_forward(*deconvBwd_pd
        , *grad_primitive, *weights_primitive, *gdata_output_memory));
      deconvBwd.submit();
    }
  }

 private:
  bool init_mkldnn_;
  DeconvolutionParam param_;
  std::shared_ptr<MKLDNNData<DType> > fwd_bottom_data, fwd_top_data, fwd_filter_data;
  std::shared_ptr<convolution_backward_data::primitive_desc> deconvFwd_pd;
  MKLDNNPrimitive<DType> deconvFwd;
  /* Bwd filter step */
  std::shared_ptr<MKLDNNData<DType> > bwdf_bottom_data, bwdf_top_diff, bwdf_filter_diff,
    bwdb_bias_diff;
  std::shared_ptr<convolution_backward_weights::primitive_desc> deconvBwdWeight_pd;
  MKLDNNPrimitive<DType> deconvBwdWeight;
  std::shared_ptr<convolution_forward::primitive_desc> deconvBwd_pd;
  MKLDNNPrimitive<DType> deconvBwd;
  std::shared_ptr<MKLDNNData<DType> > bwdd_top_diff, bwdd_bottom_diff,
    bwdd_filter_data;
  memory::dims input_tz;
  memory::dims bias_tz;
  memory::dims output_tz;
  memory::dims weights_tz;
};  // class MKLDNNDeConvolutionOp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_DECONVOLUTION_INL_H_

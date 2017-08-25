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
* \file mkldnn_quantized_conv-inl.h
* \brief
* \author young.jin.kim@intel.com
*         deepthi.karkada@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_CONV_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_CONV_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../mkl/mkl_conv-common-inl.h"
#include "../mkl/mkldnn_base-inl.h"
#include "../quantized_conv2d-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename WgtType, typename DstType>
class MKLDNNQuantConvOp : public Operator {/*, public MKLDNNLayer<DType>, public MKLConvCommon<cpu, DType> { */
 public:
  std::string getName() {
    std::string name = "MKLDNNQuantConvOp";
    return name;
  }
  explicit MKLDNNQuantConvOp(QuantizedConv2DParam p)
    : fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL)
    , convFwd_pd(NULL) {
    this->param_ = p;
    // param_.workspace = (param_.workspace << 20) / sizeof(DType);
    width_ = 0;
    height_ = 0;
    width_out_ = 0;
    height_out_ = 0;
    kernel_w_ = 0;
    kernel_h_ = 0;
    stride_w_ = 0;
    stride_h_ = 0;
    group_ = 1;
    num_ = 0;
    channel_output_ = 0;
    channels_ = 0;
    pad_w_ = 0;
    pad_h_ = 0;

    b_init_conv = false;
  }

  virtual ~MKLDNNQuantConvOp() {
  }
  void init_properties(const mshadow::Tensor<cpu, 4, SrcType> &data,
    const mshadow::Tensor<cpu, 4, DstType> &out) {

    if (param_.layout == mshadow::kNCHW) {
      this->width_ = data.shape_[3];
      this->height_ = data.shape_[2];
      this->channels_ = data.shape_[1];
      this->width_out_ = out.shape_[3];
      this->height_out_ = out.shape_[2];
      this->channel_output_ = out.shape_[1];
    } else if (param_.layout == mshadow::kNHWC) {
      this->width_ = data.shape_[2];
      this->height_ = data.shape_[1];
      this->channels_ = data.shape_[3];
      this->width_out_ = out.shape_[2];
      this->height_out_ = out.shape_[1];
      this->channel_output_ = out.shape_[3];
    }
    this->stride_w_ = param_.stride[1];
    this->stride_h_ = param_.stride[0];
    this->pad_w_ = param_.pad[1];
    this->pad_h_ = param_.pad[0];
    this->kernel_w_ = param_.kernel[1];
    this->kernel_h_ = param_.kernel[0];
    this->num_ = data.shape_[0];
    // this->group_ = param_.num_group;
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

      // memory::data_type mpcsn = memory::data_type::f32;
      memory::format mfmt_any = memory::format::any;
      mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

      memory::dims bottom_tz = { n, ic, ih, iw };
      memory::dims bias_tz = { oc };
      memory::dims top_tz = { n, oc, oh, ow };
      memory::dims weights_tz =
        (g != 1) ? memory::dims{ g, oc / g, ic / g, kh, kw } : memory::dims{ oc, ic, kh, kw };
// std::cout << "6666666" << std::endl;
      memory::desc init_bottom_md(bottom_tz, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, mfmt_any);
      memory::desc init_bias_md(bias_tz, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, mfmt_any);
      memory::desc init_top_md(top_tz, (mkldnn::memory::data_type)data_type_enum<DstType>::type, mfmt_any);
      memory::desc init_weights_md(weights_tz, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, mfmt_any);

      // ---- Initialize convolution primitive descriptor
      std::shared_ptr<convolution_forward::desc> convFwd_desc;
      // if (!this->param_.no_bias) {
      //   convFwd_desc.reset(
      //     new convolution_forward::desc(propagation, algorithm::convolution_direct
      //     , init_bottom_md, init_weights_md, init_bias_md, init_top_md
      //     , convolutionStrides, padding, padding, padding_kind::zero));
      // } else {
// std::cout << "55555555" << std::endl;
      // No bias, so far
        convFwd_desc.reset(
          new convolution_forward::desc(propagation, algorithm::convolution_direct
          , init_bottom_md, init_weights_md, init_top_md
          , convolutionStrides, padding, padding, padding_kind::zero));
        convFwd_desc->set_output_shift(param_.out_shift);
      // }
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
// std::cout << "4444444" << std::endl;
      // ---- Create usr memory primitive descriptors -------------
      memory::format data_format;
      memory::format weights_mfmt;
      if (param_.layout == mshadow::kNHWC) {
        data_format = memory::format::nhwc;
        weights_mfmt = (g != 1) ? memory::format::goihw : memory::format::oihw;
      }
      else {
        data_format = memory::format::nchw;
        weights_mfmt = (g != 1) ? memory::format::goihw : memory::format::oihw;
      }
// std::cout << "3333333" << std::endl;
      std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
        new MemPD({ bottom_tz, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, data_format }, cpu_engine));
      std::shared_ptr<MemPD> usr_bias_data_memory_pd(
        new MemPD({ bias_tz, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, memory::format::x }, cpu_engine));
      std::shared_ptr<MemPD> usr_top_data_memory_pd(
        new MemPD({ top_tz, (mkldnn::memory::data_type)data_type_enum<DstType>::type, data_format }, cpu_engine));
      std::shared_ptr<MemPD> usr_weights_data_memory_pd(
        new MemPD({ weights_tz, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, weights_mfmt }, cpu_engine));

// std::cout << "22222222" << std::endl;
      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_bottom_data.reset(
        new MKLDNNData<SrcType>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
      fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
      fwd_top_data.reset(
        new MKLDNNData<DstType>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
      fwd_top_data->name = "fwd_top_data      @ " + this->getName();
      fwd_weights_data.reset(
        new MKLDNNData<WgtType>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
      fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
      if (!this->param_.no_bias) {
        // std::shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(
        //   new MemPD(convFwd_pd->bias_primitive_desc()));
        // fwd_bias_data.reset(
        //   new MKLDNNData<WgtType>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));
        // fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
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
      // CHECK_EQ(out_data.size(), 1);
      Stream<cpu> *s = ctx.get_stream<cpu>();
      Tensor<cpu, 4, SrcType> data =
          mkl_experimental_direct_get<cpu, 4, SrcType>(in_data[conv::kData], s);
      Tensor<cpu, 4, DstType> out =
          mkl_experimental_direct_get<cpu, 4, DstType>(out_data[conv::kOut], s);
      Tensor<cpu, 4, WgtType> wmat =
          mkl_experimental_direct_get<cpu, 4, WgtType>(in_data[conv::kWeight], s);
      Tensor<cpu, 1, WgtType> bias;
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      SrcType *data_ptr = data.dptr_;
      WgtType *wmat_ptr = wmat.dptr_;
      DstType *out_ptr = out.dptr_;
      if (!b_init_conv) {
        this->init_properties(data, out);
        this->b_init_conv = true;
      }
      if (convFwd_pd == NULL) {
        InitForward(ctx);
      }
      std::shared_ptr<memory> fwd_bottom_data_primitive,
        fwd_weights_data_primitive, fwd_bias_data_primitive;
      std::shared_ptr<memory> fwd_top_data_memory;
      // ---  init primitive and prv_memory descriptors ---------
      fwd_bottom_data_primitive =
        fwd_bottom_data->get_converted_prv(data_ptr, false, in_data[conv::kData]);
      fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(wmat_ptr, true,
        in_data[conv::kWeight]);
      if (!this->param_.no_bias) {
        // bias = mkl_experimental_direct_get<cpu, 1, WgtType>(in_data[conv::kBias], s);
        // fwd_bias_data_primitive =
        //   fwd_bias_data->get_converted_prv(bias.dptr_, true, in_data[conv::kBias]);
      }
      fwd_top_data_memory = fwd_top_data->create_output_memory(out_ptr, out_data[conv::kOut],
        fwd_top_data);
      // std::cout << "11111" << std::endl;
      if (!this->param_.no_bias) {
        convFwd.reset(new convolution_forward(*convFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_bias_data_primitive, *fwd_top_data_memory));
      } else {
        auto conv_desc = new convolution_forward(*convFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_top_data_memory);
        // conv_desc.set
        convFwd.reset(conv_desc);
      }
      // std::cout << "00000" << std::endl;
      convFwd.submit();
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    LOG(FATAL) << "Quantized backward is not supported, yet.";
  }

 private:
  std::shared_ptr<MKLDNNData<SrcType> > fwd_bottom_data;
  std::shared_ptr<MKLDNNData<WgtType> > fwd_weights_data;
  std::shared_ptr<MKLDNNData<DstType> > fwd_top_data;
  // TODO Acc data?
  std::shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
  MKLDNNPrimitive<DstType> convFwd;
  // MKLDNNPrimitive<DType> convBwdData, convBwdWeights;
  QuantizedConv2DParam param_;
  bool b_init_conv;
  memory::dims input_tz;
  memory::dims bias_tz;
  memory::dims output_tz;
  memory::dims weights_tz;
  int width_,
    height_,
    width_out_,
    height_out_,
    kernel_w_,
    kernel_h_,
    stride_w_,
    stride_h_;
  int group_,
    num_,
    channel_output_;
  size_t channels_;
  int pad_w_,
    pad_h_;
};  // class MKLDNNQuantConvOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_CONV_INL_H_

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
* \file mkl_fully_connected-inl.h
* \brief
* \author young.jin.kim@intel.com
*          deepthi.karkada@intel.com
*         
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_FULLY_CONNECTED_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "mkldnn_base-inl.h"
#include "mkl_util-inl.h"
#include "../quantized_fully_connected-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename WgtType, typename AccType, typename DstType>
class MKLDNNQuantFullyConnectedOp : public Operator {
 public:
  explicit MKLDNNQuantFullyConnectedOp(QuantizedFullyConnectedParam p):
    init_mkldnn_(false),
    fwd_bottom_data(NULL),
    fwd_top_data(NULL),
    fwd_weights_data(NULL),
    fwd_bias_data(NULL),
    ipFwd_pd(NULL),
    w_(0), h_(0) {
    this->param_ = p;
  }

  ~MKLDNNQuantFullyConnectedOp() {}
  std::string getName() {
    return "MKLDNNQuantFullyConnectedOp";
  }

 private:
  void LayerSetUp(const mshadow::Tensor<cpu, 2, SrcType> &data,
     const mshadow::Tensor<cpu, 2, DstType> &out) {
     this->w_ = 1;
     this->h_ = 1;
     this->M_ = data.shape_[0];
     this->channels_ = data.shape_[1];
     this->N_ = out.shape_[1];
  }
    void InitInnerProductFwd(const std::vector<TBlob> &in_data) {
      int32_t n = this->M_;
      int32_t w = this->w_;
      int32_t h = this->h_;
      int32_t oc = this->N_;
      int32_t ic = this->channels_;
      bool has_spatial = h > 1 || w > 1;

      // Initialize memory descriptors (fromat = any) to create inner_product descriptor
      //memory::data_type mpcsn = memory::data_type::f32;
      // memory::format mfmt = memory::format::any;

    // std::cout << "fc 11111111!!!!!" << std::endl;

      memory::dims bottom_tz =
        (has_spatial) ? memory::dims{ n, ic, h, w } : memory::dims{ n, ic };
      memory::dims top_tz = { n, oc };
      memory::dims weights_tz =
        (has_spatial) ? memory::dims{ oc, ic, h, w } : memory::dims{ oc, ic };
      memory::dims bias_tz = { oc };
      memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
      memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;

      memory::desc init_bottom_md({ bottom_tz }, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, input_mfmt);
      memory::desc init_top_md({ top_tz }, (mkldnn::memory::data_type)data_type_enum<DstType>::type, input_mfmt);
      memory::desc init_weights_md({ weights_tz }, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, weights_mfmt);
      memory::desc init_bias_md({ bias_tz }, (mkldnn::memory::data_type)data_type_enum<AccType>::type, memory::format::x);

    // std::cout << "fc 22222222222!!!!" << std::endl;

      // Initialize inner_product primitive descriptor
      std::shared_ptr<inner_product_forward::desc> ipFwd_desc;
      if (!param_.no_bias) {
        ipFwd_desc.reset(new inner_product_forward::desc(
          prop_kind::forward_inference, init_bottom_md,
          init_weights_md, init_bias_md, init_top_md));
        // ipFwd_desc->set_output_shift(param_.out_shift);
      } else {
        ipFwd_desc.reset(new inner_product_forward::desc(
          prop_kind::forward_inference, init_bottom_md,
          init_weights_md, init_top_md)); //Supports only scoring for now
        // ipFwd_desc->set_output_shift(param_.out_shift);
      }

    // std::cout << "fc 333333333333!!!!!" << std::endl;
      mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
      auto ipf = new inner_product_forward::primitive_desc(*ipFwd_desc, cpu_engine);
          // std::cout << "fc 33333333333311111!!!!!" << std::endl;
      ipFwd_pd.reset(ipf);
      CHECK(ipFwd_pd);

      // Create priv memory primitive descriptors stored as class members
      typedef typename memory::primitive_desc MemPD;

      std::shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(
        new MemPD(ipFwd_pd->src_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_top_data_memory_pd(
        new MemPD(ipFwd_pd->dst_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(
        new MemPD(ipFwd_pd->weights_primitive_desc()));
      std::shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(
        new MemPD(ipFwd_pd->bias_primitive_desc()));

      std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
        new MemPD({ { bottom_tz }, (mkldnn::memory::data_type)data_type_enum<SrcType>::type, input_mfmt }, cpu_engine));
      std::shared_ptr<MemPD> usr_bias_data_memory_pd(
        new MemPD({ { bias_tz }, (mkldnn::memory::data_type)data_type_enum<AccType>::type, memory::format::x }, cpu_engine));
      std::shared_ptr<MemPD> usr_top_data_memory_pd(
        new MemPD({ { top_tz }, (mkldnn::memory::data_type)data_type_enum<DstType>::type, input_mfmt }, cpu_engine));
      std::shared_ptr<MemPD> usr_weights_data_memory_pd(
        new MemPD({ { weights_tz }, (mkldnn::memory::data_type)data_type_enum<WgtType>::type, weights_mfmt }, cpu_engine));

    // std::cout << "fc 44444444444!!!!!" << std::endl;

      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_bottom_data.reset(new MKLDNNData<SrcType>(
        usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
      fwd_top_data.reset(new MKLDNNData<DstType>(
        usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
      fwd_weights_data.reset(new MKLDNNData<WgtType>(
        usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
      fwd_bias_data.reset(new MKLDNNData<AccType>(
        usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));

    // std::cout << "fc 555555555555!!!!!" << std::endl;

      // Names are for debugging purposes only.
      fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
      fwd_top_data->name = "fwd_top_data      @ " + this->getName();
      fwd_weights_data->name = "fwd_weights_data  @ " + this->getName();
      fwd_bias_data->name = "fwd_bias_data     @ " + this->getName();
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    // std::cout << "Fully connected!!!!!" << std::endl;

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    /*size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);*/
    int status;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 2, SrcType> data;
    Tensor<cpu, 2, DstType> out;

    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    data = mkl_experimental_direct_get_with_shape<cpu, 2, SrcType>(
      in_data[fullc::kData],
      Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    out = mkl_experimental_direct_get_with_shape<cpu, 2, DstType>(
      out_data[fullc::kOut],
      Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    Tensor<cpu, 2, WgtType> wmat =
      mkl_experimental_direct_get<cpu, 2, WgtType>(in_data[fullc::kWeight], s);


    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    if (ipFwd_pd == NULL) {
      InitInnerProductFwd(in_data);
    }

    // std::cout << "Setup completed!!!!!" << std::endl;

    std::shared_ptr<memory> fwd_top_data_memory;
    std::shared_ptr<primitive> fwd_bottom_data_primitive,
      fwd_weights_data_primitive, fwd_bias_data_primitive;
    fwd_bottom_data_primitive = fwd_bottom_data->get_converted_prv(data.dptr_,
      false, in_data[fullc::kData]);
    fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(wmat.dptr_,
      false, in_data[fullc::kWeight]);

    fwd_top_data_memory = fwd_top_data->create_output_memory(
      out.dptr_, out_data[fullc::kOut], fwd_top_data);
    if (!param_.no_bias) {
      Tensor<cpu, 1, AccType> bias =
        mkl_experimental_direct_get<cpu, 1, AccType>(in_data[fullc::kBias], s);
      fwd_bias_data_primitive = fwd_bias_data->get_converted_prv(bias.dptr_,
        false, in_data[fullc::kBias]);
      ipFwd.reset(new inner_product_forward(*ipFwd_pd
        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
        , *fwd_bias_data_primitive, *fwd_top_data_memory));
    } else {
      ipFwd.reset(new inner_product_forward(*ipFwd_pd
        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
        , *fwd_top_data_memory));
    }

    // std::cout << "before submit!!!!!" << std::endl;

    ipFwd.submit();

    // std::cout << "after submit!!!!!" << std::endl;

    if (fwd_top_data->conversion_needed()) {
      fwd_top_data->convert_from_prv(out.dptr_);
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
    LOG(FATAL) << "Quantized fully connected backprop is not supported, yet.";
  }

 private:
  bool init_mkldnn_;
  std::shared_ptr<MKLDNNData<SrcType> > fwd_bottom_data;
  std::shared_ptr<MKLDNNData<WgtType> > fwd_weights_data;
  std::shared_ptr<MKLDNNData<DstType> > fwd_top_data;
  std::shared_ptr<MKLDNNData<AccType> > fwd_bias_data;

  /*std::shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data,
    fwd_bias_data, bwdd_weights_data, bwdw_bottom_data, bwdd_bottom_diff, bwdd_top_diff,
    bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;*/
  std::shared_ptr<inner_product_forward::primitive_desc> ipFwd_pd;
  //std::shared_ptr<inner_product_backward_data::primitive_desc> ipBwdData_pd;
  //std::shared_ptr<inner_product_backward_weights::primitive_desc> ipBwdWeights_pd;
  MKLDNNPrimitive<DstType> ipFwd;

  int32_t w_, h_;
  int M_;
  int channels_;
  int N_;
  QuantizedFullyConnectedParam param_;
};  // class MKLDNNQuantFullyConnectedOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZED_FULLY_CONNECTED_INL_H_

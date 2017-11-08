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
* \file mkldnn_fully_connected-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*         
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKLDNN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_FULLY_CONNECTED_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "mkldnn_base-inl.h"
#include "mkl_util-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename Dtype>
class MKLDNNFullyConnectedOp : public Operator, public MKLDNNLayer<Dtype> {
 public:
  explicit MKLDNNFullyConnectedOp(FullyConnectedParam p):
    init_mkldnn_(false),
    fwd_bottom_data(NULL),
    fwd_top_data(NULL),
    fwd_weights_data(NULL),
    fwd_bias_data(NULL),
    bwdd_weights_data(NULL),
    bwdw_bottom_data(NULL),
    bwdd_bottom_diff(NULL),
    bwdd_top_diff(NULL),
    bwdw_top_diff(NULL),
    bwdw_weights_diff(NULL),
    bwdw_bias_diff(NULL),
    ipFwd_pd(NULL),
    ipBwdData_pd(NULL),
    ipBwdWeights_pd(NULL),
    w_(0), h_(0) {
    param_ = p;
  }

  ~MKLDNNFullyConnectedOp() {}
  std::string getName() {
    return "MKLDNNFullyConnectedOp";
  }

 private:
  void LayerSetUp(const mshadow::Tensor<xpu, 2, Dtype> &data,
     const mshadow::Tensor<xpu, 2, Dtype> &out) {
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
      memory::data_type mpcsn = memory::data_type::f32;
      memory::format mfmt = memory::format::any;

      memory::dims bottom_tz =
        (has_spatial) ? memory::dims{ n, ic, h, w } : memory::dims{ n, ic };
      memory::dims top_tz = { n, oc };
      memory::dims weights_tz =
        (has_spatial) ? memory::dims{ oc, ic, h, w } : memory::dims{ oc, ic };
      memory::dims bias_tz = { oc };

      memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt);
      memory::desc init_top_md({ top_tz }, mpcsn, mfmt);
      memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt);
      memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt);

      // Initialize inner_product primitive descriptor
      std::shared_ptr<inner_product_forward::desc> ipFwd_desc;
      if (!param_.no_bias) {
        ipFwd_desc.reset(new inner_product_forward::desc(
          prop_kind::forward_training, init_bottom_md,
          init_weights_md, init_bias_md, init_top_md));
      } else {
        ipFwd_desc.reset(new inner_product_forward::desc(
          prop_kind::forward_training, init_bottom_md,
          init_weights_md, init_top_md));
      }
      mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
      ipFwd_pd.reset(new inner_product_forward::primitive_desc(*ipFwd_desc, cpu_engine));
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
      memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
      std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
        new MemPD({ { bottom_tz }, mpcsn, input_mfmt }, cpu_engine));
      std::shared_ptr<MemPD> usr_bias_data_memory_pd(
        new MemPD({ { bias_tz }, mpcsn, memory::format::x }, cpu_engine));
      std::shared_ptr<MemPD> usr_top_data_memory_pd(
        new MemPD({ { top_tz }, mpcsn, memory::format::nc }, cpu_engine));
      memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
      std::shared_ptr<MemPD> usr_weights_data_memory_pd(
        new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));

      // ---  init primitive and prv_memory descriptors ----------------------
      fwd_bottom_data.reset(new MKLDNNData<Dtype>(
        usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd));
      fwd_top_data.reset(new MKLDNNData<Dtype>(
        usr_top_data_memory_pd, prv_fwd_top_data_memory_pd));
      fwd_weights_data.reset(new MKLDNNData<Dtype>(
        usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd));
      fwd_bias_data.reset(new MKLDNNData<Dtype>(
        usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd));

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

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    int status;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, Dtype> data;
    Tensor<xpu, 2, Dtype> out;

    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    data = mkl_experimental_direct_get_with_shape<xpu, 2, Dtype>(
      in_data[fullc::kData],
      Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    out = mkl_experimental_direct_get_with_shape<xpu, 2, Dtype>(
      out_data[fullc::kOut],
      Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    Tensor<xpu, 2, Dtype> wmat =
      mkl_experimental_direct_get<xpu, 2, Dtype>(in_data[fullc::kWeight], s);


    if (!init_mkldnn_) {
      LayerSetUp(data, out);
      init_mkldnn_ = true;
    }
    if (ipFwd_pd == NULL) {
      InitInnerProductFwd(in_data);
      fwd_bottom_data_primitive = fwd_bottom_data->get_converted_prv(data.dptr_,
        false, in_data[fullc::kData]);
      fwd_weights_data_primitive = fwd_weights_data->get_converted_prv(wmat.dptr_,
        true, in_data[fullc::kWeight]);
      fwd_top_data_memory = fwd_top_data->create_output_memory(
        out.dptr_, out_data[fullc::kOut], fwd_top_data);
      if (!param_.no_bias) {
        Tensor<xpu, 1, Dtype> bias =
          mkl_experimental_direct_get<xpu, 1, Dtype>(in_data[fullc::kBias], s);
        fwd_bias_data_primitive = fwd_bias_data->get_converted_prv(bias.dptr_,
          true, in_data[fullc::kBias]);
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_bias_data_primitive, *fwd_top_data_memory));
      } else {
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
          , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
          , *fwd_top_data_memory));
      }
    } else {
      fwd_bottom_data->sync_converted_prv(data.dptr_,
        false, in_data[fullc::kData]);
      fwd_weights_data->sync_converted_prv(wmat.dptr_,
        true, in_data[fullc::kWeight]);
      fwd_top_data->sync_output_memory(
        out_data[fullc::kOut], fwd_top_data);
      if (!param_.no_bias) {
        Tensor<xpu, 1, Dtype> bias =
          mkl_experimental_direct_get<xpu, 1, Dtype>(in_data[fullc::kBias], s);
        fwd_bias_data->sync_converted_prv(bias.dptr_,
          true, in_data[fullc::kBias]);
      }
    }
    ipFwd.submit();

  }
  void InitInnerProductBwd() {
    int32_t n = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->channels_;
    bool has_spatial = h > 1 || w > 1;

    // Initialize memory descriptors (format = any) to create inner_product descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;

    memory::dims bottom_tz =
      (has_spatial) ? memory::dims{ n, ic, h, w } : memory::dims{ n, ic };
    memory::dims top_tz = { n, oc };
    memory::dims weights_tz =
      (has_spatial) ? memory::dims{ oc, ic, h, w } : memory::dims{ oc, ic };
    memory::dims bias_tz = { oc };

    memory::desc init_bottom_md({ bottom_tz }, mpcsn, mfmt);
    memory::desc init_top_md({ top_tz }, mpcsn, mfmt);
    memory::desc init_weights_md({ weights_tz }, mpcsn, mfmt);
    memory::desc init_bias_md({ bias_tz }, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    std::shared_ptr<inner_product_backward_data::desc> ipBwdData_desc;
    std::shared_ptr<inner_product_backward_weights::desc> ipBwdWeights_desc;

    if (!param_.no_bias)
      ipBwdWeights_desc.reset(new inner_product_backward_weights::desc(
        init_bottom_md, init_weights_md
        , init_bias_md, init_top_md));
    else
      ipBwdWeights_desc.reset(new inner_product_backward_weights::desc(
        init_bottom_md, init_weights_md, init_top_md));
    
    ipBwdData_desc.reset(new inner_product_backward_data::desc(
      init_bottom_md, init_weights_md, init_top_md));
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    ipBwdData_pd.reset(new inner_product_backward_data::primitive_desc(*ipBwdData_desc,
      cpu_engine, *ipFwd_pd));
    CHECK(ipBwdData_pd);
    ipBwdWeights_pd.reset(new inner_product_backward_weights::primitive_desc(
      *ipBwdWeights_desc, cpu_engine, *ipFwd_pd));
    CHECK(ipBwdWeights_pd);
    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD;
    std::shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(
      new MemPD(ipBwdData_pd->diff_src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(
      new MemPD(ipBwdData_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(
      new MemPD(ipBwdData_pd->weights_primitive_desc()));

    std::shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(
      new MemPD(ipBwdWeights_pd->src_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(
      new MemPD(ipBwdWeights_pd->diff_dst_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(
      new MemPD(ipBwdWeights_pd->diff_weights_primitive_desc()));
    std::shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(
      new MemPD(ipBwdWeights_pd->diff_bias_primitive_desc()));

    // Create usr memory primitive descriptors stored as class members

    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
    std::shared_ptr<MemPD> usr_bottom_data_memory_pd(
      new MemPD({ { bottom_tz }, mpcsn, input_mfmt }, cpu_engine));
    std::shared_ptr<MemPD> usr_bias_data_memory_pd(
      new MemPD({ { bias_tz }, mpcsn, memory::format::x }, cpu_engine));
    std::shared_ptr<MemPD> usr_top_data_memory_pd(
      new MemPD({ { top_tz }, mpcsn, memory::format::nc }, cpu_engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    std::shared_ptr<MemPD> usr_weights_data_memory_pd(
      new MemPD({ { weights_tz }, mpcsn, weights_mfmt }, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bwdd_bottom_diff.reset(new MKLDNNData<Dtype>(
      usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd));
    bwdd_bottom_diff->name = "bwdd_bottom_diff   @ " + this->getName();
    bwdw_bottom_data.reset(new MKLDNNData<Dtype>(
      usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd));
    bwdw_bottom_data->name = "bwdw_bottom_data   @ " + this->getName();

    bwdd_top_diff.reset(new MKLDNNData<Dtype>(
      usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd));
    bwdd_top_diff->name = "bwdd_top_diff      @ " + this->getName();
    bwdw_top_diff.reset(new MKLDNNData<Dtype>(
      usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd));
    bwdw_top_diff->name = "bwdw_top_diff      @ " + this->getName();;

    bwdd_weights_data.reset(new MKLDNNData<Dtype>(
      usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd));
    bwdd_weights_data->name = "bwdd_weights_data  @ " + this->getName();
    bwdw_weights_diff.reset(new MKLDNNData<Dtype>(
      usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd));
    bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->getName();;

    if (!param_.no_bias) {
      bwdw_bias_diff.reset(new MKLDNNData<Dtype>(
        usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd));
      bwdw_bias_diff->name = "bwdw_bias_diff     @ " + this->getName();;
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
    CHECK_EQ(out_grad.size(), 1U);

    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_grad[fullc::kOut].shape_;

    Tensor<xpu, 2, Dtype> data = mkl_experimental_direct_get_with_shape<xpu, 2, Dtype>(
      in_data[fullc::kData],
      Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, Dtype> wmat = mkl_experimental_direct_get<xpu, 2, Dtype>(
      in_data[fullc::kWeight], s);
    Tensor<xpu, 2, Dtype> grad = mkl_experimental_direct_get_with_shape<xpu, 2, Dtype>(
      out_grad[fullc::kOut],
      Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2, Dtype> gwmat = mkl_experimental_direct_get<xpu, 2, Dtype>(
      in_grad[fullc::kWeight], s);
    Tensor<xpu, 2, Dtype> gdata = mkl_experimental_direct_get_with_shape<xpu, 2, Dtype>(
      in_grad[fullc::kData],
      Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 1, Dtype> gbias;
    if (!param_.no_bias)
        gbias = mkl_experimental_direct_get<xpu, 1, Dtype>(
            in_grad[fullc::kBias], s);
    if (!init_mkldnn_) {
      LayerSetUp(data, grad);
      init_mkldnn_ = true;
    }
    if (ipBwdData_pd == NULL) {
      InitInnerProductBwd();
    }
    if (req[fullc::kData]) {
      if (ipBwdData.aprimitive == NULL) {
      bwdd_top_diff_primitive = bwdd_top_diff->get_converted_prv(grad.dptr_,
        false, out_grad[fullc::kOut]);
      bwdd_weights_data_primitive = bwdd_weights_data->get_converted_prv(wmat.dptr_,
        false, in_data[fullc::kWeight]);
      bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory(gdata.dptr_,
        in_grad[fullc::kData], bwdd_bottom_diff);
      ipBwdData.reset(new inner_product_backward_data(*ipBwdData_pd
        , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
        , *bwdd_bottom_diff_memory));
      } else {
       bwdd_top_diff->sync_converted_prv(grad.dptr_,
        false, out_grad[fullc::kOut]);
       bwdd_weights_data->sync_converted_prv(wmat.dptr_,
        false, in_data[fullc::kWeight]);
       bwdd_bottom_diff->sync_output_memory(
        in_grad[fullc::kData], bwdd_bottom_diff);
      }
      ipBwdData.submit();
    }
    if (req[fullc::kWeight]) {
      if (ipBwdWeights.aprimitive == NULL) {
        bwdw_bottom_data_primitive = bwdw_bottom_data->get_converted_prv(data.dptr_,
          false, in_data[fullc::kData]);
        bwdw_top_diff_primitive = bwdw_top_diff->get_converted_prv(grad.dptr_,
          false, out_grad[fullc::kOut]);
        bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory(gwmat.dptr_,
        in_grad[fullc::kWeight], bwdw_weights_diff);
        if (!param_.no_bias) {
          bwdw_bias_diff_memory = bwdw_bias_diff->create_output_memory(gbias.dptr_,
            in_grad[fullc::kBias], bwdw_bias_diff);
          ipBwdWeights.reset(new inner_product_backward_weights(*ipBwdWeights_pd
            , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
            , *bwdw_weights_diff_memory, *bwdw_bias_diff_memory));
        } else {
          ipBwdWeights.reset(new inner_product_backward_weights(*ipBwdWeights_pd
            , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
            , *bwdw_weights_diff_memory));
        }
      } else {
         bwdw_bottom_data->sync_converted_prv(data.dptr_,
          false, in_data[fullc::kData]);
         bwdw_top_diff->sync_converted_prv(grad.dptr_,
          false, out_grad[fullc::kOut]);
         bwdw_weights_diff->sync_output_memory(
        in_grad[fullc::kWeight], bwdw_weights_diff);
        if (!param_.no_bias) 
           bwdw_bias_diff->sync_output_memory(
            in_grad[fullc::kBias], bwdw_bias_diff);
      }
      ipBwdWeights.submit();
    }

  }

 private:
  bool init_mkldnn_;
  std::shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data,
    fwd_bias_data, bwdd_weights_data, bwdw_bottom_data, bwdd_bottom_diff, bwdd_top_diff,
    bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;
  std::shared_ptr<inner_product_forward::primitive_desc> ipFwd_pd;
  std::shared_ptr<inner_product_backward_data::primitive_desc> ipBwdData_pd;
  std::shared_ptr<inner_product_backward_weights::primitive_desc> ipBwdWeights_pd;
  MKLDNNPrimitive<Dtype> ipFwd, ipBwdData, ipBwdWeights;

  std::shared_ptr<memory> fwd_top_data_memory;
  std::shared_ptr<primitive> fwd_bottom_data_primitive,
    fwd_weights_data_primitive, fwd_bias_data_primitive;
    std::shared_ptr<memory> bwdd_bottom_diff_memory
      , bwdw_weights_diff_memory, bwdw_bias_diff_memory;
    std::shared_ptr<primitive> bwdd_top_diff_primitive, bwdd_weights_data_primitive
      , bwdw_top_diff_primitive, bwdw_bottom_data_primitive;
  int32_t w_, h_;
  int M_;
  int channels_;
  int N_;
  FullyConnectedParam param_;
};  // class MKLDNNFullyConnectedOp
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MKL_MKLDNN_FULLY_CONNECTED_INL_H_

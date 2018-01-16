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
 * \file mkldnn_batch_norm.cc
 * \brief
 * \author Tao Lv (tao.a.lv@intel.com)
*/

#if MXNET_USE_MKLDNN == 1
#include "./mkldnn_batch_norm-inl.h"

namespace mxnet {
namespace op {

template <typename DType>
void MKLDNNBatchNormFwd<DType>::_Init(const mxnet::NDArray &src,
                                      bool scale_shift,
                                      bool global_stats) {
    this->_flag |= scale_shift ? use_scale_shift : 0U;
    // this->_flag |= global_stats ? use_global_stats : 0U;
    if (!(this->_is_train))
        this->_flag |= use_global_stats;

    auto src_md = src.GetMKLDNNData()->get_primitive_desc().desc();
    auto engine = CpuEngine::Get()->get_engine();

    mkldnn::prop_kind prop = forward_training;
    if (this->_is_train) {
        prop = forward_training;
    } else {
        prop = forward_inference;
    }

    auto fwd_desc = t_bn_f_desc(prop, src_md, this->_eps, this->_flag);
    auto fwd_pd   = t_bn_f_pdesc(fwd_desc, engine);

    this->data.reset(new mkldnn::memory(src.GetMKLDNNData()->get_primitive_desc()));
    this->out.reset(new mkldnn::memory(fwd_pd.dst_primitive_desc()));

    if (this->_flag & use_scale_shift) {
        this->weight.reset(new memory(fwd_pd.weights_primitive_desc()));
    }

    if (this->_is_train || (this->_flag & use_global_stats)) {
        this->mean.reset(new mkldnn::memory(fwd_pd.mean_primitive_desc()));
        this->variance.reset(new mkldnn::memory(fwd_pd.variance_primitive_desc()));
    }

    // for mxnet, there always has weight
    CHECK_EQ(this->_flag & use_scale_shift, use_scale_shift);
    if (!(this->_is_train)) {
        this->fwd.reset(
                new mkldnn::batch_normalization_forward(fwd_pd,
                                                        *(this->data),
                                                        mkldnn::primitive::at(*(this->mean)),
                                                        mkldnn::primitive::at(*(this->variance)),
                                                        mkldnn::primitive::at(*(this->weight)),
                                                        *(this->out)));
    } else {
        this->fwd.reset(
                new mkldnn::batch_normalization_forward(fwd_pd,
                                                        *(this->data),
                                                        mkldnn::primitive::at(*(this->weight)),
                                                        *(this->out),
                                                        *(this->mean),
                                                        *(this->variance)));
    }
    return;
}

template <typename DType>
void MKLDNNBatchNormFwd<DType>::SetDataHandle(const std::vector<OpReqType> &req,
                                              const mxnet::NDArray         &data,
                                              const mxnet::NDArray         &output,
                                              const mxnet::TBlob           &moving_mean,
                                              const mxnet::TBlob           &moving_var,
                                              const mxnet::TBlob           *out_mean,
                                              const mxnet::TBlob           *out_var,
                                              const mxnet::TBlob           *gamma,
                                              const mxnet::TBlob           *beta) {
    auto data_mem = data.GetMKLDNNData();
    auto out_mem = const_cast<NDArray&>(output).CreateMKLDNNData(this->out->get_primitive_desc());
    this->data->set_data_handle(data_mem->get_data_handle());
    this->out->set_data_handle(out_mem->get_data_handle());

    // weights
    if (gamma != nullptr && beta != nullptr && (this->_flag | use_scale_shift)) {
      _SetWeight(*gamma, *beta, req[batchnorm::kGamma]);
    }

    // mean and variance
    if (out_mean)
      this->_out_mean = out_mean->dptr<DType>();
    if (out_var)
      this->_out_var  = out_var->dptr<DType>();
    if (!(this->_is_train)) {
      this->mean->set_data_handle(moving_mean.dptr<DType>());
      this->variance->set_data_handle(moving_var.dptr<DType>());
    } else {
      CHECK(this->_out_mean != nullptr);
      CHECK(this->_out_var != nullptr);
      this->mean->set_data_handle(this->_out_mean);
      this->variance->set_data_handle(this->_out_var);
    }
}

template <typename DType>
void MKLDNNBatchNormFwd<DType>::Execute() {
    MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
    MKLDNNStream::Get()->Submit();
    if (this->_out_mean || this->_out_var)
        _SetMeanVar(reinterpret_cast<DType*>(this->mean->get_data_handle()),
                    reinterpret_cast<DType*>(this->variance->get_data_handle()),
                    this->_out_mean, this->_out_var);
}

template <typename DType>
void MKLDNNBatchNormFwd<DType>::_SetWeight(const mxnet::TBlob &gamma,
                                           const mxnet::TBlob &beta,
                                           const OpReqType    &req) {
    // CHECK_NE(this->weight, nullptr);
    DType *gamma_ptr  = gamma.dptr<DType>();
    DType *beta_ptr   = beta.dptr<DType>();
    DType *weight_ptr = reinterpret_cast<DType*>(this->weight->get_data_handle());
    // std::cout << "_SetWeight: channel size: " << this->_channels << std::endl;
    if (!(this->_fix_gamma)) {
#pragma omp parallel for simd
      for (int i = 0; i < this->_channels; i++) {
        weight_ptr[i] = gamma_ptr[i];
        weight_ptr[this->_channels + i] = beta_ptr[i];  // bias
      }
    } else if (IsBNWriting(req)) {
#pragma omp parallel for simd
      for (int i = 0; i < this->_channels; i++) {
        weight_ptr[i] = (DType)1.0f;
        weight_ptr[this->_channels + i] = beta_ptr[i];  // bias
        gamma_ptr[i] = (DType)1.0f;
      }
    } else {
#pragma omp parallel for simd
      for (int i = 0; i < this->_channels; i++) {
        weight_ptr[i] = (DType)1.0f;
        weight_ptr[this->_channels + i] = beta_ptr[i];  // bias
      }
    }
}

template <typename DType>
void MKLDNNBatchNormFwd<DType>::_SetMeanVar(const DType *imean,
                                            const DType *ivar,
                                            DType *omean,
                                            DType *ovar) {
    float e = this->_eps;
#pragma omp parallel for firstprivate(e)
    for (int i = 0; i < this->_channels; i++) {
      if (omean)
        omean[i] = imean[i];
      if (ovar)
        ovar[i] = VARIANCE_TO_INVSTD(ivar[i], e);
    }
}

template class MKLDNNBatchNormFwd<float>;
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN

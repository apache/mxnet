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
 * \file mkldnn_deconvolution-inl.h
 * \brief
 * \Author: Paweł Głomski, pawel.glomski@intel.com
 */
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H_

#if MXNET_USE_MKLDNN == 1
#include "../deconvolution-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

namespace mxnet {
namespace op {

using deconv_fwd_t = mkldnn::deconvolution_forward;
using deconv_fwd_pd_t = mkldnn::deconvolution_forward::primitive_desc;

using deconv_bwd_t = mkldnn::deconvolution_backward_data;
using deconv_bwd_data_pd_t = mkldnn::deconvolution_backward_data::primitive_desc;

using deconv_bwd_weight_t = mkldnn::deconvolution_backward_weights;
using deconv_bwd_weight_pd_t = mkldnn::deconvolution_backward_weights::primitive_desc;

class MKLDNNDeconvFwd {
 public:
  struct Tensors {
    Tensors(const NDArray &data, const NDArray &weight, const NDArray *bias, const NDArray &out);
    Tensors(bool no_bias, const std::vector<NDArray> &inputs, const std::vector<NDArray> &outputs);

    const NDArray &data;
    const NDArray &weight;
    const NDArray *bias;
    const NDArray &out;
  };

  static MKLDNNDeconvFwd &GetCached(const DeconvolutionParam &param, const Tensors &tensors);
  static std::shared_ptr<deconv_fwd_pd_t> MakePD(const DeconvolutionParam &param,
                                                 const Tensors &tensors);

  MKLDNNDeconvFwd(const DeconvolutionParam &param, const Tensors &tensors);
  void ControlWeightFormat(uint32_t num_group, bool is_train, const NDArray &weight);
  void Execute(uint32_t num_group, const std::vector<OpReqType> &req, const Tensors &tensors);

 private:
  const mkldnn::memory *DataMem(const NDArray &data) const;
  const mkldnn::memory *WeightMem(uint32_t num_group, const NDArray &weight) const;
  const mkldnn::memory *BiasMem(const NDArray &bias) const;

  mkldnn_output_t OutMem(OpReqType req, const NDArray &out) const;

  std::shared_ptr<deconv_fwd_t> fwd;
  std::shared_ptr<deconv_fwd_pd_t> fwd_pd;
};

class MKLDNNDeconvBwd {
 public:
  struct ReadTensors {
    ReadTensors(bool no_bias, const std::vector<NDArray> &inputs);
    const NDArray &data;
    const NDArray &weight;
    const NDArray *bias;
    const NDArray &out_grad;
  };
  struct WriteTensors {
    WriteTensors(bool no_bias, const std::vector<NDArray> &outputs);
    const NDArray &data_grad;
    const NDArray &weight_grad;
    const NDArray *bias_grad;
  };

  static MKLDNNDeconvBwd &GetCached(const DeconvolutionParam &param, const ReadTensors &rt);
  static std::shared_ptr<deconv_bwd_data_pd_t> MakeDataPD(const DeconvolutionParam &param,
                                                          const ReadTensors &rt,
                                                          const deconv_fwd_pd_t &fwd_pd);
  static std::shared_ptr<deconv_bwd_weight_pd_t> MakeWeightsPD(const DeconvolutionParam &param,
                                                               const ReadTensors &rt,
                                                               const deconv_fwd_pd_t &fwd_pd);

  MKLDNNDeconvBwd(const DeconvolutionParam &param, const ReadTensors &rt);
  void Execute(uint32_t num_group, const std::vector<OpReqType> &req, const ReadTensors &rt,
               const WriteTensors &wt);

 private:
  void IOSwapWeightTensors(uint32_t num_group, const std::vector<OpReqType> &req,
                           const NDArray &weight, const NDArray &weight_grad);

  const mkldnn::memory *ScheduleBwdData(uint32_t num_group, const std::vector<OpReqType> &req,
                                        const ReadTensors &rt, const WriteTensors &wt);

  void ScheduleBwdWeight(uint32_t num_group, const std::vector<OpReqType> &req,
                         const ReadTensors &rt, const WriteTensors &wt,
                         const mkldnn::memory *out_grad_mem);

  const mkldnn::memory *DataMem(const NDArray &data) const;
  const mkldnn::memory *WeightMem(uint32_t num_group, const NDArray &weight) const;
  const mkldnn::memory *OutGradMem(const NDArray &out_grad) const;  // for bwd data
  const mkldnn::memory *OutGradMem(const NDArray &out_grad,         // for bwd weight
                                   const mkldnn::memory *out_grad_mem) const;

  mkldnn_output_t DataGradMem(OpReqType req, const NDArray &data_grad) const;
  mkldnn_output_t WeightGradMem(uint32_t num_group, OpReqType req,
                                const NDArray &weight_grad) const;
  mkldnn_output_t BiasGradMem(OpReqType req, const NDArray *bias) const;

  std::shared_ptr<deconv_bwd_data_pd_t> bwd_data_pd;
  std::shared_ptr<deconv_bwd_weight_pd_t> bwd_weight_pd;
  std::shared_ptr<deconv_bwd_t> bwd_data;
  std::shared_ptr<deconv_bwd_weight_t> bwd_weight;
};  // namespace op

struct DeconvDescCreator {
  DeconvDescCreator(const DeconvolutionParam &param, const NDArray &data, const NDArray &weight,
                    const NDArray *bias, const NDArray &out);

  // Imposes plain formats on memory descriptors with padding
  // Changing only one at a time, so maybe better implementations will be selected
  // (than entirely plain one)
  void ImposePlainWherePadding(size_t data_size, size_t weight_size, size_t out_size);
  bool CheckImpl(size_t data_size, size_t weight_size, size_t out_size) const;

  deconv_fwd_t::desc MakeFwdDesc() const;
  deconv_bwd_t::desc MakeBwdDataDesc() const;
  deconv_bwd_weight_t::desc MakeBwdWeightDesc() const;

  mkldnn::memory::desc data_md;
  mkldnn::memory::desc weight_md;
  mkldnn::memory::desc bias_md;
  mkldnn::memory::desc out_md;

  mkldnn::memory::dims strides;
  mkldnn::memory::dims padding;
  mkldnn::memory::dims dilates;

  mkldnn::engine &engine;
};

mkldnn::memory::desc IOLogicalSwapDesc(mkldnn::memory::desc desc, int num_groups);
void IOLogicalSwapMKLDNNMem(const NDArray &arr, int num_groups);

// Version of GetWeightDesc for deconvolution (with swap)
static inline mkldnn::memory::desc GetDeconvWeightDesc(const NDArray &weight, int num_groups) {
  return IOLogicalSwapDesc(GetWeightDesc(weight, num_groups), num_groups);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__

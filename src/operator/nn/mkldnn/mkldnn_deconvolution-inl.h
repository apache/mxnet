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
 *          ________
 * Data---->|Deconv|
 * Weight-->|  FWD |--->out
 * Bias---->|______|
 *               ________
 * Data_grad<----|Deconv|<---out_grad
 * Weight_grad<--|  BWD |<---data
 * Bias_grad<----|      |<---Weight
 *               |______|<---Bias
 *
 * "out" in this (and .cc) file will always refer to the output of Deconv FWD and
 * "out_grad" to its gradient
 */
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DECONVOLUTION_INL_H_

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

using deconv_bwd_weights_t = mkldnn::deconvolution_backward_weights;
using deconv_bwd_weights_pd_t = mkldnn::deconvolution_backward_weights::primitive_desc;

class MKLDNNDeconvFwd {
 public:
  struct Tensors {
    Tensors(const NDArray &data, const NDArray &weights, const NDArray *const bias,
            const NDArray &out);
    Tensors(const bool no_bias, const std::vector<NDArray> &inputs,
            const std::vector<NDArray> &outputs);

    const NDArray &data;
    const NDArray &weights;
    const NDArray *const bias;
    const NDArray &out;
  };

  static MKLDNNDeconvFwd &GetCached(const DeconvolutionParam &param, const Tensors &tensors);
  static std::shared_ptr<deconv_fwd_pd_t> MakePD(const DeconvolutionParam &param,
                                                 const Tensors &tensors);

  MKLDNNDeconvFwd(const DeconvolutionParam &param, const Tensors &tensors);
  void ControlWeightsFormat(const uint32_t num_group, const bool is_train, const NDArray &weights);
  void Execute(const uint32_t num_group, const std::vector<OpReqType> &req, const Tensors &tensors);

 private:
  const mkldnn::memory *DataMem(const NDArray &data) const;
  const mkldnn::memory *WeightsMem(const uint32_t num_group, const NDArray &weights) const;
  const mkldnn::memory *BiasMem(const NDArray &bias) const;

  mkldnn_output_t OutMem(const OpReqType req, const NDArray &out) const;

  std::shared_ptr<deconv_fwd_t> fwd;
  std::shared_ptr<deconv_fwd_pd_t> fwd_pd;
};

class MKLDNNDeconvBwd {
 public:
  struct ReadTensors {
    ReadTensors(const bool no_bias, const std::vector<NDArray> &inputs);
    const NDArray &data;
    const NDArray &weights;
    const NDArray *const bias;
    const NDArray &out_grad;
  };
  struct WriteTensors {
    WriteTensors(const bool no_bias, const std::vector<NDArray> &outputs);
    const NDArray &data_grad;
    const NDArray &weights_grad;
    const NDArray *const bias_grad;
  };

  static MKLDNNDeconvBwd &GetCached(const DeconvolutionParam &param,
                                    const ReadTensors &read_tensors);
  static std::shared_ptr<deconv_bwd_data_pd_t> MakeDataPD(const DeconvolutionParam &param,
                                                          const ReadTensors &read_tensors,
                                                          const deconv_fwd_pd_t &fwd_pd);
  static std::shared_ptr<deconv_bwd_weights_pd_t> MakeWeightsPD(const DeconvolutionParam &param,
                                                                const ReadTensors &read_tensors,
                                                                const deconv_fwd_pd_t &fwd_pd);

  MKLDNNDeconvBwd(const DeconvolutionParam &param, const ReadTensors &read_tensors);
  void Execute(const uint32_t num_group, const std::vector<OpReqType> &req,
               const ReadTensors &read_tensors, const WriteTensors &write_tensors);

 private:
  void IOSwapWeightsTensors(const uint32_t num_group, const std::vector<OpReqType> &req,
                            const NDArray &weights, const NDArray &weights_grad);

  // returns the output gradient memory used to calculate the data (input) gradient, which
  // might be reused when calculating the gradient of weights
  const mkldnn::memory *ScheduleBwdData(const uint32_t num_group, const std::vector<OpReqType> &req,
                                        const ReadTensors &read_tensors,
                                        const WriteTensors &write_tensors);

  void ScheduleBwdWeights(const uint32_t num_group, const std::vector<OpReqType> &req,
                          const ReadTensors &read_tensors, const WriteTensors &write_tensors,
                          const mkldnn::memory *const out_grad_mem);

  const mkldnn::memory *DataMem(const NDArray &data) const;
  const mkldnn::memory *WeightsMem(const uint32_t num_group, const NDArray &weights) const;

  // for calculating the gradient of data (input)
  const mkldnn::memory *OutGradMem(const NDArray &out_grad) const;
  // for calculating the gradient of weights
  const mkldnn::memory *OutGradMem(const NDArray &out_grad,
                                   const mkldnn::memory *const out_grad_mem) const;

  mkldnn_output_t DataGradMem(const OpReqType req, const NDArray &data_grad) const;
  mkldnn_output_t WeightsGradMem(const uint32_t num_group, const OpReqType req,
                                 const NDArray &weights_grad) const;
  mkldnn_output_t BiasGradMem(const OpReqType req, const NDArray *const bias) const;

  std::shared_ptr<deconv_bwd_data_pd_t> bwd_data_pd;
  std::shared_ptr<deconv_bwd_weights_pd_t> bwd_weights_pd;
  std::shared_ptr<deconv_bwd_t> bwd_data;
  std::shared_ptr<deconv_bwd_weights_t> bwd_weights;
};

// Utility class for creating operation descriptors of deconvolution primitives
struct DeconvDescCreator {
  DeconvDescCreator(const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
                    const NDArray *const bias, const NDArray &out);

  // Imposes plain formats on memory descriptors with padding (so the next selected implementation
  // will pass CheckImplSizeReq). After calling this method, new primitive descriptor (with new
  // operator descriptor) should be created, which should select an implementation with matching
  // size requirements.
  // data_size, weights_size, out_size - size requirements of current implementation
  // Returns whether successfully imposed a plain format on any of the data, weights, and output
  // memory descriptors.
  bool ImposePlainWherePadding(const size_t data_size, const size_t weights_size,
                               const size_t out_size);
  bool CheckImplSizeReq(const size_t data_size, const size_t weights_size,
                        const size_t out_size) const;

  deconv_fwd_t::desc MakeFwdDesc() const;
  deconv_bwd_t::desc MakeBwdDataDesc() const;
  deconv_bwd_weights_t::desc MakeBwdWeightsDesc() const;

  mkldnn::memory::desc data_md;
  mkldnn::memory::desc weights_md;
  mkldnn::memory::desc bias_md;
  mkldnn::memory::desc out_md;

  mkldnn::memory::dims strides;
  mkldnn::memory::dims padding;
  mkldnn::memory::dims dilates;

  mkldnn::engine &engine;
};

// Swaps the logical order of dimensions that in plain format would correspond to input and output
// channels (for example: oihw => iohw, iohw => oihw, goihw => giohw).
mkldnn::memory::desc IOLogicalSwapDesc(const mkldnn::memory::desc &desc, const int num_groups);

// Applies IOLogicalSwapDesc to MKLDNN memory of arr
void IOLogicalSwapMKLDNNMem(const NDArray &arr, const int num_groups);

// Version of GetWeightsDesc for deconvolution (with swap)
inline mkldnn::memory::desc GetDeconvWeightsDesc(const NDArray &weights, const int num_groups) {
  return IOLogicalSwapDesc(GetWeightDesc(weights, num_groups), num_groups);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DECONVOLUTION_INL_H__

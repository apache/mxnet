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
 * \file dnnl_deconvolution-inl.h
 * Naming convention:
 *                 ________
 *  (src) data --->|Deconv|
 *     weights --->|  FWD |---> out (dst)
 *        bias --->|______|
 *                                 ________
 *        (diff_src) data_grad <---|Deconv|<--- out_grad (diff_dst)
 *  (diff_weight) weights_grad <---|  BWD |<--- data (src)
 *       (diff_bias) bias_grad <---|      |<--- weight
 *                                 |______|<--- bias
 *
 * "out" in this (and .cc) file will always refer to the output of Deconv FWD and
 * "out_grad" to its gradient. The corresponding DNNL names are in parentheses.
 */
#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_DECONVOLUTION_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <numeric>
#include <utility>
#include <vector>

#include "operator/nn/deconvolution-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

using deconv_fwd_t    = dnnl::deconvolution_forward;
using deconv_fwd_pd_t = dnnl::deconvolution_forward::primitive_desc;

using deconv_bwd_data_t    = dnnl::deconvolution_backward_data;
using deconv_bwd_data_pd_t = dnnl::deconvolution_backward_data::primitive_desc;

using deconv_bwd_weights_t    = dnnl::deconvolution_backward_weights;
using deconv_bwd_weights_pd_t = dnnl::deconvolution_backward_weights::primitive_desc;

// Swaps the logical order of dimensions that in plain format would correspond to input and output
// channels (for example: oihw => iohw, iohw => oihw, goihw => giohw).
inline dnnl::memory::desc IOLogicalSwapDesc(const dnnl::memory::desc& desc,
                                            const uint32_t num_group) {
  std::vector<int> order(desc.data.ndims);
  std::iota(std::begin(order), std::end(order), 0);
  const int offset = static_cast<int>(num_group > 1);
  std::swap(order[offset + 0], order[offset + 1]);
  return desc.permute_axes(order);
}

// Applies IOLogicalSwapDesc to DNNL memory of arr
inline void IOLogicalSwapDNNLMem(const NDArray& arr, const uint32_t num_group) {
  dnnl::memory::desc desc;
  if (arr.IsDNNLData()) {
    desc = arr.GetDNNLData()->get_desc();
  } else {
    // GetDNNLData won't take groups into account when creating dnnl::memory, we need to use
    // descriptor from GetWeightDesc but with default format
    const auto& temp = GetWeightDesc(arr, num_group);
    desc             = dnnl::memory::desc(
        temp.dims(),
        temp.data_type(),
        static_cast<dnnl::memory::format_tag>(GetDefaultFormat(temp.data.ndims)));
  }
  auto iOLogicalSwapDesc = IOLogicalSwapDesc(desc, num_group);
  const_cast<NDArray&>(arr).UpdateDNNLMemDesc(&iOLogicalSwapDesc);
}

// Version of GetWeightsDesc for deconvolution (with swap)
inline dnnl::memory::desc GetDeconvWeightsDesc(const NDArray& weights, const uint32_t num_group) {
  return IOLogicalSwapDesc(GetWeightDesc(weights, num_group), num_group);
}

class DNNLDeconvFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& data,
            const NDArray& weights,
            const NDArray* const bias,
            const NDArray& out);
    Tensors(const bool no_bias,
            const std::vector<NDArray>& inputs,
            const std::vector<NDArray>& outputs);

    const NDArray& data;
    const NDArray& weights;
    const NDArray* const bias;
    const NDArray& out;
  };

  static DNNLDeconvFwd& GetCached(const DeconvolutionParam& param, const Tensors& tensors);
  static std::shared_ptr<deconv_fwd_pd_t> CreatePrimitiveDesc(const DeconvolutionParam& param,
                                                              const Tensors& tensors);

  DNNLDeconvFwd(const DeconvolutionParam& param, const Tensors& tensors);
  void ControlWeightsFormat(const uint32_t num_group,
                            const bool is_train,
                            const NDArray& weights) const;
  void Execute(const uint32_t num_group, const OpReqType req, const Tensors& tensors) const;

 private:
  const dnnl::memory* DataMem(const NDArray& data) const;
  const dnnl::memory* WeightsMem(const uint32_t num_group, const NDArray& weights) const;
  const dnnl::memory* BiasMem(const NDArray& bias) const;

  dnnl_output_t OutMem(const OpReqType req, const NDArray& out) const;

 private:
  std::shared_ptr<deconv_fwd_t> fwd;
  std::shared_ptr<deconv_fwd_pd_t> fwd_pd;
};

inline const dnnl::memory* DNNLDeconvFwd::DataMem(const NDArray& data) const {
  auto fwd_src_desc = fwd_pd->src_desc();
  return data.GetDNNLDataReorder(&fwd_src_desc);
}

inline const dnnl::memory* DNNLDeconvFwd::WeightsMem(const uint32_t num_group,
                                                     const NDArray& weights) const {
  return GetWeights(weights, fwd_pd->weights_desc(), num_group);
}

inline const dnnl::memory* DNNLDeconvFwd::BiasMem(const NDArray& bias) const {
  return bias.GetDNNLData();
}

inline dnnl_output_t DNNLDeconvFwd::OutMem(const OpReqType req, const NDArray& out) const {
  return CreateDNNLMem(out, fwd_pd->dst_desc(), req);
}

class DNNLDeconvBwd {
 public:
  struct ReadTensors {
    ReadTensors(const bool no_bias, const std::vector<NDArray>& inputs);
    const NDArray& data;
    const NDArray& weights;
    const NDArray* const bias;
    const NDArray& out_grad;
  };
  struct WriteTensors {
    WriteTensors(const bool no_bias, const std::vector<NDArray>& outputs);
    const NDArray& data_grad;
    const NDArray& weights_grad;
    const NDArray* const bias_grad;
  };

  static DNNLDeconvBwd& GetCached(const DeconvolutionParam& param, const ReadTensors& read_tensors);

  static std::shared_ptr<deconv_bwd_data_pd_t> CreateDataPrimitiveDesc(
      const DeconvolutionParam& param,
      const ReadTensors& read_tensors,
      const deconv_fwd_pd_t& fwd_pd);

  static std::shared_ptr<deconv_bwd_weights_pd_t> CreateWeightsPrimitiveDesc(
      const DeconvolutionParam& param,
      const ReadTensors& read_tensors,
      const deconv_fwd_pd_t& fwd_pd);

  DNNLDeconvBwd(const DeconvolutionParam& param, const ReadTensors& read_tensors);

  void Execute(const uint32_t num_group,
               const std::vector<OpReqType>& req,
               const ReadTensors& read_tensors,
               const WriteTensors& write_tensors) const;

 private:
  void IOSwapWeightsTensors(const uint32_t num_group,
                            const std::vector<OpReqType>& req,
                            const NDArray& weights,
                            const NDArray& weights_grad) const;

  // returns the output gradient memory used to calculate the data (input) gradient,
  // which might be reused when calculating the gradient of weights
  const dnnl::memory* ScheduleBwdData(const uint32_t num_group,
                                      const OpReqType req,
                                      const ReadTensors& read_tensors,
                                      const WriteTensors& write_tensors) const;

  void ScheduleBwdWeights(const uint32_t num_group,
                          const std::vector<OpReqType>& req,
                          const ReadTensors& read_tensors,
                          const WriteTensors& write_tensors,
                          const dnnl::memory* const out_grad_mem) const;

  const dnnl::memory* DataMem(const NDArray& data) const;
  const dnnl::memory* WeightsMem(const uint32_t num_group, const NDArray& weights) const;

  // for calculating the gradient of data (input)
  const dnnl::memory* OutGradMem(const NDArray& out_grad) const;
  // for calculating the gradient of weights
  const dnnl::memory* OutGradMem(const NDArray& out_grad,
                                 const dnnl::memory* const out_grad_mem) const;

  dnnl_output_t DataGradMem(const OpReqType req, const NDArray& data_grad) const;
  dnnl_output_t WeightsGradMem(const uint32_t num_group,
                               const OpReqType req,
                               const NDArray& weights_grad) const;
  dnnl_output_t BiasGradMem(const OpReqType req, const NDArray* const bias) const;

  std::shared_ptr<deconv_bwd_data_pd_t> bwd_data_pd;
  std::shared_ptr<deconv_bwd_weights_pd_t> bwd_weights_pd;
  std::shared_ptr<deconv_bwd_data_t> bwd_data;
  std::shared_ptr<deconv_bwd_weights_t> bwd_weights;
};

inline void DNNLDeconvBwd::IOSwapWeightsTensors(const uint32_t num_group,
                                                const std::vector<OpReqType>& req,
                                                const NDArray& weights,
                                                const NDArray& weights_grad) const {
  if (req[deconv::kData]) {
    IOLogicalSwapDNNLMem(weights, num_group);
  }
  if (req[deconv::kWeight] || (req.size() < deconv::kBias && req[deconv::kBias])) {
    IOLogicalSwapDNNLMem(weights_grad, num_group);
  }
}

inline const dnnl::memory* DNNLDeconvBwd::DataMem(const NDArray& data) const {
  auto bwd_weight_src_desc = bwd_weights_pd->src_desc();
  return data.GetDNNLDataReorder(&bwd_weight_src_desc);
}

inline const dnnl::memory* DNNLDeconvBwd::WeightsMem(const uint32_t num_group,
                                                     const NDArray& weights) const {
  return GetWeights(weights, bwd_data_pd->weights_desc(), num_group);
}

inline const dnnl::memory* DNNLDeconvBwd::OutGradMem(const NDArray& out_grad) const {
  auto bwd_data_diff_desc = bwd_data_pd->diff_dst_desc();
  return out_grad.GetDNNLDataReorder(&bwd_data_diff_desc);
}

inline const dnnl::memory* DNNLDeconvBwd::OutGradMem(const NDArray& out_grad,
                                                     const dnnl::memory* const out_grad_mem) const {
  auto bwd_weight_diff_desc = bwd_weights_pd->diff_dst_desc();
  return (out_grad_mem && out_grad_mem->get_desc() == bwd_weight_diff_desc) ?
             out_grad_mem :
             out_grad.GetDNNLDataReorder(&bwd_weight_diff_desc);
}

inline dnnl_output_t DNNLDeconvBwd::DataGradMem(const OpReqType req,
                                                const NDArray& data_grad) const {
  return CreateDNNLMem(data_grad, bwd_data_pd->diff_src_desc(), req);
}

inline dnnl_output_t DNNLDeconvBwd::WeightsGradMem(const uint32_t num_group,
                                                   const OpReqType req,
                                                   const NDArray& weights_grad) const {
  // CreateDNNLWeightGrad always creates a new tensor as IsDefaultFormat always fails (because
  // of the logical swap - explained in DNNLDeconvFwd::Execute). We try to reuse weights_grad
  // memory (which, when not swapped, is always in default format), so here we check if after a
  // swap, weights_md will have a default format
  const auto& weights_md = bwd_weights_pd->diff_weights_desc();
  if (req == OpReqType::kWriteTo && IsDefaultFormat(IOLogicalSwapDesc(weights_md, num_group))) {
    return {OutDataOp::Noop, const_cast<NDArray&>(weights_grad).CreateDNNLData(&weights_md)};
  }
  return CreateDNNLWeightGrad(weights_grad, weights_md, req);
}

inline dnnl_output_t DNNLDeconvBwd::BiasGradMem(const OpReqType req,
                                                const NDArray* const bias) const {
  return bias ? CreateDNNLMem(*bias, bwd_weights_pd->diff_bias_desc(), req) :
                dnnl_output_t(OutDataOp::Noop, nullptr);
}

// Utility class for creating operation descriptors of deconvolution primitives
class DeconvDescCreator {
 public:
  DeconvDescCreator(const DeconvolutionParam& param,
                    const NDArray& data,
                    const NDArray& weights,
                    const NDArray* const bias,
                    const NDArray& out);

  // Imposes plain formats on memory descriptors with padding (so the next selected implementation
  // will pass CheckImplSizeReq). After calling this method, new primitive descriptor (with new
  // operator descriptor) should be created, which should select an implementation with matching
  // size requirements.
  // data_size, weights_size, out_size - size requirements of current implementation
  // Returns whether successfully imposed a plain format on any of the data, weights, and output
  // memory descriptors.
  bool ImposePlainWherePadding(const size_t data_size,
                               const size_t weights_size,
                               const size_t out_size);
  bool CheckImplSizeReq(const size_t data_size,
                        const size_t weights_size,
                        const size_t out_size) const;

  deconv_fwd_t::desc CreateFwdDesc() const;
  deconv_bwd_data_t::desc CreateBwdDataDesc() const;
  deconv_bwd_weights_t::desc CreateBwdWeightsDesc() const;

 private:
  dnnl::memory::desc data_md;
  dnnl::memory::desc weights_md;
  dnnl::memory::desc bias_md;
  dnnl::memory::desc out_md;

  dnnl::memory::dims strides;
  dnnl::memory::dims padding;
  dnnl::memory::dims dilates;
};

inline bool DeconvDescCreator::CheckImplSizeReq(const size_t data_size,
                                                const size_t weights_size,
                                                const size_t out_size) const {
  // DNNL introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, DNNL operators
  // still reuse memory from memory planning, so here we need to accept only a
  // kernel that has the expected memory size requirements (which is suboptimal)
  return (data_size == GetMemDescSize(data_md) && weights_size == GetMemDescSize(weights_md) &&
          out_size == GetMemDescSize(out_md));
}

inline deconv_fwd_t::desc DeconvDescCreator::CreateFwdDesc() const {
  return deconv_fwd_t::desc(dnnl::prop_kind::forward_training,
                            dnnl::algorithm::deconvolution_direct,
                            data_md,
                            weights_md,
                            bias_md,
                            out_md,
                            strides,
                            dilates,
                            padding,
                            padding);
}

inline deconv_bwd_data_t::desc DeconvDescCreator::CreateBwdDataDesc() const {
  return deconv_bwd_data_t::desc(dnnl::algorithm::deconvolution_direct,
                                 data_md,
                                 weights_md,
                                 out_md,
                                 strides,
                                 dilates,
                                 padding,
                                 padding);
}

inline deconv_bwd_weights_t::desc DeconvDescCreator::CreateBwdWeightsDesc() const {
  return deconv_bwd_weights_t::desc(dnnl::algorithm::deconvolution_direct,
                                    data_md,
                                    weights_md,
                                    bias_md,
                                    out_md,
                                    strides,
                                    dilates,
                                    padding,
                                    padding);
}

void DNNLDeconvolutionForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& in_data,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& out_data);

void DNNLDeconvolutionBackward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_DECONVOLUTION_INL_H__

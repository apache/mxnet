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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_rnn-inl.h
 * \brief Common functions used by MKLDNN RNN operator
 * \author Zixuan Wei
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <vector>
#include "../../rnn-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNRnnLayerParam {
  using memory = mkldnn::memory;
  using dims = mkldnn::memory::dims;

  int mode;
  bool bidirectional;
  bool state_outputs;
  int num_layer;
  int batch_size;
  int input_size;
  int state_size;
  int seq_len;

  dims src_dims;           // Dimensions of source input in format_tag::tnc
  dims weight_layer_dims;  // Dimensions of layer weights in format_tag::ldigo
  dims weight_iter_dims;   // Dimensions of iter weights in format_tag::ldigo
  dims bias_dims;          // Dimensions of bias in format_tag::ldgo
  dims dst_dims;           // Dimensions of output in format_tag::tnc
  dims state_dims;         // Dimensions of the state cell in format_tag::ldnc

  size_t workspace_size;  // used for the cached mkl-dnn memory in Forward inference
  size_t reserve_size;    // used for the reserved cached memory in Backward
  size_t single_w_size;   // weights size of a single cell
  size_t single_b_size;   // bias size of a single cell from mkl-dnn
  size_t naive_single_b_size;  // bias size of a single cell from framework
  size_t single_state_size;    // state size of a single cell, hy, cy

  MKLDNNRnnLayerParam(int num_layer, int batch_size, int seq_len,
                      int input_size, int state_size,
                      int mode, bool bidirectional = true)
      : mode(mode), bidirectional(bidirectional), state_outputs(true),
        num_layer(num_layer), batch_size(batch_size), input_size(input_size),
        state_size(state_size), seq_len(seq_len) { }

  void SetDims();
};

typedef std::vector<MKLDNNRnnLayerParam> LayerParamVector;
struct MKLDNNRnnFullParam {
  RNNParam default_param;
  LayerParamVector layer_params;
};

MKLDNNRnnFullParam MKLDNNRnnFullParamParser(const RNNParam& rnn_param, const int seq_len,
                                            const int batch_size, const int input_size);

/*
 * Use this to allocate memory from MKLDNNRnnOp temporary space.
 */
class MKLDNNRnnMemMgr {
  // The memory buffer in NDArray life-cycle
  NDArray workspace_;
  // This points to the memory buffer from a NDArray
  char *curr_mem;
  // The total bytes of the workspace of a MKLDNNRnnOp
  size_t mem_size = 0;
  // The current available memory bytes
  size_t curr_size = 0;
  const size_t alignment = kMKLDNNAlign;
  const mkldnn::engine& cpu_engine = CpuEngine::Get()->get_engine();
  // Here we hold all memory related to the stateful RNN operators
  std::vector<std::shared_ptr<const mkldnn::memory> > mem_holder;

 public:
  void Init(dim_t size, const Context& ctx, int dtype = mshadow::kFloat32);

  void RegisterMem(std::shared_ptr<const mkldnn::memory> mem) {
    mem_holder.push_back(mem);
  }

  mkldnn::memory *Alloc(const mkldnn::memory::desc &md);
};

/*
 * Rnn Primitive.
 */
class RnnPrimitive {
 public:
  /* Create a RnnPrimitive with rnn type:
   * lstm_forward, lbr_gru_forward, vanilla_rnn_forward
   */
  template<typename rnn_fwd, typename... Args>
  static RnnPrimitive Create(Args&&... args) {
    RnnPrimitive rnn_fwd_prim;
    auto fwd_desc = typename rnn_fwd::desc(std::forward<Args>(args)...);
    rnn_fwd_prim.fwd_pd_.reset(
      new typename rnn_fwd::primitive_desc(fwd_desc, CpuEngine::Get()->get_engine()),
      [](typename rnn_fwd::primitive_desc* pd) {
        delete reinterpret_cast<typename rnn_fwd::primitive_desc*>(pd);
      });
    auto fwd_pd = reinterpret_cast<typename rnn_fwd::primitive_desc*>(rnn_fwd_prim.fwd_pd_.get());
    rnn_fwd_prim.weights_layer_desc_ = fwd_pd->weights_layer_desc();
    rnn_fwd_prim.weights_iter_desc_  = fwd_pd->weights_iter_desc();
    rnn_fwd_prim.workspace_desc_ = fwd_pd->workspace_desc();

    rnn_fwd_prim.primitive_ = std::shared_ptr<mkldnn::primitive>(new rnn_fwd(*fwd_pd));

    return rnn_fwd_prim;
  }

  RnnPrimitive() {
    this->fwd_pd_ = nullptr;
    this->primitive_ = nullptr;
    this->weights_layer_desc_ = mkldnn::memory::desc();
    this->weights_iter_desc_ = mkldnn::memory::desc();
    this->workspace_desc_ = mkldnn::memory::desc();
  }

  RnnPrimitive(const RnnPrimitive& rnn_fwd_prim) {
    this->fwd_pd_ = rnn_fwd_prim.fwd_pd_;
    this->primitive_ = rnn_fwd_prim.primitive_;
    this->weights_layer_desc_ = rnn_fwd_prim.weights_layer_desc_;
    this->weights_iter_desc_ = rnn_fwd_prim.weights_iter_desc_;
    this->workspace_desc_ = rnn_fwd_prim.workspace_desc_;
  }

  RnnPrimitive& operator=(const RnnPrimitive& rnn_fwd_prim) {
    if (this != &rnn_fwd_prim) {
      this->fwd_pd_ = rnn_fwd_prim.fwd_pd_;
      this->primitive_ = rnn_fwd_prim.primitive_;
      this->weights_layer_desc_ = rnn_fwd_prim.weights_layer_desc_;
      this->weights_iter_desc_ = rnn_fwd_prim.weights_iter_desc_;
      this->workspace_desc_ = rnn_fwd_prim.workspace_desc_;
    }

    return *this;
  }

  const void* GetPrimDesc() const { return fwd_pd_.get(); }
  const mkldnn::primitive& GetPrim() const { return *primitive_; }

  const mkldnn::memory::desc& GetLayerDesc() const {
    return weights_layer_desc_;
  }

  const mkldnn::memory::desc& GetIterDesc() const {
    return weights_iter_desc_;
  }

  const mkldnn::memory::desc& GetWorkspaceDesc() const {
    return workspace_desc_;
  }

 private:
  std::shared_ptr<void> fwd_pd_;
  std::shared_ptr<mkldnn::primitive> primitive_;
  mkldnn::memory::desc weights_layer_desc_;
  mkldnn::memory::desc weights_iter_desc_;
  mkldnn::memory::desc workspace_desc_;
};

RnnPrimitive GetRnnFwdPrim(const MKLDNNRnnLayerParam &layer_param, const bool is_train,
                           const NDArray &data, const NDArray &params);

/*
 * Use this to manage memory and primitive of MKL-DNN RNN forward inference. 
 */
class MKLDNNRnnForward {
 public:
  MKLDNNRnnForward(const MKLDNNRnnLayerParam &layer_param, const bool is_train,
                   const NDArray &data, const NDArray &params)
      : initialized_(false), param_(layer_param),
        fwd_inf_(GetRnnFwdPrim(layer_param, false, data, params)) { }

  void SetNewDataMem(void* x, void* hx, void* cx,
                     void* y, void* hy, void* cy,
                     const int dtype = mshadow::kFloat32);
  void SetWeightsMem(MKLDNNRnnMemMgr* mgr, void* w_ptr, void* b_ptr,
                     const bool is_train = false,
                     const int dtype = mshadow::kFloat32);
  void ReorderWeights();

  const mkldnn::primitive& GetFwd() const { return fwd_inf_.GetPrim(); }

  const size_t GetSize(int dtype) const {
    size_t bytes = mshadow::mshadow_sizeof(dtype);
    size_t size = 0;
    size += fwd_inf_.GetLayerDesc().get_size();
    size += fwd_inf_.GetIterDesc().get_size();
    return size / bytes + 1;
  }

  const MKLDNNRnnLayerParam &GetParam() const { return param_; }

  const mkldnn_args_map_t &GetArgsMap() const { return net_args_; }

  const bool IsInitialized() const { return initialized_; }
  void Reset() { initialized_ = false; }

 private:
  bool initialized_;
  MKLDNNRnnLayerParam param_;
  RnnPrimitive fwd_inf_;    // forward inference primitive

  mkldnn::memory *weights_layer_ = nullptr;
  mkldnn::memory *weights_iter_ = nullptr;
  mkldnn::memory *bias_ = nullptr;

  mkldnn::memory *weights_layer_r_ = nullptr;
  mkldnn::memory *weights_iter_r_ = nullptr;

  /*
   * net_args must contain some keys as below:
   *   MKLDNN_ARG_SRC
   *   MKLDNN_ARG_SRC_ITER
   *   MKLDNN_WEIGHTS_LAYER
   *   MKLDNN_WEIGHTS_ITER
   *   MKLDNN_BIAS
   *   MKLDNN_ARG_DST
   *   MKLDNN_ARG_DST_ITER
   * if mode == Lstm, it also needs two additional key:
   *   MKLDNN_ARG_SRC_ITER_C
   *   MKLDNN_ARG_DST_ITER_C
   */
  mkldnn_args_map_t net_args_;

  friend class MKLDNNRnnForwardTraining;
};

typedef std::shared_ptr<mkldnn::memory> mkldnn_shared_mem_t;
/*
 * Use this to manage memory and primitive of MKL-DNN RNN forward training.
 */
class MKLDNNRnnForwardTraining {
 public:
  MKLDNNRnnForwardTraining(const MKLDNNRnnLayerParam &layer_param, const bool is_train,
                           const NDArray &data, const NDArray &params)
    : fwd_trn_(GetRnnFwdPrim(layer_param, is_train, data, params)),
      param_(&layer_param) { }

  void SetTrnMem(const MKLDNNRnnForward& fwd);
  void FetchData(const MKLDNNRnnForward& fwd);

  const MKLDNNRnnLayerParam& GetParam() const { return *param_; }
  const void* GetPrimDesc() const { return fwd_trn_.GetPrimDesc(); }
  const mkldnn::primitive& GetFwd() const { return fwd_trn_.GetPrim(); }
  const mkldnn_args_map_t& GetArgsMap() const { return net_args_; }

 private:
  RnnPrimitive fwd_trn_;
  const MKLDNNRnnLayerParam* param_;

  mkldnn_shared_mem_t weights_layer_ = nullptr;
  mkldnn_shared_mem_t weights_iter_ = nullptr;
  mkldnn::memory* bias_ = nullptr;

  mkldnn_shared_mem_t workspace_ = nullptr;

  // Key MKLDNN_ARGS_WORKSPACE must be included in forward training
  mkldnn_args_map_t net_args_;

  friend class MKLDNNRnnBackward;
};

/*
 * Rnn Backward primitive
 */
class RnnBwdPrimitive {
 public:
  template <typename rnn_fwd, typename rnn_bwd, typename... Args>
  static RnnBwdPrimitive Create(typename rnn_fwd::primitive_desc const & fwd_pd, Args&&... args) {
    RnnBwdPrimitive bwd_prim;
    typename rnn_bwd::desc bwd_desc(std::forward<Args>(args)...);
    typename rnn_bwd::primitive_desc bwd_pd(bwd_desc, CpuEngine::Get()->get_engine(), fwd_pd);
    bwd_prim.weights_layer_desc_ = bwd_pd.weights_layer_desc();
    bwd_prim.weights_iter_desc_ = bwd_pd.weights_iter_desc();
    bwd_prim.diff_weights_layer_desc_ = bwd_pd.diff_weights_layer_desc();
    bwd_prim.diff_weights_iter_desc_ = bwd_pd.diff_weights_iter_desc();
    bwd_prim.diff_bias_desc_ = bwd_pd.diff_bias_desc();

    bwd_prim.primitive_ = std::shared_ptr<mkldnn::primitive>(new rnn_bwd(bwd_pd));

    return bwd_prim;
  }

  RnnBwdPrimitive() {
    this->primitive_ = nullptr;
    this->weights_layer_desc_ = mkldnn::memory::desc();
    this->weights_iter_desc_ = mkldnn::memory::desc();
    this->diff_weights_layer_desc_ = mkldnn::memory::desc();
    this->diff_weights_iter_desc_ = mkldnn::memory::desc();
    this->diff_bias_desc_ = mkldnn::memory::desc();
  }

  RnnBwdPrimitive(const RnnBwdPrimitive& bwd) {
    this->primitive_ = bwd.primitive_;
    this->weights_layer_desc_ = bwd.weights_layer_desc_;
    this->weights_iter_desc_ = bwd.weights_iter_desc_;
    this->diff_weights_layer_desc_ = bwd.diff_weights_layer_desc_;
    this->diff_weights_iter_desc_ = bwd.diff_weights_iter_desc_;
    this->diff_bias_desc_ = bwd.diff_bias_desc_;
  }

  RnnBwdPrimitive& operator=(const RnnBwdPrimitive& bwd) {
    if (this != &bwd) {
      this->primitive_ = bwd.primitive_;
      this->weights_layer_desc_ = bwd.weights_layer_desc_;
      this->weights_iter_desc_ = bwd.weights_iter_desc_;
      this->diff_weights_layer_desc_ = bwd.diff_weights_layer_desc_;
      this->diff_weights_iter_desc_ = bwd.diff_weights_iter_desc_;
      this->diff_bias_desc_ = bwd.diff_bias_desc_;
    }

    return *this;
  }

 private:
  std::shared_ptr<mkldnn::primitive> primitive_;
  mkldnn::memory::desc weights_layer_desc_;
  mkldnn::memory::desc weights_iter_desc_;
  mkldnn::memory::desc diff_weights_layer_desc_;
  mkldnn::memory::desc diff_weights_iter_desc_;
  mkldnn::memory::desc diff_bias_desc_;
  friend class MKLDNNRnnBackward;
};
RnnBwdPrimitive GetRnnBwdPrim(const MKLDNNRnnForwardTraining& fwd,
                              const NDArray& data, const NDArray& params);

/*
 * Use this to manage memory and primitive of MKL-DNN RNN backward.
 */
class MKLDNNRnnBackward {
 public:
  MKLDNNRnnBackward(const MKLDNNRnnForwardTraining& fwd,
                    const NDArray& data, const NDArray& params)
      : bwd_(GetRnnBwdPrim(fwd, data, params)),
        fwd_ptr_(&fwd) { }

  void FetchDataWeightsMem(const MKLDNNRnnForwardTraining& fwd);
  void SetWeightsGradsMem();
  void SetDataGradsMem(void* diff_src, void* diff_state, void* diff_statecell,
                       void* diff_out, void* diff_state_out, void* diff_statecell_out,
                       const int dtype = mshadow::kFloat32);
  void CommitWeightsDiff(void* diff_weights, void* diff_bias,
                         const OpReqType req,
                         const int dtype = mshadow::kFloat32);

  const mkldnn::primitive& GetBwd() const { return *bwd_.primitive_; }
  const mkldnn_args_map_t& GetArgsMap() const { return net_args_; }

 private:
  bool initialized_;
  RnnBwdPrimitive bwd_;
  const MKLDNNRnnForwardTraining* fwd_ptr_;

  mkldnn_shared_mem_t weights_layer_;
  mkldnn_shared_mem_t weights_iter_;

  mkldnn_shared_mem_t diff_weights_layer_;
  mkldnn_shared_mem_t diff_weights_iter_;
  mkldnn_shared_mem_t diff_bias_;

  mkldnn_args_map_t net_args_;
};

/*
 * Use MKLDNNRnnOp to manage fused or unfused RNN layers. A MKLDNNRnnOp contains
 * the parameter(s) and primitive(s) of RNN layer(s). According to the direction,
 * input size, and state size, multple layers could be inplemented by unfused and
 * fused layers - MKLDNNRnnForward, which holds the memory and forward primitive
 * of MKL-DNN.
 */
class MKLDNNRnnOp {
 public:
  explicit MKLDNNRnnOp(const RNNParam &param, const int seq_len,
                       const int batch_size, const int input_size)
      : initialized_(false), weights_version_(0),
        full_param_(MKLDNNRnnFullParamParser(param, seq_len, batch_size, input_size)) { }

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs);

  const RNNParam& GetParam() const { return full_param_.default_param; }

 private:
  bool initialized_;
  size_t weights_version_;
  MKLDNNRnnFullParam full_param_;
  MKLDNNRnnMemMgr mgr_;
  std::vector<MKLDNNRnnForward> fwd_inf_vec_;              // forward inference layers
  std::vector<MKLDNNRnnForwardTraining> fwd_trn_vec_;      // forward training layers
  std::vector<MKLDNNRnnBackward> bwd_vec_;                 // backward layers

  // Used to store the intermediate results of multi-layer
  std::vector<mkldnn::memory *> dst_;

  // Used to store the intermediate diff_src of multi_layer
  mkldnn_shared_mem_t diff_src;

  void Init(const OpContext &ctx,
            const std::vector<NDArray> &inputs,
            const std::vector<OpReqType> &req,
            const std::vector<NDArray> &outputs);
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_INL_H_

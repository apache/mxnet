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
 * \file dnnl_rnn-inl.h
 * \brief Common functions used by DNNL RNN operator
 * \author Zixuan Wei
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_RNN_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_RNN_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <vector>

#include "operator/rnn-inl.h"
#include "dnnl_base-inl.h"
#include "operator/quantization/quantized_rnn-inl.h"

namespace mxnet {
namespace op {

struct DNNLRnnParam : public dmlc::Parameter<DNNLRnnParam> {
  bool quantized;

  DMLC_DECLARE_PARAMETER(DNNLRnnParam) {
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe(
        "Whether it's a quantized RNN operator");
  }
};

struct DNNLRnnLayerParam {
  using memory = dnnl::memory;
  using dims   = dnnl::memory::dims;

  int mode;
  bool bidirectional;
  bool state_outputs;
  int num_layer;
  index_t batch_size;
  index_t input_size;
  int state_size;
  int proj_size;
  index_t seq_len;

  dims src_dims;           // Dimensions of source input in format_tag::tnc
  dims weight_layer_dims;  // Dimensions of layer weights in format_tag::ldigo
  dims weight_iter_dims;   // Dimensions of iter weights in format_tag::ldigo
  dims weight_proj_dims;   // Dimensions of projection weights in format_tag::ldio
  dims bias_dims;          // Dimensions of bias in format_tag::ldgo
  dims dst_dims;           // Dimensions of output in format_tag::tnc
  dims state_dims;         // Dimensions of the state cell in format_tag::ldnc
  dims cell_dims;          // Dimensions of LSTM cell state in format_tag::ldnc

  size_t workspace_size;        // used for the cached DNNL memory in Forward inference
  size_t reserve_size;          // used for the reserved cached memory in Backward
  size_t single_w_size;         // weights size of a single cell
  size_t single_b_size;         // bias size of a single cell from DNNL
  size_t native_single_b_size;  // bias size of a single cell from framework
  size_t single_state_size;     // state size of a single cell, hy, cy

  bool quantized;         // whether this layer is quantized
  bool enable_u8_output;  // true by default, only be false when it is the last fusion layer of the
                          // quantized rnn operator

  DNNLRnnLayerParam(int num_layer,
                    index_t batch_size,
                    index_t seq_len,
                    index_t input_size,
                    int state_size,
                    int proj_size,
                    int mode,
                    bool bidirectional = true)
      : mode(mode),
        bidirectional(bidirectional),
        state_outputs(true),
        num_layer(num_layer),
        batch_size(batch_size),
        input_size(input_size),
        state_size(state_size),
        proj_size(proj_size),
        seq_len(seq_len),
        quantized(false),
        enable_u8_output(false) {}

  void SetDims();
};

typedef std::vector<DNNLRnnLayerParam> LayerParamVector;
struct DNNLRnnFullParam {
  RNNParam default_param;
  DNNLRnnParam dnnl_param;
  LayerParamVector layer_params;
};

DNNLRnnFullParam DNNLRnnFullParamParser(const nnvm::NodeAttrs& attrs,
                                        const index_t seq_len,
                                        const index_t batch_size,
                                        const index_t input_size);

/*
 * Use this to allocate memory from DNNLRnnOp temporary space.
 */
class DNNLRnnMemMgr {
  // The memory buffer in NDArray life-cycle
  NDArray workspace_;
  // This points to the memory buffer from a NDArray
  char* curr_mem = nullptr;
  // The total bytes of the workspace of a DNNLRnnOp
  size_t mem_size = 0;
  // The current available memory bytes
  size_t curr_size               = 0;
  const size_t alignment         = kDNNLAlign;
  const dnnl::engine& cpu_engine = CpuEngine::Get()->get_engine();
  // Here we hold all memory related to the stateful RNN operators
  std::vector<std::shared_ptr<const dnnl::memory> > mem_holder;

 public:
  /*!
   * \brief Initializer for RNN memory manager
   * \param size byte number
   * \param ctx Context of device enviroment
   */
  void Init(const dim_t size, const Context& ctx);

  // Return the bytes number of the buffer
  const size_t Size() {
    return mem_size;
  }

  void RegisterMem(std::shared_ptr<const dnnl::memory> mem) {
    mem_holder.push_back(mem);
  }

  dnnl::memory* Alloc(const dnnl::memory::desc& md);
};

typedef std::shared_ptr<dnnl::primitive_attr> shared_dnnl_attr_t;

/*
 * Rnn Primitive.
 */
class RnnPrimitive {
 public:
  /* Create a RnnPrimitive with rnn type:
   * lstm_forward, lbr_gru_forward, vanilla_rnn_forward
   */
  template <typename rnn_fwd, typename... Args>
  static RnnPrimitive Create(const shared_dnnl_attr_t attr, Args&&... args) {
    RnnPrimitive rnn_fwd_prim;
    auto fwd_desc = typename rnn_fwd::desc(std::forward<Args>(args)...);
    rnn_fwd_prim.fwd_pd_.reset(
        new typename rnn_fwd::primitive_desc(
            fwd_desc, attr ? *attr : dnnl::primitive_attr(), CpuEngine::Get()->get_engine()),
        [](void* pd) { delete reinterpret_cast<typename rnn_fwd::primitive_desc*>(pd); });
    auto fwd_pd = reinterpret_cast<typename rnn_fwd::primitive_desc*>(rnn_fwd_prim.fwd_pd_.get());
    rnn_fwd_prim.attr_               = attr;
    rnn_fwd_prim.weights_layer_desc_ = fwd_pd->weights_layer_desc();
    rnn_fwd_prim.weights_iter_desc_  = fwd_pd->weights_iter_desc();
    rnn_fwd_prim.weights_proj_desc_  = fwd_pd->weights_projection_desc();
    rnn_fwd_prim.workspace_desc_     = fwd_pd->workspace_desc();

    rnn_fwd_prim.primitive_ = std::shared_ptr<dnnl::primitive>(new rnn_fwd(*fwd_pd));

    return rnn_fwd_prim;
  }

  RnnPrimitive() {
    this->attr_               = nullptr;
    this->fwd_pd_             = nullptr;
    this->primitive_          = nullptr;
    this->weights_layer_desc_ = dnnl::memory::desc();
    this->weights_iter_desc_  = dnnl::memory::desc();
    this->weights_proj_desc_  = dnnl::memory::desc();
    this->workspace_desc_     = dnnl::memory::desc();
  }

  RnnPrimitive(const RnnPrimitive& rnn_fwd_prim) {
    this->attr_               = rnn_fwd_prim.attr_;
    this->fwd_pd_             = rnn_fwd_prim.fwd_pd_;
    this->primitive_          = rnn_fwd_prim.primitive_;
    this->weights_layer_desc_ = rnn_fwd_prim.weights_layer_desc_;
    this->weights_iter_desc_  = rnn_fwd_prim.weights_iter_desc_;
    this->weights_proj_desc_  = rnn_fwd_prim.weights_proj_desc_;
    this->workspace_desc_     = rnn_fwd_prim.workspace_desc_;
  }

  RnnPrimitive& operator=(const RnnPrimitive& rnn_fwd_prim) {
    if (this != &rnn_fwd_prim) {
      this->attr_               = rnn_fwd_prim.attr_;
      this->fwd_pd_             = rnn_fwd_prim.fwd_pd_;
      this->primitive_          = rnn_fwd_prim.primitive_;
      this->weights_layer_desc_ = rnn_fwd_prim.weights_layer_desc_;
      this->weights_iter_desc_  = rnn_fwd_prim.weights_iter_desc_;
      this->weights_proj_desc_  = rnn_fwd_prim.weights_proj_desc_;
      this->workspace_desc_     = rnn_fwd_prim.workspace_desc_;
    }

    return *this;
  }

  const void* GetPrimDesc() const {
    return fwd_pd_.get();
  }
  const dnnl::primitive& GetPrim() const {
    return *primitive_;
  }

  const dnnl::memory::desc& GetLayerDesc() const {
    return weights_layer_desc_;
  }

  const dnnl::memory::desc& GetIterDesc() const {
    return weights_iter_desc_;
  }

  const dnnl::memory::desc& GetProjDesc() const {
    return weights_proj_desc_;
  }

  const dnnl::memory::desc& GetWorkspaceDesc() const {
    return workspace_desc_;
  }

  const dnnl::primitive_attr& GetPrimAttr() const {
    return *attr_;
  }

 private:
  std::shared_ptr<void> fwd_pd_;
  std::shared_ptr<dnnl::primitive> primitive_;
  shared_dnnl_attr_t attr_;
  dnnl::memory::desc weights_layer_desc_;
  dnnl::memory::desc weights_iter_desc_;
  dnnl::memory::desc weights_proj_desc_;
  dnnl::memory::desc workspace_desc_;
};

RnnPrimitive GetRnnFwdPrim(const DNNLRnnLayerParam& layer_param,
                           const bool is_train,
                           const NDArray& data,
                           const NDArray& params,
                           const shared_dnnl_attr_t attr = nullptr);

/*
 * Use this to manage memory and primitive of DNNL RNN forward inference.
 */
class DNNLRnnForward {
 public:
  DNNLRnnForward(const Context ctx,
                 const DNNLRnnLayerParam& layer_param,
                 const bool is_train,
                 const NDArray& data,
                 const NDArray& params,
                 const shared_dnnl_attr_t attr = nullptr)
      : ctx_(ctx),
        initialized_(false),
        param_(layer_param),
        fwd_inf_(GetRnnFwdPrim(layer_param, false, data, params, attr)) {}

  void SetNewDataMem(void* x,
                     void* hx,
                     void* cx,
                     void* y,
                     void* hy,
                     void* cy,
                     const int dtype = mshadow::kFloat32);
  void SetWeightsMem(void* w_ptr,
                     void* b_ptr,
                     const bool is_train = false,
                     const int dtype     = mshadow::kFloat32);
  void ReorderWeights();

  const dnnl::primitive& GetFwd() const {
    return fwd_inf_.GetPrim();
  }

  void ResetFwd(const NDArray& data, const NDArray& params, const shared_dnnl_attr_t& attr) {
    fwd_inf_ = GetRnnFwdPrim(this->param_, false, data, params, attr);
  }

  const size_t GetSize() const {
    const size_t size = fwd_inf_.GetLayerDesc().get_size() + fwd_inf_.GetIterDesc().get_size() +
                        fwd_inf_.GetProjDesc().get_size();
    return size;
  }

  const DNNLRnnLayerParam& GetParam() const {
    return param_;
  }

  const dnnl_args_map_t& GetArgsMap() const {
    return net_args_;
  }

  const bool IsInitialized() const {
    return initialized_;
  }
  void Reset() {
    initialized_ = false;
  }

 private:
  Context ctx_;
  bool initialized_;
  DNNLRnnLayerParam param_;
  RnnPrimitive fwd_inf_;  // forward inference primitive

  DNNLRnnMemMgr mem_mgr_;
  dnnl::memory* weights_layer_ = nullptr;
  dnnl::memory* weights_iter_  = nullptr;
  dnnl::memory* weights_proj_  = nullptr;
  dnnl::memory* bias_          = nullptr;

  dnnl::memory* weights_layer_r_ = nullptr;
  dnnl::memory* weights_iter_r_  = nullptr;
  dnnl::memory* weights_proj_r_  = nullptr;

  /*
   * net_args must contain some keys as below:
   *   DNNL_ARG_SRC
   *   DNNL_ARG_SRC_ITER
   *   DNNL_WEIGHTS_LAYER
   *   DNNL_WEIGHTS_ITER
   *   DNNL_BIAS
   *   DNNL_ARG_DST
   *   DNNL_ARG_DST_ITER
   * if mode == Lstm, it also needs two additional key:
   *   DNNL_ARG_SRC_ITER_C
   *   DNNL_ARG_DST_ITER_C
   */
  dnnl_args_map_t net_args_;

  friend class DNNLRnnForwardTraining;
};

typedef std::shared_ptr<dnnl::memory> dnnl_shared_mem_t;
/*
 * Use this to manage memory and primitive of DNNL RNN forward training.
 */
class DNNLRnnForwardTraining {
 public:
  DNNLRnnForwardTraining(const DNNLRnnLayerParam& layer_param,
                         const bool is_train,
                         const NDArray& data,
                         const NDArray& params)
      : fwd_trn_(GetRnnFwdPrim(layer_param, is_train, data, params)), param_(&layer_param) {}

  void SetTrnMem(const DNNLRnnForward& fwd);
  void FetchData(const DNNLRnnForward& fwd);

  const DNNLRnnLayerParam& GetParam() const {
    return *param_;
  }
  const void* GetPrimDesc() const {
    return fwd_trn_.GetPrimDesc();
  }
  const dnnl::primitive& GetFwd() const {
    return fwd_trn_.GetPrim();
  }
  const dnnl_args_map_t& GetArgsMap() const {
    return net_args_;
  }

 private:
  RnnPrimitive fwd_trn_;
  const DNNLRnnLayerParam* param_;

  dnnl_shared_mem_t weights_layer_ = nullptr;
  dnnl_shared_mem_t weights_iter_  = nullptr;
  dnnl::memory* bias_              = nullptr;

  dnnl_shared_mem_t workspace_ = nullptr;

  // Key DNNL_ARGS_WORKSPACE must be included in forward training
  dnnl_args_map_t net_args_;

  friend class DNNLRnnBackward;
};

/*
 * Rnn Backward primitive
 */
class RnnBwdPrimitive {
 public:
  template <typename rnn_fwd, typename rnn_bwd, typename... Args>
  static RnnBwdPrimitive Create(typename rnn_fwd::primitive_desc const& fwd_pd, Args&&... args) {
    RnnBwdPrimitive bwd_prim;
    typename rnn_bwd::desc bwd_desc(std::forward<Args>(args)...);
    typename rnn_bwd::primitive_desc bwd_pd(bwd_desc, CpuEngine::Get()->get_engine(), fwd_pd);
    bwd_prim.weights_layer_desc_      = bwd_pd.weights_layer_desc();
    bwd_prim.weights_iter_desc_       = bwd_pd.weights_iter_desc();
    bwd_prim.diff_weights_layer_desc_ = bwd_pd.diff_weights_layer_desc();
    bwd_prim.diff_weights_iter_desc_  = bwd_pd.diff_weights_iter_desc();
    bwd_prim.diff_bias_desc_          = bwd_pd.diff_bias_desc();

    bwd_prim.primitive_ = std::shared_ptr<dnnl::primitive>(new rnn_bwd(bwd_pd));

    return bwd_prim;
  }

  RnnBwdPrimitive() {
    this->primitive_               = nullptr;
    this->weights_layer_desc_      = dnnl::memory::desc();
    this->weights_iter_desc_       = dnnl::memory::desc();
    this->diff_weights_layer_desc_ = dnnl::memory::desc();
    this->diff_weights_iter_desc_  = dnnl::memory::desc();
    this->diff_bias_desc_          = dnnl::memory::desc();
  }

  RnnBwdPrimitive(const RnnBwdPrimitive& bwd) {
    this->primitive_               = bwd.primitive_;
    this->weights_layer_desc_      = bwd.weights_layer_desc_;
    this->weights_iter_desc_       = bwd.weights_iter_desc_;
    this->diff_weights_layer_desc_ = bwd.diff_weights_layer_desc_;
    this->diff_weights_iter_desc_  = bwd.diff_weights_iter_desc_;
    this->diff_bias_desc_          = bwd.diff_bias_desc_;
  }

  RnnBwdPrimitive& operator=(const RnnBwdPrimitive& bwd) {
    if (this != &bwd) {
      this->primitive_               = bwd.primitive_;
      this->weights_layer_desc_      = bwd.weights_layer_desc_;
      this->weights_iter_desc_       = bwd.weights_iter_desc_;
      this->diff_weights_layer_desc_ = bwd.diff_weights_layer_desc_;
      this->diff_weights_iter_desc_  = bwd.diff_weights_iter_desc_;
      this->diff_bias_desc_          = bwd.diff_bias_desc_;
    }

    return *this;
  }

 private:
  std::shared_ptr<dnnl::primitive> primitive_;
  dnnl::memory::desc weights_layer_desc_;
  dnnl::memory::desc weights_iter_desc_;
  dnnl::memory::desc diff_weights_layer_desc_;
  dnnl::memory::desc diff_weights_iter_desc_;
  dnnl::memory::desc diff_bias_desc_;
  friend class DNNLRnnBackward;
};
RnnBwdPrimitive GetRnnBwdPrim(const DNNLRnnForwardTraining& fwd,
                              const NDArray& data,
                              const NDArray& params);

/*
 * Use this to manage memory and primitive of DNNL RNN backward.
 */
class DNNLRnnBackward {
 public:
  DNNLRnnBackward(const DNNLRnnForwardTraining& fwd, const NDArray& data, const NDArray& params)
      : bwd_(GetRnnBwdPrim(fwd, data, params)), fwd_ptr_(&fwd) {}

  void FetchDataWeightsMem(const DNNLRnnForwardTraining& fwd);
  void SetWeightsGradsMem();
  void SetDataGradsMem(void* diff_src,
                       void* diff_state,
                       void* diff_statecell,
                       void* diff_out,
                       void* diff_state_out,
                       void* diff_statecell_out,
                       const int dtype = mshadow::kFloat32);
  void SetNativeWeightsGrads() const;
  void CommitWeightsGrads(void* diff_weights,
                          void* diff_bias,
                          const OpReqType req,
                          const int dtype = mshadow::kFloat32);

  const dnnl::primitive& GetBwd() const {
    return *bwd_.primitive_;
  }
  const dnnl_args_map_t& GetArgsMap() const {
    return net_args_;
  }

 private:
  RnnBwdPrimitive bwd_;
  const DNNLRnnForwardTraining* fwd_ptr_;

  dnnl_shared_mem_t weights_layer_;
  dnnl_shared_mem_t weights_iter_;

  dnnl_shared_mem_t diff_weights_layer_;
  dnnl_shared_mem_t diff_weights_iter_;
  dnnl_shared_mem_t diff_weights_layer_r_;
  dnnl_shared_mem_t diff_weights_iter_r_;
  dnnl_shared_mem_t diff_bias_;

  dnnl_args_map_t net_args_;
};

/*
 * Use DNNLRnnOp to manage fused or unfused RNN layers. A DNNLRnnOp contains
 * the parameter(s) and primitive(s) of RNN layer(s). According to the direction,
 * input size, and state size, multple layers could be inplemented by unfused and
 * fused layers - DNNLRnnForward, which holds the memory and forward primitive
 * of DNNL.
 */
class DNNLRnnOp {
 public:
  explicit DNNLRnnOp(const nnvm::NodeAttrs& attrs,
                     const int seq_len,
                     const int batch_size,
                     const int input_size)
      : initialized_(false),
        weights_version_(0),
        full_param_(DNNLRnnFullParamParser(attrs, seq_len, batch_size, input_size)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs);

  const RNNParam& GetParam() const {
    return full_param_.default_param;
  }

 private:
  bool initialized_;
  size_t weights_version_;
  DNNLRnnFullParam full_param_;
  DNNLRnnMemMgr mgr_;
  std::vector<DNNLRnnForward> fwd_inf_vec_;          // forward inference layers
  std::vector<DNNLRnnForwardTraining> fwd_trn_vec_;  // forward training layers
  std::vector<DNNLRnnBackward> bwd_vec_;             // backward layers

  // Used to store the intermediate results of multi-layer
  std::vector<dnnl::memory*> dst_;

  // Used to store the intermediate diff_src of multi_layer
  dnnl_shared_mem_t diff_src;

  void Init(const OpContext& ctx,
            const std::vector<NDArray>& inputs,
            const std::vector<OpReqType>& req,
            const std::vector<NDArray>& outputs);
};

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_rnn.html
inline bool SupportDNNLRnn(const int input_dtype) {
  if (input_dtype == mshadow::kFloat32 && dmlc::GetEnv("MXNET_USE_ONEDNN_RNN", 1)) {
    return true;
  }
  return false;
}

inline bool SupportDNNLRnn(const RNNParam& param, const int input_dtype) {
  if (param.use_sequence_length)
    return false;
  return SupportDNNLRnn(input_dtype);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_RNN_INL_H_

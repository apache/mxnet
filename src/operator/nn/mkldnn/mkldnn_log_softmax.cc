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
 * \file mkldnn_log_softmax.cc
 * \brief Implementation of log_softmax function with MKLDNN support
*/

#include "../softmax-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "../../../3rdparty/parallel-hashmap/parallel_hashmap/phmap.h"

#if MXNET_USE_ONEDNN == 1
namespace mxnet {
namespace op {

static mkldnn::logsoftmax_forward::primitive_desc GetLogSoftmaxFwdPd(
                                  bool is_train,
                                  const int axis,
                                  const mkldnn::memory &input_mem) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training
                       : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::logsoftmax_forward::desc(prop, data_md, axis);
  return mkldnn::logsoftmax_forward::primitive_desc(desc, cpu_engine);
}

static mkldnn::logsoftmax_backward::primitive_desc GetLogSoftmaxBwdPd(
                                const mkldnn::memory &diff_mem,
                                const mkldnn::memory &data_mem,
                                const int axis,
                                const mkldnn::logsoftmax_forward::primitive_desc &hint_fwd_pd) {
  mkldnn::memory::desc diff_md = diff_mem.get_desc();
  mkldnn::memory::desc data_md = data_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto desc = mkldnn::logsoftmax_backward::desc(diff_md, data_md, axis);
  return mkldnn::logsoftmax_backward::primitive_desc(desc, cpu_engine, hint_fwd_pd);
}


bool SupportMKLDNNLogSoftmax(const SoftmaxParam &param,
                             const NDArray &data,
                             const NDArray &output) {
  const int ndim = data.shape().ndim();
  const int in_dtype = data.dtype();
  const int out_dtype = output.dtype();
  const int axis = CheckAxis(param.axis, ndim);
  // MKLDNN does not support temperature argument in their log_softmax function
  // now. Need update this once they start to support it.
  // Currently, MKLDNN shows bad performance when log_softmax is not performed on the last dimension
  if (param.temperature.has_value() ||
      in_dtype != mshadow::kFloat32 ||
      in_dtype != out_dtype ||
      axis != (ndim - 1)) {
    return false;
  }

  // only supports ndim = 1, 2, 3, 4 for now
  return (ndim >= 1 && ndim <= 4);
}

class MKLDNNLogSoftmaxFwd {
 public:
  mkldnn::logsoftmax_forward::primitive_desc pd;

  MKLDNNLogSoftmaxFwd(const bool is_train,
                      const int axis,
                      const mkldnn::memory &input) : pd(GetLogSoftmaxFwdPd(is_train, axis, input)) {
    fwd_ = std::make_shared<mkldnn::logsoftmax_forward>(pd);
  }

  const mkldnn::logsoftmax_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::logsoftmax_forward> fwd_;
};

typedef ParamOpSign<SoftmaxParam> MKLDNNSoftmaxSignature;

static MKLDNNLogSoftmaxFwd &GetLogSoftmaxFwd(const SoftmaxParam &param,
                                             const int real_axis,
                                             const bool is_train,
                                             const NDArray &data,
                                             const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local phmap::flat_hash_map<MKLDNNSoftmaxSignature,
                                          MKLDNNLogSoftmaxFwd,
                                          OpHash> fwds;
#else
  static MX_THREAD_LOCAL phmap::flat_hash_map<MKLDNNSoftmaxSignature,
                                            MKLDNNLogSoftmaxFwd,
                                            OpHash> fwds;
#endif

  MKLDNNSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNLogSoftmaxFwd fwd(is_train, real_axis, *(data.GetMKLDNNData()));
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNLogSoftmaxForward(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const NDArray &in_data,
                             const OpReqType &req,
                             const NDArray &out_data) {
  if (req == kNullOp) return;
  // same as the FCompute path, log_softmax only supports kWriteTo and kWriteInplace for now.
  CHECK_NE(req, kAddTo);

  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, in_data.shape().ndim());
  auto fwd = GetLogSoftmaxFwd(param, axis, ctx.is_train, in_data, out_data);

  auto in_mem = in_data.GetMKLDNNData();
  auto out_mem = out_data.GetMKLDNNData(fwd.pd.dst_desc());
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), {{MKLDNN_ARG_SRC, *in_mem}, {MKLDNN_ARG_DST, *out_mem}});
  stream->Submit();
}

class MKLDNNLogSoftmaxBwd {
 public:
  mkldnn::logsoftmax_backward::primitive_desc pd;

  MKLDNNLogSoftmaxBwd(const mkldnn::memory &diff_mem,
                      const mkldnn::memory &data_mem,
                      const int axis,
                      const mkldnn::logsoftmax_forward::primitive_desc &hint_fwd_pd) :
                                    pd(GetLogSoftmaxBwdPd(diff_mem, data_mem, axis, hint_fwd_pd)) {
    bwd_ = std::make_shared<mkldnn::logsoftmax_backward>(pd);
  }

  const mkldnn::logsoftmax_backward &GetBwd() const {
    return *bwd_;
  }

 private:
  std::shared_ptr<mkldnn::logsoftmax_backward> bwd_;
};

static MKLDNNLogSoftmaxBwd &GetLogSoftmaxBwd(const SoftmaxParam &param,
                                             const int real_axis,
                                             const std::vector<NDArray> &data,
                                             const std::vector<NDArray> &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local phmap::flat_hash_map<MKLDNNSoftmaxSignature,
                                         MKLDNNLogSoftmaxBwd,
                                         OpHash> bwds;
#else
  static MX_THREAD_LOCAL phmap::flat_hash_map<MKLDNNSoftmaxSignature,
                                            MKLDNNLogSoftmaxBwd,
                                            OpHash> bwds;
#endif

  MKLDNNSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(data);
  key.AddSign(output);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto diff_mem = data[0].GetMKLDNNData();
    auto data_mem = data[1].GetMKLDNNData();
    auto fwd_pd = GetLogSoftmaxFwdPd(true, real_axis, *data_mem);
    MKLDNNLogSoftmaxBwd bwd(*diff_mem, *data_mem, real_axis, fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNLogSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data) {
  if (req[0] == kNullOp) return;
  CHECK_EQ(in_data.size(), 2U);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, in_data[1].shape().ndim());
  auto diff_mem = in_data[0].GetMKLDNNData();
  auto data_mem = in_data[1].GetMKLDNNData();
  auto bwd = GetLogSoftmaxBwd(param, axis, in_data, out_data);

  auto out_mem = CreateMKLDNNMem(out_data[0], bwd.pd.diff_src_desc(), req[0]);
  MKLDNNStream *stream = MKLDNNStream::Get();
  mkldnn_args_map_t args = {
    { MKLDNN_ARG_DST, *data_mem },
    { MKLDNN_ARG_DIFF_DST, *diff_mem },
    { MKLDNN_ARG_DIFF_SRC, *out_mem.second }
  };

  stream->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(out_data[0], out_mem);
  stream->Submit();
}

}   // namespace op
}   // namespace mxnet
#endif

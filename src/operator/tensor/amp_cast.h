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
 * \file amp_cast.h
 * \brief Function definition of casts used by AMP
 */

#ifndef MXNET_OPERATOR_TENSOR_AMP_CAST_H_
#define MXNET_OPERATOR_TENSOR_AMP_CAST_H_

#include <vector>
#include <utility>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct AMPCastParam : public dmlc::Parameter<AMPCastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(AMPCastParam) {
    DMLC_DECLARE_FIELD(dtype)
    MXNET_ADD_ALL_TYPES
    .describe("Output data type.");
  }
};

struct AMPMultiCastParam : public dmlc::Parameter<AMPMultiCastParam> {
  int num_outputs;
  bool cast_narrow;

  DMLC_DECLARE_PARAMETER(AMPMultiCastParam) {
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("Number of input/output pairs to be casted to the widest type.");
    DMLC_DECLARE_FIELD(cast_narrow).set_default(false)
    .describe("Whether to cast to the narrowest type");
  }
};

inline bool AMPCastType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  using mshadow::kFloat32;
  using mshadow::kFloat16;
  using mshadow::kBfloat16;
  const AMPCastParam& param = nnvm::get<AMPCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0] == kFloat32 || (*in_attrs)[0] == kFloat16 || (*in_attrs)[0] == kBfloat16) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  }
  return (*in_attrs)[0] != -1;
}

inline bool AMPMultiCastType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  using mshadow::kFloat32;
  using mshadow::kFloat16;
  using mshadow::kBfloat16;
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_outputs);
  CHECK_EQ(out_attrs->size(), param.num_outputs);
  bool ret = true;
  int widest_type = param.cast_narrow ? kFloat32 : (*in_attrs)[0];
  for (int i = 0; i < param.num_outputs; ++i) {
    if (!param.cast_narrow && ((*in_attrs)[i] == kFloat32 || (*out_attrs)[i] == kFloat32)) {
      widest_type = kFloat32;
    } else if (param.cast_narrow && ((*in_attrs)[i] == kFloat16 || (*out_attrs)[i] == kFloat16)) {
      widest_type = kFloat16;
    } else if (param.cast_narrow && ((*in_attrs)[i] == kBfloat16 || (*out_attrs)[i] == kBfloat16)) {
      widest_type = kBfloat16;
    }
  }
  for (int i = 0; i < param.num_outputs; ++i) {
    if ((*in_attrs)[i] == kFloat32 || (*in_attrs)[i] == kFloat16 || (*in_attrs)[i] == kBfloat16) {
      TYPE_ASSIGN_CHECK(*out_attrs, i, widest_type);
    } else {
      TYPE_ASSIGN_CHECK(*out_attrs, i, (*in_attrs)[i]);
    }
    ret = ret && ((*in_attrs)[i] != -1);
  }
  return ret;
}

inline bool AMPMultiCastShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  const AMPMultiCastParam& param = dmlc::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_outputs);
  CHECK_EQ(out_attrs->size(), param.num_outputs);

  bool all_inferred = true;
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    // forward inference
    SHAPE_ASSIGN_CHECK(*out_attrs, i, (*in_attrs)[i]);
    // backward inference
    SHAPE_ASSIGN_CHECK(*in_attrs, i, (*out_attrs)[i]);
    all_inferred = all_inferred && !shape_is_none((*in_attrs)[i]);
  }
  return all_inferred;
}

template<typename xpu>
void AMPCastCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      if (outputs[0].type_flag_ != inputs[0].type_flag_ ||
          req[0] != kWriteInplace) {
        Assign(out, req[0], tcast<DstDType>(data));
      }
    });
  });
}

template<typename xpu>
void AMPMultiCastCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  for (size_t i = 0; i < outputs.size(); ++i) {
    MSHADOW_TYPE_SWITCH(outputs[i].type_flag_, DstDType, {
      Tensor<xpu, 1, DstDType> out = outputs[i].FlatTo1D<xpu, DstDType>(s);
      MSHADOW_TYPE_SWITCH(inputs[i].type_flag_, SrcDType, {
        Tensor<xpu, 1, SrcDType> data = inputs[i].FlatTo1D<xpu, SrcDType>(s);
        if (outputs[i].type_flag_ != inputs[i].type_flag_ ||
            req[i] != kWriteInplace) {
          Assign(out, req[i], tcast<DstDType>(data));
        }
      });
    });
  }
}

#if MXNET_USE_MKLDNN == 1
static void AMPCastExCPU(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kWriteInplace) {
    return;
  }
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  auto data = inputs[0];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();
  const auto i_mem = data.GetMKLDNNData();
  const size_t i_ndim = data.shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(data.shape()[i]);
  }
  const auto o_desc =
      mkldnn::memory::desc(i_dims, get_mkldnn_type(outputs[0].dtype()),
                           static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(i_ndim)));
  const auto out_mem = CreateMKLDNNMem(outputs[0], o_desc, req[0]);
  mkldnn_args_map_t reorder_args;
  reorder_args[MKLDNN_ARG_SRC] = *i_mem;
  reorder_args[MKLDNN_ARG_DST] = *out_mem.second;
  MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::reorder(*i_mem, *out_mem.second), reorder_args);
  MKLDNNStream::Get()->Submit();
}

inline static bool AMPCastStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                      DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  auto ret = MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
  return ret;
}

static void AMPMultiCastExCPU(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                              const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_outputs);
  CHECK_EQ(outputs.size(), param.num_outputs);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  for (int i = 0; i < param.num_outputs; ++i) {
    if (req[i] == kWriteInplace) {
      continue;
    }
    auto data = inputs[i];
    if (data.IsView() && data.IsMKLDNNData())
      data = data.Reorder2Default();
    const auto i_mem = data.GetMKLDNNData();
    const size_t i_ndim = data.shape().ndim();
    mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
    for (size_t j = 0; j < i_ndim; j++) {
      i_dims[j] = static_cast<int>(data.shape()[j]);
    }
    const auto o_desc =
        mkldnn::memory::desc(i_dims, get_mkldnn_type(outputs[i].dtype()),
                             static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(i_ndim)));
    const auto out_mem = CreateMKLDNNMem(outputs[i], o_desc, req[i]);
    mkldnn_args_map_t reorder_args;
    reorder_args[MKLDNN_ARG_SRC] = *i_mem;
    reorder_args[MKLDNN_ARG_DST] = *out_mem.second;
    MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::reorder(*i_mem, *out_mem.second), reorder_args);
  }
  MKLDNNStream::Get()->Submit();
}

inline static bool AMPMultiCastStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                           DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  const AMPMultiCastParam& param = nnvm::get<AMPMultiCastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_outputs);
  CHECK_EQ(out_attrs->size(), param.num_outputs);
  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

#endif  // MXNET_USE_MKLDNN == 1

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_AMP_CAST_H_

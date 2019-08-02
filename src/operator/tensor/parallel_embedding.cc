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
 * Copyright (c) 2017 by Contributors
 * \file parallel_embedding.cc
 * \brief CPU implementation of parallel embedding
 * \author Lingyan Guo
*/

#include <string>
#include <vector>
#include "parallel_embedding.h"
namespace mxnet {
namespace op {

static EmbeddingParam GetEmbeddedParam(const ParallelEmbeddingParam& param_, int i) {
  EmbeddingParam embedding_param;
  embedding_param.input_dim = param_.input_dims[i];
  embedding_param.output_dim = param_.output_dims[i];
  embedding_param.dtype = param_.dtypes[i];
  embedding_param.sparse_grad = param_.sparse_grads[i];
  return embedding_param;
}
// storage type inference function for Embedding
inline bool ParallelEmbeddingOpForwardStorageType(const nnvm::NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  DispatchMode* dispatch_mode,
                                                  std::vector<int>* in_attrs,
                                                  std::vector<int>* out_attrs) {
  const ParallelEmbeddingParam& param_ =
      nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
  bool ret = true;
  for (int i = 0; i < param_.num_args; ++i) {
    nnvm::NodeAttrs attrs;
    attrs.parsed = GetEmbeddedParam(param_, i);
    std::vector<int> e_in;
    std::vector<int> e_out;
    int& d = (*in_attrs)[i * 2];
    int& w = (*in_attrs)[i * 2 + 1];
    e_in.push_back(d);
    e_in.push_back(w);
    int& o = (*out_attrs)[i];
    e_out.push_back(o);
    ret &= EmbeddingOpForwardStorageType(attrs, dev_mask, dispatch_mode, &e_in,
                                         &e_out);
    o = e_out[0];
    w = e_in[1];
  }
  return ret;
}

static bool ParallelEmbeddingOpShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape>* in_shape,
                                     std::vector<TShape>* out_shape) {
  const ParallelEmbeddingParam& param_ =
      nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
  bool ret = true;
  for (int i = 0; i < param_.num_args; i++) {
    nnvm::NodeAttrs attrs;
    attrs.parsed = GetEmbeddedParam(param_, i);
    std::vector<TShape> e_in;
    std::vector<TShape> e_out;
    TShape& d = (*in_shape)[i * 2];
    TShape& w = (*in_shape)[i * 2 + 1];
    e_in.push_back(d);
    e_in.push_back(w);
    TShape& o = (*out_shape)[i];
    e_out.push_back(o);
    ret &= EmbeddingOpShape<EmbeddingParam>(attrs, &e_in, &e_out);
    o = e_out[0];
    w = e_in[1];
  }
  return ret;
}

inline bool ParallelEmbeddingOpType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_type,
                                    std::vector<int>* out_type) {
  const ParallelEmbeddingParam& param_ =
      nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
  bool ret = true;
  for (int i = 0; i < param_.num_args; i++) {
    nnvm::NodeAttrs attrs;
    attrs.parsed = GetEmbeddedParam(param_, i);
    std::vector<int> e_in;
    std::vector<int> e_out;
    int& d = (*in_type)[i * 2];
    int& w = (*in_type)[i * 2 + 1];
    e_in.push_back(d);
    e_in.push_back(w);
    int& o = (*out_type)[i];
    e_out.push_back(o);
    ret &= EmbeddingOpType<EmbeddingParam>(attrs, &e_in, &e_out);
    o = e_out[0];
    w = e_in[1];
  }
  return ret;
}
template <typename xpu>
void ParallelEmbeddingOpForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const ParallelEmbeddingParam& param_ =
      nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
#pragma omp parallel for num_threads(param_.num_args)
  for (int i = 0; i < param_.num_args; i++) {
    nnvm::NodeAttrs attrs;
    attrs.parsed = GetEmbeddedParam(param_, i);
    std::vector<TBlob> e_in;
    std::vector<TBlob> e_out;
    const TBlob& d = (inputs)[i * 2];
    const TBlob& w = (inputs)[i * 2 + 1];
    e_in.push_back(d);
    e_in.push_back(w);
    const TBlob& o = (outputs)[i];
    e_out.push_back(o);
    EmbeddingOpForward<cpu>(attrs, ctx, e_in, req, e_out);
  }
}
template <typename IType, typename DType>
struct TakeCPUInfo {
  DType* out_data;
  DType* in_data;
  IType* idx;
  int N;
  size_t M;
  int64_t K;
};

template <typename xpu>
void ParallelSparseEmbeddingOpForwardEx(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  const ParallelEmbeddingParam& param_ =
      nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
  using namespace mxnet_op;
  using namespace rowsparse;

  typedef float IType;
  typedef float DType;
  TakeCPUInfo<IType, DType>* takecpu_info =
      new TakeCPUInfo<IType, DType>[param_.num_args];
  for (int em = 0; em < param_.num_args; em++) {
    const NDArray& d = (inputs)[em * 2];
    const NDArray& w = (inputs)[em * 2 + 1];
    const NDArray& o = (outputs)[em];
    const TShape& oshape = o.shape();
    const TShape& wshape = w.shape();
    takecpu_info[em].N = oshape.Size() / wshape[1];
    takecpu_info[em].out_data = o.data().dptr<DType>();
    takecpu_info[em].in_data = w.data().dptr<DType>();
    takecpu_info[em].idx = d.data().dptr<IType>();
    takecpu_info[em].M = wshape[1];
    takecpu_info[em].K = wshape[0];
  }

  bool clip = true;
  int em = 0;
  int i = 0;
  int N = takecpu_info[0].N;
  const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#ifdef _MSC_VER
  #pragma omp parallel for num_threads(omp_threads)
#else
  #pragma omp parallel for num_threads(omp_threads) collapse(2)
#endif  // _MSC_VER
  for (em = 0; em < param_.num_args; em++)
    for (i = 0; i < N; ++i) {
      int64_t j = static_cast<int64_t>(takecpu_info[em].idx[i]);
      if (clip) {
        if (j <= 0)
          j = 0;
        else if (j >= takecpu_info[em].K)
          j = takecpu_info[em].K - 1;
      } else {
        j = j % takecpu_info[em].K;
        j += (j < 0) ? takecpu_info[em].K : 0;
      }
      std::memcpy(takecpu_info[em].out_data + i * takecpu_info[em].M,
                  takecpu_info[em].in_data + j * takecpu_info[em].M,
                  takecpu_info[em].M * sizeof(DType));
    }

  delete[] takecpu_info;
}

DMLC_REGISTER_PARAMETER(ParallelEmbeddingParam);

NNVM_REGISTER_OP(ParallelEmbedding)
.describe(R"code( Parallel exec embedding in Mulit-core CPU

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
    const ParallelEmbeddingParam& params = nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
    return params.num_args*2;
})
.set_num_outputs([](const NodeAttrs& attrs) {
    const ParallelEmbeddingParam& params = nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
    return params.num_args;
})
.set_attr_parser(ParamParser<ParallelEmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
    const ParallelEmbeddingParam& params = nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
        ret.push_back(std::string("arg_") + std::to_string(i));
        ret.push_back(std::string("embed_") + std::to_string(i) + std::string("_weight"));
    }
    return ret;
})
.set_attr<nnvm::FListInputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    const ParallelEmbeddingParam& params = nnvm::get<ParallelEmbeddingParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
        ret.push_back(std::string("out_") + std::to_string(i));
    }
    return ret;
})
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<mxnet::FInferShape>("FInferShape", ParallelEmbeddingOpShape)
.set_attr<nnvm::FInferType>("FInferType", ParallelEmbeddingOpType)
.set_attr<FInferStorageType>("FInferStorageType", ParallelEmbeddingOpForwardStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
    [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", ParallelEmbeddingOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ParallelSparseEmbeddingOpForwardEx<cpu>)
.add_argument("data_weight", "NDArray-or-Symbol[]",
              "List of arrays (data/weight) to embedding weight.")
.add_arguments(ParallelEmbeddingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

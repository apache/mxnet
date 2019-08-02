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

#include "../slice_channel-inl.h"
#include "indexing_op.h"
#include "matrix_op-inl.h"
#include "slice_split_embedding.h"

namespace mxnet {
namespace op {

bool ConcatSetShape(mxnet::ShapeVector *in_shape,
        mxnet::ShapeVector *out_shape, int num_args, int dim);

// call from SliceOpShape
static TShape get_slice_output_shape(
    const mxnet::Tuple<dmlc::optional<int>>& _pbegin,
    const mxnet::Tuple<dmlc::optional<int>>& _pend,
    const mxnet::Tuple<dmlc::optional<int>>& _pstep,
    const mxnet::TShape& dshape) {
  TShape oshape(dshape);
  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
  common::StaticArray<index_t, ndim> begin, end, step;
  GetIndexRange(dshape, _pbegin, _pend, _pstep, &begin, &end, &step);
  for (index_t i = 0; i < _pbegin.ndim(); ++i) {
    const int b = begin[i], e = end[i], s = step[i];
    SetSliceOpOutputDimSize(i, b, e, s, &oshape);
  }
  });
  return oshape;
}

static EmbeddingParam GetEmbeddedParam(
    const SliceSplitEmbeddingConcatFuseParam& param_, int i) {
  EmbeddingParam embedding_param;
  embedding_param.input_dim = param_.input_dims[i];
  embedding_param.output_dim = param_.output_dims[i];
  embedding_param.dtype = mshadow::kFloat32;
  embedding_param.sparse_grad = false;
  return embedding_param;
}
static bool SliceSplitEmbeddingConcatOpShape(const nnvm::NodeAttrs& attrs,
                                             std::vector<TShape>* in_shape,
                                             std::vector<TShape>* out_shape) {
  const SliceSplitEmbeddingConcatFuseParam& param_ =
      nnvm::get<SliceSplitEmbeddingConcatFuseParam>(attrs.parsed);
  bool ret = true;
  TShape& dshape = (*in_shape)[0];

  mxnet::Tuple<dmlc::optional<int>> param_step;
  TShape cont_slice_oshape = get_slice_output_shape(param_.cont_begin,
                        param_.cont_end, param_step, dshape);
  TShape split_slice_oshape = get_slice_output_shape(param_.embed_begin,
                        param_.embed_end, param_step, dshape);
  std::vector<TShape> split_in_shapes;
  split_in_shapes.push_back(split_slice_oshape);
  std::vector<TShape> split_out_shapes;
  split_out_shapes.resize(param_.num_outputs);
  std::vector<TShape> split_aux_shapes;
  SliceChannelInferShape(&split_in_shapes, &split_out_shapes, &split_aux_shapes,
                         param_.num_outputs, 1, param_.squeeze_axis);
  std::vector<TShape> embed_out_shapes;

  for (int i = 0; i < param_.num_outputs; ++i) {
    nnvm::NodeAttrs em_attrs;
    em_attrs.parsed = GetEmbeddedParam(param_, i);
    std::vector<TShape> e_in;
    std::vector<TShape> e_out;
    e_in.push_back(split_out_shapes[i]);
    e_in.push_back((*in_shape)[1 + i]);
    e_out.resize(1);
    EmbeddingOpShape<EmbeddingParam>(em_attrs, &e_in, &e_out);
    SHAPE_ASSIGN_CHECK(*in_shape, i + 1, e_in[1]);
    embed_out_shapes.push_back(e_out[0]);
  }
  embed_out_shapes.push_back(cont_slice_oshape);
  ConcatSetShape(&embed_out_shapes, out_shape, param_.num_outputs + 1,
                 param_.concat_dim);
  return ret;
}

inline bool SliceSplitEmbeddingConcatOpType(const nnvm::NodeAttrs& attrs,
                                            std::vector<int>* in_type,
                                            std::vector<int>* out_type) {
  bool ret = true;
  (*out_type)[0] = (*in_type)[0];
  int in_size = (*in_type).size();
  for (int i = 1; i < in_size; i++) (*in_type)[i] = (*out_type)[0];
  return ret;
}

inline bool SliceSplitEmbeddingConcatOpStorageType(
    const nnvm::NodeAttrs& attrs, const int dev_mask,
    DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
    std::vector<int>* out_attrs) {
  bool dispatched = false;
  auto& out_stype = out_attrs->at(0);

  dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                   DispatchMode::kFComputeEx);

  return dispatched;
}

template <int ndim, int req, typename xpu>
struct slice_forward_window;
template <int ndim, int req>
struct slice_forward_window<ndim, req, cpu> {
  // i is the i-th row after flattening out into 2D tensor
  template <typename DType>
  MSHADOW_XINLINE static void Map(
      int i, DType* out, const DType* data, const mshadow::Shape<ndim> dshape,
      const mshadow::Shape<ndim> oshape,
      const common::StaticArray<mxnet::index_t, ndim> begin,
      const common::StaticArray<mxnet::index_t, ndim> step, int out_count_per_row) {
    const int data_last_dim_size = dshape[ndim - 1];
    const int out_last_dim_size = oshape[ndim - 1];
    const int step_last_dim = step[ndim - 1];
    const int begin_last_dim = begin[ndim - 1];
    int out_offset = i * out_last_dim_size;
    for (int j = 0; j < out_count_per_row; ++j) {
      int irow = 0;
      int stride = 1;
      int idx = i;
#pragma unroll
      for (int k = ndim - 2; k >= 0; --k) {
        irow += stride * ((idx % oshape[k]) * step[k] + begin[k]);
        idx /= oshape[k];
        stride *= dshape[k];
      }
      KERNEL_ASSIGN(
          out[out_offset++], req,
          data[irow * data_last_dim_size + j * step_last_dim + begin_last_dim]);
    }
  }
};
template <typename IType, typename DType>
struct TakeCPUInfoWindow {
  DType* in_data;
  int idx_offset;
  int out_offset;

  size_t M;
  int64_t K;
};
template <typename xpu>
void SliceSplitEmbeddingConcatOpForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<TBlob>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& outputs) {
  const SliceSplitEmbeddingConcatFuseParam& param_ =
      nnvm::get<SliceSplitEmbeddingConcatFuseParam>(attrs.parsed);
  // by default Cont_features is in the first
  TShape dshape = inputs[0].shape_;
  TShape oshape = outputs[0].shape_;

  // For Cont feature
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  mxnet::Tuple<dmlc::optional<int>> param_step;
  TShape cont_slice_oshape = get_slice_output_shape(param_.cont_begin, param_.cont_end,
                         param_step, dshape);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<mxnet::index_t, ndim> begin, end, step;
    GetIndexRange(data.shape_, param_.cont_begin, param_.cont_end, param_step,
                  &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(
        out.type_flag_, DType, {MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          int num_threads = out.shape_.FlatTo2D()[0];
          if (std::is_same<xpu, gpu>::value) {
            num_threads *= out.shape_.get<ndim>()[ndim - 1];
          }
          mxnet_op::Kernel<slice_forward_window<ndim, Req, xpu>, xpu>::Launch(
              s, num_threads, out.dptr<DType>(), data.dptr<DType>(),
              data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step,
              cont_slice_oshape[ndim - 1]);
        })})
  })
  // Here make assumption steps is 1

  using namespace mxnet_op;
  using namespace rowsparse;

  typedef float IType;
  typedef float DType;

  int ndim = data.ndim();
  int cont_slice_last_dim_size = cont_slice_oshape[ndim - 1];
  int data_last_dim_size = dshape[ndim - 1];
  int out_last_dim_size = oshape[ndim - 1];
  int emb_in_last_dim_size =
      (data_last_dim_size - cont_slice_last_dim_size) / param_.num_outputs;
  int emb_out_last_dim_size =
      (out_last_dim_size - cont_slice_last_dim_size) / param_.num_outputs;
  int batch_size = dshape.Size() / data_last_dim_size;
  const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  TakeCPUInfoWindow<IType, DType>* takecpu_info =
      new TakeCPUInfoWindow<IType, DType>[param_.num_outputs];
  DType* out_data = out.dptr<DType>();
  IType* idx = data.dptr<IType>();
  for (int em = 0; em < param_.num_outputs; ++em) {
    const TBlob& w = (inputs)[em + 1];
    const TShape& wshape = w.shape_;
    takecpu_info[em].idx_offset = em * emb_in_last_dim_size;
    takecpu_info[em].out_offset =
        cont_slice_last_dim_size + em * emb_out_last_dim_size;
    takecpu_info[em].in_data = w.dptr<DType>();
    takecpu_info[em].M = wshape[1];
    takecpu_info[em].K = wshape[0];
  }
  bool clip = true;
  int em = 0;
  int i = 0;
  int N = batch_size;
#ifdef _MSC_VER
  #pragma omp parallel for num_threads(omp_threads)
#else
  #pragma omp parallel for num_threads(omp_threads) collapse(2)
#endif  // _MSC_VER
  for (em = 0; em < param_.num_outputs; ++em)
    for (i = 0; i < N; ++i) {
      int64_t j = static_cast<int64_t>(
          *(idx + takecpu_info[em].idx_offset + i * data_last_dim_size));
      if (clip) {
        if (j <= 0)
          j = 0;
        else if (j >= takecpu_info[em].K)
          j = takecpu_info[em].K - 1;
      } else {
        j = j % takecpu_info[em].K;
        j += (j < 0) ? takecpu_info[em].K : 0;
      }
      std::memcpy(
          out_data + takecpu_info[em].out_offset + i * out_last_dim_size,
          takecpu_info[em].in_data + j * takecpu_info[em].M,
          takecpu_info[em].M * sizeof(DType));
    }

  delete[] takecpu_info;
  return;
}
static void MxnetFallBackCompute(FCompute fn, const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  std::vector<TBlob> in_blobs(inputs.size());
  std::vector<NDArray> in_bufs;
  for (size_t i = 0; i < in_blobs.size(); ++i) {
    in_blobs[i] = inputs[i].data();
  }

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); ++i) {
    NDArray output = outputs[i];
    out_blobs[i] = output.data();
  }

  fn(attrs, ctx, in_blobs, req, out_blobs);
}

template <typename xpu>
void SliceSplitEmbeddingConcatOpForwardEx(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  MxnetFallBackCompute(SliceSplitEmbeddingConcatOpForward<cpu>, attrs, ctx,
                       inputs, req, outputs);
}
DMLC_REGISTER_PARAMETER(SliceSplitEmbeddingConcatFuseParam);

NNVM_REGISTER_OP(SliceSplitEmbeddingConcatFuse)
.describe(R"code( Fuse Slice Split Embedding Concat for Wide & Deep Model
)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const SliceSplitEmbeddingConcatFuseParam& params =
        nnvm::get<SliceSplitEmbeddingConcatFuseParam>(attrs.parsed);
  return 1 + params.num_outputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  return 1;
})
.set_attr_parser(ParamParser<SliceSplitEmbeddingConcatFuseParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  const SliceSplitEmbeddingConcatFuseParam& params =
        nnvm::get<SliceSplitEmbeddingConcatFuseParam>(attrs.parsed);
  std::vector<std::string> ret;
  ret.emplace_back(std::string("dns_data"));
  for (int i = 0; i < params.num_outputs; ++i) {
    ret.emplace_back(std::string("embed_") + std::to_string(i) + std::string("_weight"));
  }
  return ret;
})
.set_attr<nnvm::FListInputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  std::vector<std::string> ret = { "out_data" };
  return ret;
})
.set_attr<std::string>("key_var_num_args", "num_outputs")
.set_attr<mxnet::FInferShape>("FInferShape", SliceSplitEmbeddingConcatOpShape)
.set_attr<nnvm::FInferType>("FInferType", SliceSplitEmbeddingConcatOpType)
.set_attr<FInferStorageType>("FInferStorageType", SliceSplitEmbeddingConcatOpStorageType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", SliceSplitEmbeddingConcatOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SliceSplitEmbeddingConcatOpForwardEx<cpu>)
.add_argument("data_weight", "NDArray-or-Symbol[]",
              "List of arrays (data/weight) to embedding weight.")
.add_arguments(SliceSplitEmbeddingConcatFuseParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

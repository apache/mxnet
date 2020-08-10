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
 * Copyright (c) 2018 by Contributors
 * \file np_indexing_op.cc
*/

#include "./np_indexing_op.h"

namespace mxnet {
namespace op {

struct AdvancedIndexingTakeCPU {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const IType* idx, const size_t M, const int64_t K) {
    int64_t j = static_cast<int64_t>(idx[i]);
    j = j % K;
    j += (j < 0) ? K : 0;
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    std::memcpy(out_data + i * M, in_data + j * M, M * sizeof(DType));
#pragma GCC diagnostic pop
  }
};

struct AdvancedIndexingTakeMultiDimensionCPU {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const IType* idx, const size_t M, const int64_t K) {
    int64_t j = static_cast<int64_t>(idx[i]);
    j = j % K;
    j += (j < 0) ? K : 0;
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    std::memcpy(out_data + i * M, in_data + (i * K + j) * M, M * sizeof(DType));
#pragma GCC diagnostic pop
  }
};

struct AdvancedIndexingBooleanMaskBackwardCPUWriteKernel {
  template<typename DType>
  static void Map(int i,
                  DType* igrad,
                  const OpReqType /*req*/,
                  const DType* ograd,
                  const int32_t* idx,
                  const size_t col_size) {
    // i is row id already
    int32_t prev = (i == 0) ? 0 : idx[i - 1];
    int32_t curr = idx[i];
#pragma GCC diagnostic push
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    if (prev != curr) {
      std::memcpy(igrad + i * col_size, ograd + prev * col_size, col_size * sizeof(DType));
    } else {
      std::memset(igrad + i * col_size, 0, col_size * sizeof(DType));
    }
#pragma GCC diagnostic pop
  }
};

template<typename DType>
bool CheckIndexOutOfBound(const DType* data_ptr, size_t data_size,
                          const DType min, const DType max) {
  bool is_valid = true;
  for (size_t i = 0; i < data_size; i++) {
    if (data_ptr[i] > max || data_ptr[i] < min) {
      is_valid = false;
      break;
    }
  }
  return is_valid;
}

template<typename DType>
void GatherNDCheckBoundCPU(mshadow::Stream<cpu> *s, const DType* idx_ptr, index_t N,
                        index_t M, const mshadow::Shape<10> mshape, DType* is_valid_dim_ptr) {
  using namespace mxnet_op;
  Kernel<set_zero, cpu>::Launch(s, M, is_valid_dim_ptr);
  Kernel<is_valid_check_gather_nd, cpu>::Launch(s, M, is_valid_dim_ptr, idx_ptr, N, mshape);
  for (int m = 0; m < M; m++) {
    if (is_valid_dim_ptr[m] > mshape[m] - 1 || is_valid_dim_ptr[m] < - mshape[m]) {
      LOG(FATAL)<< "IndexError: index " << is_valid_dim_ptr[m] << " is out of bounds for axis "
        << m << " with size " << mshape[m];
    }
  }
}

inline bool AdvancedIndexingOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[1], -1) << "Index type must be set for take operator";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*in_attrs)[0] != -1;
}

bool AdvancedIndexingOpStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

bool AdvancedIndexingOpBackStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 2);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  for (int & out_attr : *out_attrs)
    out_attr = kDefaultStorage;
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

template<>
void AdvancedIndexingOpForward<cpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
  using namespace mxnet_op;
  if (req[np_indexing_::kOut] == kNullOp) return;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[np_indexing_::kIdx].dtype() == mshadow::kBool) {
    CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
    const NDArray &data = inputs[0];
    const NDArray &idx = inputs[1];
    const NDArray &out = outputs[0];
    CHECK_EQ(data.shape()[0], idx.shape()[0]);
    CHECK_EQ(idx.shape().ndim(), 1U);  // idx is required to be 1-d.
    // count the number of 1s in `idx`, so that we could know the output dimension
    size_t idx_size = idx.shape()[0];
    std::vector<int32_t> prefix_sum(idx_size, 0);
    size_t valid_num = 0;
    // Calculate prefix sum
    bool* idx_dptr = idx.data().dptr<bool>();
    for (size_t i = 0; i < idx_size; i++) {
      prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
      prefix_sum[i] += (idx_dptr[i]) ? 1 : 0;
    }
    valid_num = prefix_sum[idx_size - 1];
    // set the output shape forcefully
    mxnet::TShape s = data.shape();
    s[0] = valid_num;

    const_cast<NDArray &>(out).Init(s);
    // do the copy
    MSHADOW_TYPE_SWITCH_WITH_BOOL(data.dtype(), DType, {
      size_t input_size = data.shape().Size();
      size_t col_size = input_size / idx_size;
      mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
      mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
        stream, idx_size, out.data().dptr<DType>(), data.data().dptr<DType>(),
        prefix_sum.data(), col_size);
    });
  } else if (inputs[np_indexing_::kIdx].dtype() == mshadow::kInt8 ||
             inputs[np_indexing_::kIdx].dtype() == mshadow::kInt16 ||
             inputs[np_indexing_::kIdx].dtype() == mshadow::kInt32 ||
             inputs[np_indexing_::kIdx].dtype() == mshadow::kInt64) {
    using namespace mshadow;
    const mxnet::TShape& idxshape = inputs[np_indexing_::kIdx].shape();
    const mxnet::TShape& arrshape = inputs[np_indexing_::kArr].shape();

    if (idxshape.Size() == 0) {
      return;
    }

    mxnet::TShape oshape(idxshape.ndim() + arrshape.ndim() - 1, -1);
    for (index_t i = 0; i < idxshape.ndim(); ++i) {
      oshape[i] = idxshape[i];
    }
    for (index_t i = 0; i < arrshape.ndim(); i++) {
      if (i < 0) {
        oshape[i] = arrshape[i];
      } else if (i > 0) {
        oshape[i + idxshape.ndim() - 1] = arrshape[i];
      }
    }

    const NDArray &out = outputs[0];
    const_cast<NDArray &>(out).Init(oshape);

    Stream<cpu> *s = ctx.get_stream<cpu>();

    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[np_indexing_::kOut].dtype(), DType, {  // output data type
      MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[np_indexing_::kIdx].dtype(), IType, {  // index data type
        IType min = 0;
        IType max = static_cast<IType>(arrshape[0] - 1);
        // check with single thread is faster since data is small
        IType* idx_ptr = inputs[np_indexing_::kIdx].data().dptr<IType>();
        size_t idx_size = idxshape.Size();
        bool is_valid = CheckIndexOutOfBound(idx_ptr, idx_size, min, max);
        CHECK(is_valid) << "take operator contains indices out of bound";
        Kernel<AdvancedIndexingTakeCPU, cpu>::Launch(s, idxshape.Size(),
                                                  outputs[np_indexing_::kOut].data().dptr<DType>(),
                                                  inputs[np_indexing_::kArr].data().dptr<DType>(),
                                                  inputs[np_indexing_::kIdx].data().dptr<IType>(),
                                                  oshape.Size()/idxshape.Size(), arrshape[0]);
      });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type. "
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

template<>
void AdvancedIndexingOpBackward<cpu>(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  if (req[0] == kNullOp) return;

  if (inputs[np_indexing_::kIdx+1].dtype() == mshadow::kBool) {
    // inputs: {ograd, data, idx}
    // outputs: {igrad_data, igrad_idx}
    const NDArray& ograd = inputs[0];
    const NDArray& idx = inputs[2];
    const NDArray& igrad_data = outputs[0];
    MSHADOW_TYPE_SWITCH(igrad_data.dtype(), DType, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), IType, {
        size_t input_size = igrad_data.shape().Size();
        size_t idx_size = idx.shape()[0];
        size_t col_size = input_size / idx_size;
        std::vector<int32_t> prefix_sum(idx_size, 0);
        bool* idx_dptr = idx.data().dptr<bool>();
        for (size_t i = 0; i < idx_size; i++) {
          prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
          prefix_sum[i] += (idx_dptr[i]) ? 1 : 0;
        }
        mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
        if (req[0] == kAddTo) {
          mxnet_op::Kernel<BooleanMaskBackwardKernel, cpu>::Launch(
            stream, idx_size, igrad_data.data().dptr<DType>(), req[0],
            ograd.data().dptr<DType>(), prefix_sum.data(), col_size);
        } else {
          mxnet_op::Kernel<AdvancedIndexingBooleanMaskBackwardCPUWriteKernel, cpu>::Launch(
            stream, idx_size, igrad_data.data().dptr<DType>(), req[0],
            ograd.data().dptr<DType>(), prefix_sum.data(), col_size);
        }
      });
    });
  } else if (inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt8 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt16 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt32 ||
             inputs[np_indexing_::kIdx+1].dtype() == mshadow::kInt64) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_NE(req[np_indexing_::kIdx], kAddTo)
      << "take layer doesn't support gradient of req type kAddTo to index";

    // grad_out is the gradient of the outputs in the feed-forward
    // grad_in is the gradient of the inputs in the feed-forward
    Stream<cpu> *s = ctx.get_stream<cpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {  // output data type
      MSHADOW_TYPE_SWITCH(inputs[2].dtype(), IType, {  // index data type
        // inputs are specified in the .cc file, which are the gradients from
        // the upper layer and the input index
        // outputs are the gradients of inputs in the feed-forward pass
        const mxnet::TShape& idxshape = inputs[2].shape();
        const mxnet::TShape& arrshape = outputs[0].shape();
        const mxnet::TShape& oshape = inputs[0].shape();

        if (idxshape.Size() == 0) {
          return;
        }

        if (req[np_indexing_::kIdx] != kNullOp) {
          mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
            s, idxshape.Size(), outputs[np_indexing_::kIdx].data().dptr<IType>());
        }

        int idxndim = idxshape.ndim();
        Tensor<cpu, 1, IType> idx = inputs[2].data().get_with_shape<cpu, 1, IType>(
            Shape1(idxshape.ProdShape(0, idxndim)), s);
        Tensor<cpu, 2, DType> grad_out = inputs[0].data().get_with_shape<cpu, 2, DType>(
            Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);
        Tensor<cpu, 2, DType> grad_in = outputs[0].data().get_with_shape<cpu, 2, DType>(
            Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);

        // re-using the previous code for axis = 0 case
        if (req[np_indexing_::kArr] == kWriteTo || req[np_indexing_::kArr] == kAddTo) {
          if (req[np_indexing_::kArr] == kWriteTo) {
            grad_in = scalar<DType>(0.0f);
          }
          AddTakeGrad<false>(grad_in, idx, grad_out);
        } else {
          LOG(FATAL) << "wrong req";
        }
      });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type. "
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

void AdvancedIndexingMultipleForwardCPU(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;

  if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kBool) {
    LOG(FATAL)
    << "Multi-dimension boolean indexing is not supported.";
  } else if (inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt8 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt16 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt32 ||
             inputs[np_indexing_::kIdx].type_flag_ == mshadow::kInt64) {
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    const mxnet::TShape& dshape = inputs[0].shape_;
    const mxnet::TShape& ishape = inputs[1].shape_;
    int M = ishape[0];
    int N = ishape.Size() / M;
    int K = dshape.ProdShape(M, dshape.ndim());
    mshadow::Shape<10> strides;
    mshadow::Shape<10> mshape;
    for (int i = M-1, stride = K; i >= 0; stride *= dshape[i], --i) {
      strides[i] = stride;
      mshape[i] = dshape[i];
    }
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {  // output data type switch
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // indices data type switch
        // check whether indices are out of bound
        IType* idx_ptr = inputs[1].dptr<IType>();
        Tensor<cpu, 1, IType> workspace =
          ctx.requested[0].get_space_typed<cpu, 1, IType>(Shape1(M), s);
        IType* is_valid_dim_ptr = reinterpret_cast<IType*>(workspace.dptr_);
        GatherNDCheckBoundCPU(s, idx_ptr, N, M, mshape, is_valid_dim_ptr);
        Kernel<gather_nd, cpu>::Launch(
          s, N, req[0], N, M, K, strides, mshape, outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<IType>());
      });
    });
  } else {
    LOG(FATAL)
    << "arrays used as indices must be explictly declared as integer (or boolean) type."
    << "Use np.astype() to cast indices to integer or boolean.";
  }
}

template<typename DType, typename IType>
inline typename std::enable_if<(!std::is_same<DType, mshadow::half::half_t>::value), void>::type
GatherNDBackwardImpl(index_t N, index_t M, index_t K,
                     const mshadow::Shape<10> strides,
                     DType* out,
                     const DType* data,
                     const IType* indices,
                     mshadow::Stream<cpu> *s) {
#pragma omp parallel for
  for (index_t i = 0; i < N; i++) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<index_t>(indices[j*N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
#pragma omp atomic
      out[offset + j] += data[i * K + j];
    }
  }
}

template<typename DType, typename IType>
inline typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value, void>::type
GatherNDBackwardImpl(index_t N, index_t M, index_t K,
                     const mshadow::Shape<10> strides,
                     DType* out,
                     const DType* data,
                     const IType* indices,
                     mshadow::Stream<cpu> *s) {
  for (index_t i = 0; i < N; i++) {
    index_t offset = 0;
    for (index_t j = 0; j < M; ++j) {
      offset += strides[j] * static_cast<index_t>(indices[j*N + i]);
    }
    for (index_t j = 0; j < K; ++j) {
      out[offset + j] += data[i * K + j];
    }
  }
}

NNVM_REGISTER_OP(_npi_advanced_indexing)
.describe(R"code(
Combination of boolean indexing and advanced ndarray indexing
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferType>("FInferType", AdvancedIndexingOpType)
.set_attr<FComputeEx>("FComputeEx<cpu>", AdvancedIndexingOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_np_advanced_indexing"})
.set_attr<FInferStorageType>("FInferStorageType", AdvancedIndexingOpStorageType)
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("indices", "NDArray-or-Symbol", "Indices");

NNVM_REGISTER_OP(_backward_np_advanced_indexing)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", AdvancedIndexingOpBackStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", AdvancedIndexingOpBackward<cpu>);

NNVM_REGISTER_OP(_npi_advanced_indexing_multiple)
.describe(R"code(
Combination of multiple boolean indexing and advanced indexing
)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", GatherNDShape)
.set_attr<nnvm::FInferType>("FInferType", GatherNDType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", AdvancedIndexingMultipleForwardCPU)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_backward_np_advanced_indexing_multiple");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);

    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(p);
    ret.emplace_back(zero);
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices");

NNVM_REGISTER_OP(_backward_np_advanced_indexing_multiple)
.describe(R"code(Accumulates data according to indices and get the result. It's the backward of
`_npi_advanced_indexing_multiple`.
)code")
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr_parser(ParamParser<ScatterNDParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ScatterNDShape)
.set_attr<nnvm::FInferType>("FInferType", ScatterNDType)
.set_attr<FCompute>("FCompute<cpu>", AdvancedIndexingMultipleBackward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_npi_advanced_indexing_multiple");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_indices",
                         {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(p);
    ret.emplace_back(zero);
    return ret;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.add_argument("data", "NDArray-or-Symbol", "data")
.add_argument("indices", "NDArray-or-Symbol", "indices")
.add_arguments(ScatterNDParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

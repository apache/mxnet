/*!
 * Copyright (c) 2017 by Contributors
 * \file indexing_op.h
 * \brief
 * \author Bing Xu, Siyi Li, Chi Zhang
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEXING_OP_H_
#define MXNET_OPERATOR_TENSOR_INDEXING_OP_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/omp.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <type_traits>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#include "./sort_op.h"

namespace mxnet {
namespace op {

namespace embedding {
enum EmbeddingOpInputs {kData, kWeight};
enum EmbeddingOpOutputs {kOut};
enum EmbeddingOpResource {kTempSpace};
}  // namespace embedding

struct EmbeddingParam: public dmlc::Parameter<EmbeddingParam> {
  int input_dim;
  int output_dim;
  int dtype;
  DMLC_DECLARE_PARAMETER(EmbeddingParam) {
    DMLC_DECLARE_FIELD(input_dim).set_lower_bound(1)
    .describe("Vocabulary size of the input indices.");
    DMLC_DECLARE_FIELD(output_dim).set_lower_bound(1)
    .describe("Dimension of the embedding vectors.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Data type of weight.");
  }
};

/*!
 * \brief CPU/GPU: Return the amount of temporary storage in bytes required by
                   AddTakeGradLargeBatch
 * \param num_items number of keys
 */
template <typename IndexType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, cpu>::value, size_t>::type
AddTakeGradLargeBatchWorkspaceSize(size_t num_keys) {
  return 0;
}
/*!
 * \brief CPU/GPU: Return the amount of temporary storage in bytes required by
                   AddTakeGradLargeBatch
 * \param num_items number of keys
 */
template <typename IndexType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
AddTakeGradLargeBatchWorkspaceSize(size_t num_keys);
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[sorted[i]] += src[index[i]]
                   Called when the batchsize of src is larger than the featuredim
 * \param dst destination
 * \param sorted the sorted indices
 * \param index original index of the sorted indices
 * \param src source output
 * \param workspace (optional) temporary storage
 */
template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(mshadow::Tensor<cpu, 2, DType> dst,
                                  const mshadow::Tensor<cpu, 1, IndexType>& sorted,
                                  const mshadow::Tensor<cpu, 1, IndexType>& index,
                                  const mshadow::Tensor<cpu, 2, DType> &src,
                                  mshadow::Tensor<cpu, 1, char>* workspace = NULL) {
  for (index_t y = 0; y < sorted.size(0); ++y) {
    dst[sorted[y]] += src[index[y]];
  }
}
/*!
 * \brief CPU/GPU: Gradient accumulate of embedding matrix.
                   dst[sorted[i]] += src[index[i]]
                   Called when the batchsize of src is larger than the featuredim
 * \param dst destination
 * \param sorted the sorted indices
 * \param index original index of the sorted indices
 * \param src source output
 * \param workspace (optional) temporary storage
 */
template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(mshadow::Tensor<gpu, 2, DType> dst,
                                  const mshadow::Tensor<gpu, 1, IndexType>& sorted,
                                  const mshadow::Tensor<gpu, 1, IndexType>& index,
                                  const mshadow::Tensor<gpu, 2, DType> &src,
                                  mshadow::Tensor<gpu, 1, char>* workspace = NULL);

inline bool EmbeddingOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  using namespace mshadow;
  const TShape &dshape = (*in_attrs)[embedding::kData];
  if (dshape.ndim() ==  0) return false;
  const EmbeddingParam& param = nnvm::get<EmbeddingParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*in_attrs, embedding::kWeight, Shape2(param.input_dim,
                                                           param.output_dim));
  out_attrs->clear();

  TShape oshape(dshape.ndim()+1);
  for (size_t i = 0; i < dshape.ndim(); ++i) {
    oshape[i] = dshape[i];
  }
  oshape[dshape.ndim()] = param.output_dim;

  out_attrs->push_back(oshape);
  return true;
}

inline bool EmbeddingOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type,
                            std::vector<int> *out_type) {
  const EmbeddingParam& param = nnvm::get<EmbeddingParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 2U);
  CHECK_GE(out_type->size(), 1U);
  int itype = (*in_type)[0];
  CHECK_NE(itype, -1) << "First input must have specified type";
  int dtype_in = (*in_type)[1];
  int dtype_out = (*out_type)[0];
  int dtype = param.dtype;
  if (dtype_in != -1 && dtype_out != -1) {
    // Both types defined, make sure they are the same
    CHECK_EQ(dtype_in, dtype_out) << "Input and output weights must have same type";
    dtype = dtype_in;
  } else if (dtype_in != -1 || dtype_out != -1) {
    // One of the types defined, choose the one that was defined
    dtype = (dtype_in != -1) ? dtype_in : dtype_out;
  }
  if ((*in_type)[1] == -1) (*in_type)[1] = dtype;
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

/*! \brief name the struct Take instead of take
 * to avoid conflict with the take function in mshadow
 */
struct Take {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // K is the number of rows of in_data
  // i is the index of out_data
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const IType* idx, const int M, const int K) {
    int j = static_cast<int>(idx[i/M]);
    if (j <= 0) j = 0;
    else if (j >= K) j = K - 1;
    out_data[i] = in_data[j * M + i % M];
  }
};

template<typename xpu>
void EmbeddingOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(req[embedding::kOut], kWriteTo);
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs[embedding::kWeight].ndim(), 2U)
          << "Embedding layer expects its weight to be two-dimensional. "
          << inputs[embedding::kWeight].ndim()
          << " dimensional input is given instead";

  const TShape& ishape = inputs[embedding::kData].shape_;
  const TShape& oshape = outputs[embedding::kOut].shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      Tensor<xpu, 1, IType> data = inputs[embedding::kData].get_with_shape<xpu, 1, IType>(
        Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<xpu, 2, DType> wmat = inputs[embedding::kWeight].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[embedding::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
      Kernel<Take, xpu>::Launch(s, oshape.Size(), out.dptr_, wmat.dptr_,
        data.dptr_, wmat.shape_[1], wmat.shape_[0]);
    });
  });
}

// Returns integer log2(a) rounded up
inline int ilog2(unsigned int a) {
  int k = 1;
  while (a >>= 1) k++;
  return k;
}

/*! \brief cast to type and clip to range [0, K - 1]
 */
struct tcast_clip {
  template<typename OType, typename IType>
  MSHADOW_XINLINE static void Map(int i, OType* out_data, const IType* in_data,
                                  const OType K) {
    OType j = static_cast<OType>(in_data[i]);
    if (j <= 0) j = 0;
    else if (j >= K) j = K - 1;
    out_data[i] = j;
  }
};

template<typename xpu, typename IndexType, typename DType>
void AddTakeGradLargeBatchCaller(const OpContext& ctx, mshadow::Tensor<xpu, 2, DType> dst,
                                 const mshadow::Tensor<xpu, 1, IndexType>& index,
                                 const mshadow::Tensor<xpu, 2, DType> &src) {
  using namespace mxnet_op;
  using namespace mshadow::expr;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  // Calculate amount of temporary storage
  size_t sort_workspace_size = mxnet::op::SortByKeyWorkspaceSize<int, int, xpu>
    (index.shape_.Size());
  size_t addtake_workspace_size = mxnet::op::AddTakeGradLargeBatchWorkspaceSize<int, xpu>
    (index.shape_.Size());
  size_t temp_storage_size = std::max(sort_workspace_size, addtake_workspace_size);
  size_t workspace_size = 2*(index.shape_.Size()*sizeof(int)) + temp_storage_size;

  // Request temporary storage
  Tensor<xpu, 1, char> workspace =
    ctx.requested[embedding::kTempSpace].get_space_typed<xpu, 1, char>(
      Shape1(workspace_size), s);

  // Create tensors
  size_t pos = 0;
  Tensor<xpu, 1, int> sorted_data(reinterpret_cast<int*>(&workspace[pos]),
    Shape1(index.shape_.Size()), s);
  pos += index.shape_.Size()*sizeof(int);
  Tensor<xpu, 1, int> original_index(reinterpret_cast<int*>(&workspace[pos]),
    Shape1(index.shape_.Size()), s);
  pos += index.shape_.Size()*sizeof(int);
  Tensor<xpu, 1, char> temp_storage(&workspace[pos], Shape1(temp_storage_size), s);
  Kernel<tcast_clip, xpu>::Launch(s, index.shape_.Size(), sorted_data.dptr_, index.dptr_,
    static_cast<int>(dst.shape_[0]));
  original_index = range<int>(0, index.shape_.Size());
  int num_bits = ilog2((dst.shape_[0] - 1));
  mxnet::op::SortByKey(sorted_data, original_index, true, &temp_storage, 0, num_bits);
  mxnet::op::AddTakeGradLargeBatch(dst, sorted_data, original_index, src, &temp_storage);
}

template<typename xpu>
void EmbeddingOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req[embedding::kData], kNullOp)
          << "Embedding layer doesn't support calculate data gradient";
  CHECK_EQ(outputs[1].type_flag_, inputs[0].type_flag_);

  const TShape& ishape = inputs[1].shape_;
  const TShape& oshape = inputs[0].shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[1].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {
      Tensor < xpu, 1, IType > data = inputs[1].get_with_shape<xpu, 1, IType>(
        Shape1(ishape.ProdShape(0, ishape.ndim())), s);
      Tensor<xpu, 2, DType> grad_out = inputs[0].get_with_shape<xpu, 2, DType>(
      Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
      Tensor<xpu, 2, DType> grad_in = outputs[1].get<xpu, 2, DType>(s);


      if (req[embedding::kWeight] == kWriteTo || req[embedding::kWeight] == kAddTo) {
        if (req[embedding::kWeight] == kWriteTo) {
          grad_in = scalar<DType>(0.0f);
        }
        // shape_out_prod ~= the number of elements loaded in AddTakeGrad
        // shape_in_prod  ~= the number of elements stored in AddTakeGrad
        // When the number of elements processed is low, use AddTakeGrad.
        // The approximate cut-off value 16384 was found experimentally on Titan X Pascal
        uint64_t shape_in_prod =
          static_cast<uint64_t>(grad_in.shape_[0])*
          static_cast<uint64_t>(grad_in.shape_[1]);
        uint64_t shape_out_prod =
          static_cast<uint64_t>(grad_out.shape_[0])*
          static_cast<uint64_t>(grad_out.shape_[1]);
        if (shape_out_prod < (uint64_t)16384 && shape_in_prod < (uint64_t)16384) {
          AddTakeGrad(grad_in, data, grad_out);
        } else {
          AddTakeGradLargeBatchCaller(ctx, grad_in, data, grad_out);
        }
      } else {
        LOG(FATAL) << "wrong req";
      }
    });
  });
}

template<int req>
struct EmbeddingBackwardRsp {
  template<typename DType, typename IType>
  // each thread i is responsible for target gradient row ids in [segment_start, segment_end)
  MSHADOW_XINLINE static void Map(int i, const size_t width, IType* dst_idx, DType* dst_val,
                                  const IType* idx, const size_t num_idx, const DType* src,
                                  const size_t segment_len, const size_t num_rows) {
    auto req_type = req;
    size_t segment_start = i * segment_len;
    size_t segment_end = (i + 1) * segment_len;
    for (size_t y = 0; y < num_idx; y++) {
      size_t j = idx[y];
      if (j >= num_rows) j = num_rows - 1;
      if (j < segment_start || j >= segment_end) continue;
      dst_idx[j] = j;
      for (size_t k = 0; k < width; k++) {
        if (req_type == kWriteTo) req_type = kAddTo;
        KERNEL_ASSIGN(dst_val[j * width + k], req_type, src[y * width + k]);
      }
    }
  }
};

/*
 * for sparse embedding, the storage type for weight gradient is row_sparse.
 * we don't care about the storage type for data gradient, since it is not
 * differentiable.
 */
inline bool SparseEmbeddingBackwardStorageType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ((*in_attrs)[0], kDefaultStorage);
  CHECK_EQ((*in_attrs)[1], kDefaultStorage);
  (*out_attrs)[0] = kRowSparseStorage;
  (*out_attrs)[1] = kRowSparseStorage;
  return true;
}

template<typename xpu>
void SparseEmbeddingOpBackwardDnsDnsRsp(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  if (req[1] == kNullOp) return;
  // check storage types
  auto idx = inputs[1];  // idx shape (d1, d2 .. dk)
  auto grad = inputs[0];  // grad shape (d1, d2, .. dk, out_dim)
  auto output = outputs[1];  // weight shape (in_dim, out_dim)
  CHECK_EQ(idx.storage_type(), kDefaultStorage);
  CHECK_EQ(grad.storage_type(), kDefaultStorage);
  CHECK_EQ(output.dtype(), grad.dtype());
  CHECK_EQ(idx.dtype(), output.aux_type(rowsparse::kIdx)) << "Index type doesn't match";
  // CHECK_EQ(req[embedding::kData], kNullOp)
  //       << "Embedding layer doesn't support calculate data gradient" << req[embedding::kData];

  const TShape& ishape = idx.shape();
  const TShape& oshape = grad.shape();

  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(idx.dtype(), output.aux_type(rowsparse::kIdx))
           << "embedding input index and gradient row sparse type doesn't match!";
  // Alloc dense output
  unsigned int num_rows = output.shape()[0];
  output.CheckAndAlloc({mshadow::Shape1(num_rows)});
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    MSHADOW_INT_TYPE_SWITCH(idx.dtype(), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
        // input embedding indice, each idx in [0, input_dim)
        auto idx_data = idx.data().FlatTo1D<xpu, IType>(s);
        auto grad_data = grad.data().get_with_shape<xpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
        auto output_idx = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        auto output_val = output.data().FlatTo2D<xpu, DType>(s);
        int num_threads = omp_get_num_threads();
        size_t width = output.shape()[1];
        size_t segment_len = (num_rows + num_threads - 1) / num_threads;
        // fill indices with invalid row ids
        Kernel<mxnet_op::fill, xpu>::Launch(s, num_rows, output_idx.dptr_,
                                            static_cast<IType>(num_rows));
        // fill zeros if needed
        if (req_type == kWriteTo) {
          Kernel<mxnet_op::set_zero, xpu>::Launch(s, output_val.shape_.Size(), output_val.dptr_);
        }
        Kernel<EmbeddingBackwardRsp<req_type>, xpu>::Launch(s, num_threads, width,
                                                            output_idx.dptr_,
                                                            output_val.dptr_, idx_data.dptr_,
                                                            ishape.Size(), grad_data.dptr_,
                                                            segment_len, num_rows);
      });
    });
  });
}

// todo replace xpu with cpu
template<typename xpu>
void SparseEmbeddingOpBackwardEx(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  // CHECK_EQ(req[embedding::kData], kNullOp)
  //       << "Embedding layer doesn't support calculate data gradient" << req[0] << " " << req[1];
  // idx shape (d1, d2 .. dk)
  auto idx_stype = inputs[1].storage_type();
  // grad shape (d1, d2, .. dk, out_dim)
  auto grad_stype = inputs[0].storage_type();
  // weight shape (in_dim, out_dim)
  auto output_stype = outputs[1].storage_type();
  if (idx_stype == kDefaultStorage && grad_stype == kDefaultStorage &&
      output_stype == kRowSparseStorage) {
    SparseEmbeddingOpBackwardDnsDnsRsp<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

namespace take_ {  // to avoid name conflict
enum TakeOpInputs {kArr, kIdx};
enum TakeOpOutputs {kOut};
enum TakeOpResource {kTempSpace};
enum TakeOpMode {kRaise, kWrap, kClip};
}  // namespace take_

// TODO(somebody): behaviors specified by params
struct TakeParam: public dmlc::Parameter<TakeParam> {
  int axis;
  int mode;
  DMLC_DECLARE_PARAMETER(TakeParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_lower_bound(0)
    .set_default(0)
    .describe("The axis of input array to be taken.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("raise", take_::kRaise)
    .add_enum("wrap", take_::kWrap)
    .add_enum("clip", take_::kClip)
    .set_default(take_::kClip)
    .describe("Specify how out-of-bound indices bahave."
              " \"clip\" means clip to the range. So, if all indices mentioned are too large,"
              " they are replaced by the index that addresses the last element along an axis. "
              " \"wrap\" means to wrap around. "
              " \"raise\" means to raise an error. ");
  }
};

template<typename PType>
inline void TakeParamParser(nnvm::NodeAttrs *attrs) {
    PType param;
    param.Init(attrs->dict);
    if (param.axis != 0) {
        LOG(FATAL) << "Axis other than 0 currently not supported.";
    }
    if (param.mode != take_::kClip) {
        LOG(FATAL) << "Mode other than clip currently not supported.";
    }
}

inline bool TakeOpShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
    using namespace mshadow;
    const TShape &arrshape = (*in_attrs)[take_::kArr];
    const TShape &idxshape = (*in_attrs)[take_::kIdx];
    if (idxshape.ndim() == 0) return false;

    out_attrs->clear();

    TShape oshape(idxshape.ndim() + arrshape.ndim() - 1);
    for (size_t i = 0; i < idxshape.ndim(); ++i) {
        oshape[i] = idxshape[i];
    }
    for (size_t i = 0; i < arrshape.ndim() - 1; i++) {
        oshape[i + idxshape.ndim()] = arrshape[i + 1];
    }
    out_attrs->push_back(oshape);
    return true;
}

inline bool TakeOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[1], -1) << "Index type must be set for take operator";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void TakeOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[take_::kOut] == kNullOp) return;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TShape& idxshape = inputs[take_::kIdx].shape_;
  const TShape& arrshape = inputs[take_::kArr].shape_;
  const TShape& oshape = outputs[take_::kOut].shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
      Kernel<Take, xpu>::Launch(s, oshape.Size(),
                                outputs[take_::kOut].dptr<DType>(),
                                inputs[take_::kArr].dptr<DType>(),
                                inputs[take_::kIdx].dptr<IType>(),
                                oshape.Size()/idxshape.Size(), arrshape[0]);
    });
  });
}

template<typename xpu>
void TakeOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req[take_::kIdx], kNullOp)
    << "take layer doesn't support gradient into index";

  // inputs are specified in the .cc file, which are the gradients from
  // the upper layer and the input index
  // outputs are the gradients of inputs in the feed-forward pass
  const TShape& idxshape = inputs[1].shape_;
  const TShape& arrshape = outputs[0].shape_;
  const TShape& oshape = inputs[0].shape_;

  int idxndim = idxshape.ndim();

  // grad_out is the gradient of the outputs in the feed-forward
  // grad_in is the gradient of the inputs in the feed-forward
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType, {  // index data type
      Tensor<xpu, 1, IType> idx = inputs[1].get_with_shape<xpu, 1, IType>(
          Shape1(idxshape.ProdShape(0, idxndim)), s);
      Tensor<xpu, 2, DType> grad_out = inputs[0].get_with_shape<xpu, 2, DType>(
          Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);
      Tensor<xpu, 2, DType> grad_in = outputs[0].get_with_shape<xpu, 2, DType>(
          Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);

      if (req[take_::kArr] == kWriteTo || req[take_::kArr] == kAddTo) {
        if (req[take_::kArr] == kWriteTo) {
          grad_in = scalar<DType>(0.0f);
        }
        // shape_out_prod ~= the number of elements loaded in AddTakeGrad
        // shape_in_prod  ~= the number of elements stored in AddTakeGrad
        // When the number of elements processed is low, use AddTakeGrad.
        // The approximate cut-off value 16384 was found experimentally on Titan X Pascal
        uint64_t shape_in_prod =
          static_cast<uint64_t>(grad_in.shape_[0])*
          static_cast<uint64_t>(grad_in.shape_[1]);
        uint64_t shape_out_prod =
          static_cast<uint64_t>(grad_out.shape_[0])*
          static_cast<uint64_t>(grad_out.shape_[1]);
        if (shape_out_prod < (uint64_t)16384 && shape_in_prod < (uint64_t)16384) {
          AddTakeGrad(grad_in, idx, grad_out);
        } else {
          AddTakeGradLargeBatchCaller(ctx, grad_in, idx, grad_out);
        }
      } else {
        LOG(FATAL) << "wrong req";
      }
    });
  });
}

inline bool BatchTakeOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  LOG(INFO) << "batch_take is deprecated. Please use pick instead.";
  CHECK_EQ(in_attrs->size(), 2U) << "BatchTake op requires two inputs";
  if ((*in_attrs)[1].ndim() != 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[1]);
  } else if ((*out_attrs)[0].ndim() != 0) {
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, (*out_attrs)[0]);
  }
  if ((*in_attrs)[0].ndim() == 0) return false;
  CHECK_GE((*in_attrs)[0].ndim(), 2U) << "Data array must have at least 2 dimensional";
  if ((*out_attrs)[0].ndim() == 0) return false;
  CHECK_EQ((*in_attrs)[0].Size()/(*in_attrs)[0][(*in_attrs)[0].ndim()-1],
           (*out_attrs)[0].Size())
    << "Index array's size must be the same as data array's size excluding the first dimension";
  return true;
}

inline bool BatchTakeOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  if ((*in_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  } else if ((*out_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  }
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kInt32);
  return true;
}

/*! \brief take scalar value from 2d data array */
template<int req>
struct batch_take {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* a,
                                  const int *idx, int M) {
    int j = idx[i];
    if (j < 0) j = 0;
    else if (j >= M) j = M-1;
    KERNEL_ASSIGN(out[i], req, a[i*M+j]);
  }
};

template<typename xpu>
void BatchTakeOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<batch_take<req_type>, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                                inputs[0].dptr<DType>(), inputs[1].dptr<int>(),
                                                inputs[0].Size()/inputs[0].shape_[0]);
    });
  });
}

/*!
 * \brief The parameters of the one_hot operator.
 */
struct OneHotParam : public dmlc::Parameter<OneHotParam> {
  int depth;
  double on_value;
  double off_value;
  int axis;
  int dtype;
  DMLC_DECLARE_PARAMETER(OneHotParam) {
    DMLC_DECLARE_FIELD(depth)
      .describe("Depth of the one hot dimension.");
    DMLC_DECLARE_FIELD(on_value)
      .set_default(1.0f)
      .describe("The value assigned to the locations represented by indices.");
    DMLC_DECLARE_FIELD(off_value)
      .set_default(0.0f)
      .describe("The value assigned to the locations not represented by indices.");
    DMLC_DECLARE_FIELD(dtype)
      .set_default(mshadow::kFloat32)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("uint8", mshadow::kUint8)
      .add_enum("int32", mshadow::kInt32)
      .describe("DType of the output");
  }
};

inline void GetOneHotParams(const OneHotParam& param, int* depth, double* on_value,
                            double* off_value, int* dtype) {
  *depth = param.depth;
  CHECK_GE(*depth, 0) << "Dimension size, depth, must be a non-negative integer";
  *on_value = param.on_value;
  *off_value = param.off_value;
  *dtype = param.dtype;
}

inline bool OneHotOpShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const OneHotParam& param = nnvm::get<OneHotParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  // The shape of indices
  const TShape& ishape = (*in_attrs)[0];

  int depth = 0;
  double on_value = 1.0;
  double off_value = 0.0;
  int dtype = mshadow::kFloat32;
  GetOneHotParams(param, &depth, &on_value, &off_value, &dtype);

  TShape oshape(ishape.ndim() + 1);
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    oshape[i] = ishape[i];
  }
  oshape[oshape.ndim()-1] = depth;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return true;
}

inline bool OneHotOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[0], -1) << "Index type must be set for one_hot operator";
  int depth = 0;
  double on_value = 1.0;
  double off_value = 0.0;
  int dtype = -1;
  const OneHotParam& param = nnvm::get<OneHotParam>(attrs.parsed);
  GetOneHotParams(param, &depth, &on_value, &off_value, &dtype);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);  // assign output type

  return true;
}

template<int req>
struct one_hot {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const IType* indices,
                                  int depth, DType on_value) {
    int offset = i * depth;
    int j = static_cast<int>(indices[i]);
    if (j >= 0 && j < depth) {
      KERNEL_ASSIGN(out[offset+j], req, on_value);
    }
  }
};

template<typename xpu>
void OneHotOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  // The following line is needed to guard the situation when
  // an output array is empty on GPU. In that case, out.dptr() = 0x0
  if (outputs[0].Size() == 0) return;
  int depth = 0;
  double on_value = 1.0;
  double off_value = 0.0;
  int dtype = mshadow::kFloat32;
  const OneHotParam& param = nnvm::get<OneHotParam>(attrs.parsed);
  GetOneHotParams(param, &depth, &on_value, &off_value, &dtype);
  using namespace mxnet_op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {  // output data type switch
    mshadow::Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], static_cast<DType>(off_value));
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {  // request type switch
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {  // indices data type switch
        Kernel<one_hot<req_type>, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
                                               inputs[0].dptr<IType>(), depth,
                                               static_cast<DType>(on_value));
      });
    });
  });
}

/*!
 * \brief sparse retain namespace
 */
namespace sr {
enum SparseRetainOpInputs {kArr, kIdx};
enum SparseRetainOpOutputs {kOut};
}  // namespace sr

inline bool SparseRetainOpShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape> *in_attrs,
                                std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U)
    << "sparse_retain operator takes 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  TShape tshape((*in_attrs)[sr::kArr]);
  shape_assign(&tshape, (*out_attrs)[sr::kOut]);
  SHAPE_ASSIGN_CHECK(*in_attrs, sr::kArr, tshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, sr::kOut, tshape);
  return true;
}

inline bool SparseRetainOpType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE((*in_attrs)[sr::kIdx], -1) << "Index type must be set for sparse_retain operator";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[sr::kArr]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[sr::kOut]);
  return (*in_attrs)[0] != -1;
}

inline bool SparseRetainForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (kRowSparseStorage == in_attrs->at(sr::kArr)) {
    out_attrs->at(sr::kOut) = kRowSparseStorage;
  }
  return true;
}

inline bool SparseRetainBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  out_attrs->at(sr::kArr) = kRowSparseStorage;
  out_attrs->at(sr::kIdx) = kDefaultStorage;
  return true;
}

struct SparseRetainRspForward {
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, RType* out_idx,
                                  const DType* in_data, const RType* in_idx,
                                  const IType* idx, const size_t nnr,
                                  const size_t num_cols) {
    const RType irow = idx[i];
    int j = -1, left = 0, right = nnr - 1;
    while (left <= right) {
      int m = left + (right - left) / 2;
      const auto in_idx_m = in_idx[m];
      if (in_idx_m == irow) {
        j = m;
        break;
      } else if (in_idx_m < irow) {
        left = m + 1;
      } else {
        right = m - 1;
      }
    }
    out_idx[i] = idx[i];
    if (j >= 0) {
      const size_t in_offset = j * num_cols;
      const size_t out_offset = i * num_cols;
      for (size_t k = 0; k < num_cols; ++k) {
        out_data[out_offset+k] = in_data[in_offset+k];
      }
    }
  }
};

template<typename xpu>
void SparseRetainOpForwardEx(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[sr::kOut], kWriteTo) << "sparse_retain only supports req=\'write\'";

  CHECK_EQ(inputs[sr::kArr].storage_type(), kRowSparseStorage)
    << "sparse_retain operator only takes row sparse NDArray as input";
  CHECK_EQ(inputs[sr::kIdx].storage_type(), kDefaultStorage)
    << "sparse_retain operator only takes default NDArray as its index array";
  CHECK_EQ(outputs[sr::kOut].storage_type(), kRowSparseStorage)
    << "sparse_retain operator only outputs row sparse NDArray";

  const NDArray& input_nd = inputs[sr::kArr];
  const TBlob idx_data = inputs[sr::kIdx].data();

  if (req[sr::kOut] == kNullOp
      || !input_nd.storage_initialized()
      || idx_data.Size() == 0U) return;

  const TBlob input_data = input_nd.data();
  if (input_data.shape_[0] == 0) return;
  const TBlob input_idx = input_nd.aux_data(rowsparse::kIdx);

  NDArray output_nd = outputs[sr::kOut];
  output_nd.CheckAndAlloc({mshadow::Shape1(idx_data.Size())});
  TBlob output_data = output_nd.data();
  TBlob output_idx = output_nd.aux_data(rowsparse::kIdx);

  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(output_data.type_flag_, DType, {  // output data type
    MSHADOW_INT_TYPE_SWITCH(output_idx.type_flag_, RType, {  // row index data type
      MSHADOW_TYPE_SWITCH(idx_data.type_flag_, IType, {  // index array data type
        Kernel<set_zero, xpu>::Launch(s, output_data.Size(), output_data.dptr<DType>());
        Kernel<SparseRetainRspForward, xpu>::Launch(s, idx_data.Size(), output_data.dptr<DType>(),
            output_idx.dptr<RType>(), input_data.dptr<DType>(), input_idx.dptr<RType>(),
            idx_data.dptr<IType>(), input_data.shape_[0], input_data.shape_[1]);
      });
    });
  });
}

template<int req>
struct SparseRetainRspBackward {
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, RType* in_grad_idx,
                                  const DType* out_grad, const IType* idx,
                                  const size_t num_cols) {
    const RType irow = idx[i];
    in_grad_idx[i] = irow;
    const size_t out_offset = irow * num_cols;
    const size_t in_offset = i * num_cols;
    for (size_t j = 0; j < num_cols; ++j) {
      KERNEL_ASSIGN(in_grad[in_offset+j], req, out_grad[out_offset+j]);
    }
  }
};

template<typename xpu>
void SparseRetainOpBackwardEx(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_NE(req[sr::kArr], kWriteInplace);
  CHECK_EQ(req[sr::kIdx], kNullOp)
    << "sparse_retain does not support calculating gradients of indices";

  CHECK_EQ(inputs[sr::kOut].storage_type(), kDefaultStorage)
    << "sparse_retain backward only takes default NDArray as ograd";
  CHECK_EQ(inputs[sr::kIdx].storage_type(), kDefaultStorage)
    << "sparse_retain backward only takes default NDArray as its index array";
  CHECK_EQ(outputs[sr::kArr].storage_type(), kRowSparseStorage)
    << "sparse_retain backward only outputs row sparse NDArray as grad of input";

  const TBlob out_grad_data = inputs[sr::kOut].data();
  const TBlob idx_data = inputs[sr::kIdx].data();

  NDArray in_grad_nd = outputs[sr::kArr];
  in_grad_nd.CheckAndAlloc({mshadow::Shape1(idx_data.Size())});
  TBlob in_grad_data = in_grad_nd.data();
  TBlob in_grad_idx = in_grad_nd.aux_data(rowsparse::kIdx);

  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out_grad_data.type_flag_, DType, {  // output data type
    MSHADOW_INT_TYPE_SWITCH(in_grad_idx.type_flag_, RType, {  // row index data type
      MSHADOW_TYPE_SWITCH(idx_data.type_flag_, IType, {  // index array data type
        MXNET_ASSIGN_REQ_SWITCH(req[sr::kArr], req_type, {
          Kernel<SparseRetainRspBackward<req_type>, xpu>::Launch(
              s, in_grad_idx.Size(), in_grad_data.dptr<DType>(), in_grad_idx.dptr<RType>(),
              out_grad_data.dptr<DType>(), idx_data.dptr<IType>(), out_grad_data.shape_[1]);
        });
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./indexing_op-inl.cuh"
#endif
#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_H_

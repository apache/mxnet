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

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./indexing_op-inl.cuh"
#endif
#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_H_

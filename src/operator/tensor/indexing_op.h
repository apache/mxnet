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
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

namespace indexing {
enum IndexingOpInputs {kIdx, kData};
enum IndexingOpOutputs {kOut};
enum IndexingOpResource {kTempSpace};
enum IndexingOpMode {kRaise, kWrap, kClip};
}  // namespace indexing

struct EmbeddingParam: public dmlc::Parameter<EmbeddingParam> {
  int input_dim;
  int output_dim;
  DMLC_DECLARE_PARAMETER(EmbeddingParam) {
    DMLC_DECLARE_FIELD(input_dim).set_lower_bound(1)
    .describe("vocabulary size of the input indices.");
    DMLC_DECLARE_FIELD(output_dim).set_lower_bound(1)
    .describe("dimension of the embedding vectors.");
  }
};

inline bool EmbeddingOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  using namespace mshadow;
  const TShape &dshape = (*in_attrs)[indexing::kIdx];
  if (dshape.ndim() ==  0) return false;
  const EmbeddingParam& param = nnvm::get<EmbeddingParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*in_attrs, indexing::kData, Shape2(param.input_dim,
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
  CHECK_GE(in_type->size(), 1);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                     << "Expected " << dtype << " v.s. given "
                                     << (*in_type)[i];
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

// TODO(somebody): behaviors specified by params
struct TakeParam: public dmlc::Parameter<TakeParam> {
  int axis;
  int mode;
  DMLC_DECLARE_PARAMETER(TakeParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_lower_bound(0)
    .set_default(0)
    .describe("the axis of data tensor to be taken.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("raise", indexing::kRaise)
    .add_enum("wrap", indexing::kWrap)
    .add_enum("clip", indexing::kClip)
    .set_default(indexing::kRaise)
    .describe("specify how out-of-bound indices bahave.");
  }
};

inline bool TakeOpShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
    using namespace mshadow;
    const TShape &arrshape = (*in_attrs)[indexing::kData];
    const TShape &idxshape = (*in_attrs)[indexing::kIdx];
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
                       std::vector<int> *in_type,
                       std::vector<int> *out_type) {
  // using single dtype ("float32") for safety reason
  CHECK_GE(in_type->size(), 2);
  int dtype = (*in_type)[1];
  CHECK_NE(dtype, -1) << "idx must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                     << "Expected " << dtype << " v.s. given "
                                     << (*in_type)[i];
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

template<typename xpu>
void TakeOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[indexing::kOut], kWriteTo);
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_GE(inputs[indexing::kData].ndim(), 2)
        << "Indexing layer expects its array's size to be at least 2. "
        << inputs[indexing::kData].ndim()
        << " dimensional input is given instead";

    const TShape& idxshape = inputs[indexing::kIdx].shape_;
    const TShape& arrshape = inputs[indexing::kData].shape_;
    const TShape& oshape = outputs[indexing::kOut].shape_;

    int idxndim = idxshape.ndim();

    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, 1, DType> idx = inputs[indexing::kIdx].get_with_shape<xpu, 1, DType>(
            Shape1(idxshape.ProdShape(0, idxndim)), s);
        Tensor<xpu, 2, DType> data = inputs[indexing::kData].get_with_shape<xpu, 2, DType>(
            Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);
        Tensor<xpu, 2, DType> out = outputs[indexing::kOut].get_with_shape<xpu, 2, DType>(
            Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);
        out = take(idx, data);
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
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 2);
    CHECK_EQ(req[indexing::kIdx], kNullOp)
        << "Indexing op doesn't support gradient into index";

    // inputs are specified in the .cc file, which are the gradients from
    // the upper layer and the input index
    // outputs are the gradients of inputs in the feed-forward pass
    const TShape& idxshape = inputs[1].shape_;
    const TShape& arrshape = outputs[1].shape_;
    const TShape& oshape = inputs[0].shape_;

    int idxndim = idxshape.ndim();

    // grad_out is the gradient of the outputs in the feed-forward
    // grad_in is the gradient of the inputs in the feed-forward
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Tensor<xpu, 1, DType> idx = inputs[1].get_with_shape<xpu, 1, DType>(
            Shape1(idxshape.ProdShape(0, idxndim)), s);
        Tensor<xpu, 2, DType> grad_out = inputs[0].get_with_shape<xpu, 2, DType>(
            Shape2(oshape.ProdShape(0, idxndim), oshape.ProdShape(idxndim, oshape.ndim())), s);
        Tensor<xpu, 2, DType> grad_in = outputs[1].get_with_shape<xpu, 2, DType>(
            Shape2(arrshape[0], arrshape.ProdShape(1, arrshape.ndim())), s);

        if (req[indexing::kData] == kWriteTo || req[indexing::kData] == kAddTo) {
            if (req[indexing::kData] == kWriteTo) {
                grad_in = scalar<DType>(0.0f);
            }
            if ((grad_out.shape_[0] < grad_out.shape_[1]) && (grad_out.shape_[0] < 512)) {
                AddTakeGrad(grad_in, idx, grad_out);
            } else {
                Tensor<xpu, 2, int> workspace =
                    ctx.requested[indexing::kTempSpace].get_space_typed<xpu, 2, int>(
                        mshadow::Shape2(2, idx.shape_.Size()), s);
                Tensor<xpu, 1, int> sorted_idx = workspace[0];
                Tensor<xpu, 1, int> original_idx = workspace[1];
                sorted_idx = tcast<int>(idx);
                original_idx = range<int>(0, idx.shape_.Size());
                SortByKey(sorted_idx, original_idx, true);
                AddTakeGradLargeBatch(grad_in, sorted_idx, original_idx, grad_out);
            }
        } else {
            LOG(FATAL) << "wrong req";
        }
    });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_H_

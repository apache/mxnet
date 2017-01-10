/*!
 * Copyright (c) 2016 by Contributors
 * \file indexing_op.h
 * \brief
 * \author Bing Xu, Siyi Li
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

namespace embedding {
enum EmbeddingOpInputs {kData, kWeight};
enum EmbeddingOpOutputs {kOut};
enum EmbeddingOpResource {kTempSpace};
}  // namespace embedding

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

template<typename xpu>
void EmbeddingOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(req[embedding::kOut], kWriteTo);
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(inputs[embedding::kWeight].ndim(), 2)
          << "Embedding layer expects its weight to be two-dimensional. "
          << inputs[embedding::kWeight].ndim()
          << " dimensional input is given instead";

  const TShape& ishape = inputs[embedding::kData].shape_;
  const TShape& oshape = outputs[embedding::kOut].shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> data = inputs[embedding::kData].get_with_shape<xpu, 1, DType>(
      Shape1(ishape.ProdShape(0, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = inputs[embedding::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[embedding::kOut].get_with_shape<xpu, 2, DType>(
      Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
    out = take(data, wmat);
  });
}

template<typename xpu>
void EmbeddingOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 2);
  CHECK_EQ(req[embedding::kData], kNullOp)
          << "Embedding layer doesn't support calculate data gradient";

  const TShape& ishape = inputs[1].shape_;
  const TShape& oshape = inputs[0].shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor < xpu, 1, DType > data = inputs[1].get_with_shape<xpu, 1, DType>(
      Shape1(ishape.ProdShape(0, ishape.ndim())), s);
    Tensor<xpu, 2, DType> grad_out = inputs[0].get_with_shape<xpu, 2, DType>(
    Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
    Tensor<xpu, 2, DType> grad_in = outputs[1].get<xpu, 2, DType>(s);


    if (req[embedding::kWeight] == kWriteTo || req[embedding::kWeight] == kAddTo) {
      if (req[embedding::kWeight] == kWriteTo) {
        grad_in = scalar<DType>(0.0f);
      }
      if ((grad_out.shape_[0] < grad_out.shape_[1]) && (grad_out.shape_[0] < 512)) {
        AddTakeGrad(grad_in, data, grad_out);
      } else {
        Tensor<xpu, 2, int> workspace =
          ctx.requested[embedding::kTempSpace].get_space_typed<xpu, 2, int>(
            mshadow::Shape2(2, data.shape_.Size()), s);
        Tensor<xpu, 1, int> sorted_data = workspace[0];
        Tensor<xpu, 1, int> original_index = workspace[1];
        sorted_data = tcast<int>(data);
        original_index = range<int>(0, data.shape_.Size());
        SortByKey(sorted_data, original_index, true);
        AddTakeGradLargeBatch(grad_in, sorted_data, original_index, grad_out);
      }
    } else {
      LOG(FATAL) << "wrong req";
    }
  });
}


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_H_

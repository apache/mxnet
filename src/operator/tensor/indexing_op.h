/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_INDEXING_OP_H_
#define MXNET_OPERATOR_INDEXING_OP_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

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

template<typename xpu, typename DType>
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
  Tensor<xpu, 1, DType> data = inputs[embedding::kData].get_with_shape<xpu, 1, DType>(
          Shape1(ishape.ProdShape(0, ishape.ndim())), s);
  Tensor<xpu, 2, DType> wmat = inputs[embedding::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> out = outputs[embedding::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
  out = take(data, wmat);
}

template<typename xpu, typename DType>
void EmbeddingOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {

}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_INDEXING_OP_H_

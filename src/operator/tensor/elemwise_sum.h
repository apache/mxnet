/*!
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.h
 * \brief elementwise sum
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_

#include <dmlc/logging.h>
#include <cstring>
#include <vector>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
void ElementWiseSumCompute_(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  size_t size = in_data.size();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 1, DType> out = out_data[0].FlatTo1D<xpu, DType>(s);
  switch (size) {
    case 2: {
      Tensor<xpu, 1, DType> in_0 = in_data[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_1 = in_data[1].FlatTo1D<xpu, DType>(s);
      Assign(out, req[0], in_0 + in_1);
      break;
    }
    case 3: {
      Tensor<xpu, 1, DType> in_0 = in_data[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_1 = in_data[1].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_2 = in_data[2].FlatTo1D<xpu, DType>(s);
      Assign(out, req[0], in_0 + in_1 + in_2);
      break;
    }
    case 4: {
      Tensor<xpu, 1, DType> in_0 = in_data[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_1 = in_data[1].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_2 = in_data[2].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> in_3 = in_data[3].FlatTo1D<xpu, DType>(s);
      Assign(out, req[0], in_0 + in_1 + in_2 + in_3);
      break;
    }
    default: {
      Tensor<xpu, 1, DType> in_0 = in_data[0].FlatTo1D<xpu, DType>(s);
      Assign(out, req[0], F<mshadow_op::identity>(in_0));
      for (size_t i = 1; i < size; ++i) {
        out += in_data[i].FlatTo1D<xpu, DType>(s);
      }
      break;
    }
  }
}

template<typename xpu>
void ElementWiseSumCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      ElementWiseSumCompute_<xpu, DType>(attrs, ctx, inputs, req, outputs);
    });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_

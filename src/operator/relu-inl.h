/*!
 * Copyright (c) 2017 by Contributors
 * \file relu-inl.h
 * \brief
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_RELU_INL_H_
#define MXNET_OPERATOR_RELU_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mxnet_op.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

struct relu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = x > DType(0) ? x : DType(0);
  }
};

template<typename xpu>
void ReluCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<relu, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
  });
}

struct relu_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = x > 0.0 ? 1.0 : 0.0;
  }
};

template<typename xpu>
void ReluBackward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<relu_backward, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_

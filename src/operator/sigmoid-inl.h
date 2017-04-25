/*!
 * Copyright (c) 2017 by Contributors
 * \file sigmoid-inl.h
 * \brief
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_SIGMOID_INL_H_
#define MXNET_OPERATOR_SIGMOID_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mxnet_op.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    out[i] = DType(DType(1.0f) / (DType(1.0f) + expf(-xs[i])));
  }
};

template<typename xpu>
void SigmoidCompute(const nnvm::NodeAttrs& attrs,
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
    Kernel<sigmoid, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
  });
}

struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *xs) {
    DType x = xs[i];
    out[i] = DType(x * (DType(1.0f) - x));
  }
};

template<typename xpu>
void SigmoidBackward(const nnvm::NodeAttrs& attrs,
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
    Kernel<sigmoid_grad, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGMOID_INL_H_

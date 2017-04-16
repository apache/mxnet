/*!
 *  Copyright (c) 2017 by Contributors
 * \file dequantize-inl.h
 * \brief Implementation of dequantize operation
 */
#ifndef MXNET_OPERATOR_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_DEQUANTIZE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct DequantizeParam : public dmlc::Parameter<DequantizeParam> {
  float min_range;
  float max_range;
  int out_type;
  DMLC_DECLARE_PARAMETER(DequantizeParam) {
    DMLC_DECLARE_FIELD(min_range)
    .describe("The minimum scalar value possibly produced for the input");
    DMLC_DECLARE_FIELD(max_range)
    .describe("The maximum scalar value possibly produced for the input");
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("float32", mshadow::kFloat32)
    .describe("Output data type.");
  }
};

template<typename xpu>
void DequantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      double half_range = !std::is_signed<SrcDType>::value
        ? 0.0f
        : ((static_cast<double>(std::numeric_limits<SrcDType>::max()) -
            static_cast<double>(std::numeric_limits<SrcDType>::min()) + 1) / 2.0);
      double scale =
        (param.max_range - param.min_range) /
        (static_cast<double>(std::numeric_limits<SrcDType>::max()) -
         static_cast<double>(std::numeric_limits<SrcDType>::min()));

      Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0],
        (tcast<DstDType>(data) + scalar<DstDType>(half_range))
          * scalar<DstDType>(scale) + scalar<DstDType>(param.min_range));
    });
  });
}

inline bool DequantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ((*in_attrs)[0], mshadow::kUint8)
    << "`dequantize` only supports uint8 input for now";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.out_type);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DEQUANTIZE_INL_H_

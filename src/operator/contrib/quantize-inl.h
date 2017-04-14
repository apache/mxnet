/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZE_H_
#define MXNET_OPERATOR_QUANTIZE_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  float min_range;
  float max_range;
  int   out_type;
  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(min_range)
    .describe("The minimum scalar value possibly produced for the input");
    DMLC_DECLARE_FIELD(max_range)
    .describe("The maximum scalar value possibly produced for the input");
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("uint8", mshadow::kUint8)
    .describe("Output data type.");
  }
};

template<typename xpu>
void QuantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    double scale =
      (static_cast<double>(std::numeric_limits<DstDType>::max()) -
       static_cast<double>(std::numeric_limits<DstDType>::min())) /
      (param.max_range - param.min_range);

    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0],
        tcast<DstDType>((data - scalar<SrcDType>(param.min_range))
          * scalar<SrcDType>(scale)
          + scalar<SrcDType>(0.5)));
    });
  });
}

inline bool QuantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ((*in_attrs)[0] = mshadow::kFloat32)
    << "`quantize` only supports float32 input for now";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.out_type);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZE_H_

/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZE_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  int   out_type;
  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(out_type)
    .add_enum("uint8", mshadow::kUint8)
    .set_default(mshadow::kUint8)
    .describe("Output data type.");
  }
};

struct quantize {
  template<typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, const SrcDType *in,
                                  float min_range, float max_range, float scale) {
    out[i] = static_cast<DstDType>((in[i] - min_range) * scale + 0.5);
  }
};

template<typename xpu>
void QuantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
    float min_range = inputs[1].dptr<float>()[0];
    float max_range = inputs[2].dptr<float>()[0];
    float scale =
      (static_cast<double>(std::numeric_limits<DstDType>::max()) -
       static_cast<double>(std::numeric_limits<DstDType>::min())) /
      (max_range - min_range);

    Kernel<quantize, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DstDType>(),
      inputs[0].dptr<SrcDType>(), min_range, max_range, scale);
    outputs[1].dptr<float>()[0] = min_range;
    outputs[2].dptr<float>()[0] = max_range;
  });
  });
}

inline bool QuantizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);

  CHECK(!shape_is_none(in_attrs->at(0)));
  for (size_t i = 1; i < 3; ++i) {
    CHECK(shape_is_scalar(in_attrs->at(i)));
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape{1});
  return true;
}

inline bool QuantizeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32)
    << "`quantize` only supports float32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "the second input of `quantize` should be a tensor with type of float";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
    << "the third input of `quantize` should be a tensor with type of float";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZE_INL_H_

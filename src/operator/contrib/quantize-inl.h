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
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range,
                                  float *omax_range, const SrcDType *in,
                                  const float *imin_range, const float *imax_range,
                                  double min_limit, double max_limit) {
    float scale = (max_limit - min_limit) / (*imax_range - *imin_range);
    out[i] = static_cast<DstDType>((in[i] - *imin_range) * scale + 0.5);
    *omin_range = *imin_range;
    *omax_range = *imax_range;
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
  // for now, only supports quantize from uint8 to float
  // TODO(ziheng) consider add MSHADOW_INTEGER_TYPE_SWITCH
  typedef uint8_t DstDType;
  typedef float SrcDType;
  Kernel<quantize, xpu>::Launch(s, outputs[0].Size(),
    outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
    inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
    std::numeric_limits<DstDType>::min(), std::numeric_limits<DstDType>::max());
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

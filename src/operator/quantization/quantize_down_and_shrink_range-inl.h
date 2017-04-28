/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZE_DOWN_AND_SHRINK_RANGE_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZE_DOWN_AND_SHRINK_RANGE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct QuantizeDownAndShrinkRangeParam : public dmlc::Parameter<QuantizeDownAndShrinkRangeParam> {
  DMLC_DECLARE_PARAMETER(QuantizeDownAndShrinkRangeParam) {
  }
};

inline bool QuantizeDownAndShrinkRangeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const QuantizeDownAndShrinkRangeParam& param =
    nnvm::get<QuantizeDownAndShrinkRangeParam>(attrs.parsed);
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

inline bool QuantizeDownAndShrinkRangeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  const QuantizeDownAndShrinkRangeParam& param =
    nnvm::get<QuantizeDownAndShrinkRangeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  CHECK_EQ((*in_attrs)[0], mshadow::kInt32)
    << "`quantize_down_and_shrink_range` only supports int32 input for now";
  CHECK_EQ((*in_attrs)[1], mshadow::kFloat32)
    << "the second input of `quantize_down_and_shrink_range` "
    << "should be a tensor with type of float";
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32)
      << "the third input of `quantize_down_and_shrink_range` "
      << "should be a tensor with type of float";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZE_DOWN_AND_SHRINK_RANGE_INL_H_

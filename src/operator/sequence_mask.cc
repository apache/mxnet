/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_mask.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_mask-inl.h"

namespace mshadow {

template <typename DType>
inline void SequenceMask(const Tensor<cpu, 3, DType> &dst,
                        const Tensor<cpu, 1, DType> label) {
  for (index_t b = 0; b < dst.size(1); ++b)
    for (index_t s = label[b]; s < dst.size(0); ++s)
      for (index_t r = 0; r < dst.size(2); ++r)
      dst[s][b][r] = 0.;
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SequenceMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SequenceMaskOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceMaskProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceMaskParam);

MXNET_REGISTER_OP_PROPERTY(SequenceMask, SequenceMaskProp)
.describe("Get the last element of a sequence.")
.add_argument("data", "Symbol", "Input data of the form (seq len, other dims).")
.add_argument("sequence_length", "Symbol", "vector of sequence lengths.")
.add_arguments(SequenceMaskParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet

/*!
 * Copyright (c) 2015 by Contributors
 * \file svm_output.cc
 * \brief
 * \author Jonas Amaro
*/
#include "./svm_output-inl.h"
#include "./mshadow_op.h"

namespace mshadow {
  template<typename DType>
  inline void L1_SVM(const DType & margin,
                     const DType & reg_coef,
                     Tensor<cpu, 2, DType> dst,
                     const Tensor<cpu, 1, DType> & label,
                     const Tensor<cpu, 2, DType> & src) {
    for (index_t y = 0; y < dst.size(0); y++) {
      const index_t k = static_cast<int>(label[y]);
      for (index_t x = 0; x < dst.size(1); x++) {
        if (x == k) {
          dst[y][k] = -DType(margin > src[y][k]) * reg_coef;
        } else {
          dst[y][x] = DType(margin > -src[y][x]) * reg_coef;
        }
      }
    }
  }


  template<typename DType>
  inline void L2_SVM(const DType & margin,
                     const DType & reg_coef,
                     Tensor<cpu, 2, DType> dst,
                     const Tensor<cpu, 1, DType> & label,
                     const Tensor<cpu, 2, DType> & src) {
    for (index_t y = 0; y < dst.size(0); y++) {
      const index_t k = static_cast<int>(label[y]);
      for (index_t x = 0; x < dst.size(1); x++) {
        if (x == k) {
          dst[y][k] = margin > src[y][k] ?  2*(margin - src[y][k]) : DType(0.0f);
          dst[y][k] *= -reg_coef;
        } else {
          dst[y][x] = margin > -src[y][x] ? (-2)*(margin + src[y][x]) : DType(0.0f);
          dst[y][x] *= -reg_coef;
        }
      }
    }
  }
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SVMOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SVMOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SVMOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SVMOutputParam);

MXNET_REGISTER_OP_PROPERTY(SVMOutput, SVMOutputProp)
.describe("Support Vector Machine based transformation on input, backprop L2-SVM")
.add_argument("data", "Symbol", "Input data to svm.")
.add_argument("label", "Symbol", "Label data.")
.add_arguments(SVMOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


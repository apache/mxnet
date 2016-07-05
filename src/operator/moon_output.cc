/*!
 * Copyright (c) 2016 by Contributors
 * \file moon_output.cc
 * \brief
 * \author Wei Wu
*/
#include <vector>
#include <math.h>
#include "./moon_output-inl.h"

namespace mshadow {
template<typename Dtype>
inline void MoonBackward(const Tensor<cpu, 2, Dtype> &grad_data,
	const Tensor<cpu, 2, Dtype> &out_data,
	const Tensor<cpu, 2, Dtype> &input_label,
	const std::vector<float> &src_dist) {
	const Dtype *data = out_data.dptr_;
	const Dtype *label = input_label.dptr_;
	Dtype *grad = grad_data.dptr_;
	Dtype weight = 0.0;
	for (index_t n = 0; n < out_data.size(0); ++n) {
		for (index_t c = 0; c < out_data.size(1); ++c) {
			const int index = c * out_data.size(0) + n;
			if (1 == int(label[index]) && src_dist[c] > 0.5) {
				weight = (1 - src_dist[c]) / src_dist[c];
			}
			else if (-1 == int(label[index]) && src_dist[c] < 0.5) {
				weight = src_dist[c] / (1 - src_dist[c]);
			}
			else {
				weight = 1.0;
			}
			grad[index] = 2.0 * (data[index] - label[index]) * weight;
		}
	}
}
} // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MoonOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MoonOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *MoonOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(MoonOutputParam);

MXNET_REGISTER_OP_PROPERTY(MoonOutput, MoonOutputProp)
.describe("Perform a moon transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to moon.")
.add_argument("label", "Symbol", "Label data.")
.add_arguments(MoonOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Moon, DeprecatedMoonProp)
.describe("DEPRECATED: Perform a moon transformation on input. Please use MoonOutput")
.add_argument("data", "Symbol", "Input data to moon.")
.add_arguments(MoonOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


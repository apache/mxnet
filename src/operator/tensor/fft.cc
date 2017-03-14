#include "./fft-inl.h"
namespace mxnet {
namespace op{
/*
template<>
Operator* CreateOp<cpu>(FFTParam param, int dtype) {
	Operator *op = NULL;
	switch (dtype) {
		case mshadow::kFloat32:
			op = new FFTOp<cpu, float>(param);
			break;
		case mshadow::kFloat64:
			op = new FFTOp<cpu, double>(param);
			break;
		case mshadow::kFloat16:
			LOG(FATAL) << "float16 FFT layer is currently"
										"only suppported by CuDNN version.";
			break;
		default:
			LOG(FATAL) << "Unsupported type " << dtype;
	}
	return op;
}

// 
Operator *FFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
													std::vector<int> *in_type) const {
	std::vector<TShape> out_shape, aux_shape;
	std::vector<int> out_type, aux_type;
	CHECK(InferType(in_type, &out_type, &aux_type));
	CHECK(InferShape(in_shape, &out_shape, &aux_shape));
	DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}
DMLC_REGISTER_PARAMETER(FFTParam);
*/
template<>
Operator *CreateOp<cpu>(FFTParam param, int dtype){
	LOG(FATAL) << "fft is only available for GPU.";
	return NULL;
}

Operator *FFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
													std::vector<int> *in_type) const {
	std::vector<TShape> out_shape, aux_shape;
	std::vector<int> out_type, aux_type;
	CHECK(InferType(in_type, &out_type, &aux_type));
	CHECK(InferShape(in_shape, &out_shape, &aux_shape));
	DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(FFTParam);

MXNET_REGISTER_OP_PROPERTY(FFT, FFTProp)
.describe("Apply FFT to input.")
.add_argument("data", "Symbol", "Input data to the FFTOp.")
.add_arguments(FFTParam::__FIELDS__());
} // namespace op
} // namespace mxnet
#include "./fft-inl.h"
namespace mxnet {
namespace op{
	template<>
	Operator* CreateOp<gpu>(FFTParam param, int dtype) {
      /*
      Operator *op = NULL;
	    switch (dtype) {
            case mshadow::kFloat32:
                op = new FFTOp<gpu, float>(param);
                break;
            case mshadow::kFloat64:
                op = new FFTOp<gpu, double>(param);
                break;
            case mshadow::kFloat16:
                LOG(FATAL) << "float16 FFT layer is currently"
									"only suppported by CuDNN version.";
                break;
            default:
                LOG(FATAL) << "Unsupported type " << dtype;
        }
        return op;
    
    */
    
    
    
    
		Operator *op = NULL;
		MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
			op = new FFTOp<gpu, DType>(param);
		})
		return op;
   
	}
}
}
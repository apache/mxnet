#include <iostream>
#include "lib_api.h"

void relu_cpu(float *out, float *in, int64_t N) {
    for (int i=0; i<N; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void relu_gpu(float *out, float *in, int64_t N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        out[tid] = in[tid] > 0 ? in[tid] : 0;
    }
}

MXReturnValue forward(std::map<std::string, std::string> attrs,
    std::vector<MXTensor> inputs,
    std::vector<MXTensor> outputs,
    OpResource res) {

    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();
    
    if (inputs[0].ctx == MX_GPU){
        cudaStream_t gpu_stream = reinterpret_cast<cudaStream_t>(res.get_gpu_stream());
        int64_t N = inputs[0].size();
        int grid = (N+255)/256;
        int block = 256;
        relu_gpu<<<grid,block,0,gpu_stream>>>(out_data, in_data, N);
    } else {
        relu_cpu(out_data, in_data, inputs[0].size());
    }

    return MX_SUCCESS;
}

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs, int* num_in, int* num_out) {
    *num_in = 1;
    *num_out = 1;
    return MX_SUCCESS;
  }

MXReturnValue inferType(std::map<std::string, std::string> attrs,
    std::vector<int> &intypes,
    std::vector<int> &outtypes) {
    outtypes[0] = intypes[0];
    return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string, std::string> attrs,
    std::vector<std::vector<unsigned int>> &inshapes,
    std::vector<std::vector<unsigned int>> &outshapes) {
    outshapes[0] = inshapes[0];
    return MX_SUCCESS;
}

REGISTER_OP(my_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(forward);

MXReturnValue initialize(int version) {
    if (version >= 10400) {
      std::cout << "MXNet version " << version << " supported" << std::endl;
      return MX_SUCCESS;
    } else {
      std::cout << "MXNet version " << version << " not supported" << std::endl;
      return MX_FAIL;
    }
}

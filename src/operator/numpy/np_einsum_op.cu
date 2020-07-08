/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file np_einsum_op.cu
 * \brief GPU Implementation of numpy-compatible einsum
 */

#include "./np_einsum_op-inl.h"
#include <nvToolsExt.h>

//#if MXNET_USE_CUTENSOR == 1
//#include "cutensor.h"
//#endif

namespace mxnet {
namespace op {

#if MXNET_USE_CUTENSOR == 1
template<typename U>
struct CuTensorTypeTraits;
template<>
struct CuTensorTypeTraits<double> {
  static const cudaDataType_t cudaType = CUDA_R_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_64F;
  typedef double ScalarType;
};
template<>
struct CuTensorTypeTraits<float> {
  static const cudaDataType_t cudaType = CUDA_R_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_32F;
  typedef float ScalarType;
};
template<>
struct CuTensorTypeTraits<mshadow::half::half_t> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_16F;
  typedef float ScalarType;
};


/*!
 * \brief The Operator used to perform einsum using cuTensor library.
 */
template<typename DType>
class CuTensorEinsumOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);

 public:
  CuTensorEinsumOp() {
    // not cutensorCreateTensorDescriptor or similar ?
  }
  ~CuTensorEinsumOp() {
  }
  void Init() {
  }
  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
  }

  //cutensorHandle_t handle;
  // modes
  std::vector<int> modes_a;
  std::vector<int> modes_b;
  std::vector<int> modes_c;
  // tensor descriptors
  cutensorTensorDescriptor_t descriptor_a;
  cutensorTensorDescriptor_t descriptor_b;
  cutensorTensorDescriptor_t descriptor_c;

  cudaDataType_t cudaType;
  cutensorComputeType_t cutensorType;
  
  std::unordered_map<char, int64_t> mode_2_size;

};
// end CuTensorEinsumOp class

template<typename DType>
static CuTensorEinsumOp<DType>& GetCuTensorEinsumOp() {}



std::unordered_map<char, int64_t> mode_2_size;
std::vector<int> modes_a;
std::vector<int> modes_b;
std::vector<int> modes_c;

/*inline cutensorHandle_t CreateHandle() {
  cutensorHandle_t handle;
  cutensorInit(&handle);
  return handle;
}
inline cutensorHandle_t* GetHandle() {
  static cutensorHandle_t handle = CreateHandle();
  return &handle;
}*/

inline void initialize(std::string equation,
                       const TBlob &tensor_a,
                       const TBlob &tensor_b) {
 auto comma_pos = equation.find(",");
 auto arrow_pos = equation.find("->");
 auto op_a = equation.substr(0, comma_pos);
 auto op_b = equation.substr(comma_pos + 1, arrow_pos - comma_pos - 1);
 auto op_c = equation.substr(arrow_pos + 2, equation.size() - arrow_pos + 2);

 assert(op_a.size() == tensor_a.ndim());
 for (size_t i = 0; i < op_a.size(); i++) {
   mode_2_size[op_a[i]] = tensor_a.size(i);
 }
 assert(op_b.size() == tensor_b.ndim());
 for (size_t i = 0; i < op_b.size(); i++) {
   if (mode_2_size.find(op_b[i]) == mode_2_size.end()) {
     mode_2_size[op_b[i]] = tensor_b.size(i);
   } else {
      assert(tensor_b.size(i) == mode_2_size[op_b[i]]);
   }
 }
 for (auto it = op_a.rbegin(); it != op_a.rend(); it++) {
   modes_a.push_back(*it);
 }
 for (auto it = op_b.rbegin(); it != op_b.rend(); it++) {
   modes_b.push_back(*it);
 }
 for (auto it = op_c.rbegin(); it != op_c.rend(); it++) {
   modes_c.push_back(*it);
 }
}

inline void EinsumForwardCutensor(const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs,
                                  const std::string equation,
                                  const OpContext& ctx) {
  mxnet_op::Stream<gpu>* stream = ctx.get_stream<gpu>();
  cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
  nvtxRangePush("INIT");

  const TBlob &tensor_a = inputs[0];
  const TBlob &tensor_b = inputs[1];
  const TBlob &tensor_c = outputs[0];

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    cudaDataType_t cudaType = CuTensorTypeTraits<DType>::cudaType;
    cutensorComputeType_t cutensorType = CuTensorTypeTraits<DType>::cutensorType;

    const DType* tensor_a_ptr =  tensor_a.FlatTo2D<gpu, DType>(stream).dptr_;
    const DType* tensor_b_ptr =  tensor_b.FlatTo2D<gpu, DType>(stream).dptr_;
    DType* tensor_c_ptr =  tensor_c.FlatTo2D<gpu, DType>(stream).dptr_;

    //cutensorHandle_t* handle = GetHandle();
    // using defaul algo
    cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    // initialize
    initialize(equation, tensor_a, tensor_b);

    // tensor descriptors
    cutensorTensorDescriptor_t descriptor_a;
    cutensorTensorDescriptor_t descriptor_b;
    cutensorTensorDescriptor_t descriptor_c;

    std::vector<int64_t> sizes_a;
    for(auto mode : modes_a)
        sizes_a.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&stream->cutensor_handle_,
                                               &descriptor_a,
                                               tensor_a.ndim(),
                                               sizes_a.data(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));
    std::vector<int64_t> sizes_b;
    for(auto mode : modes_b)
        sizes_b.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&stream->cutensor_handle_,
                                               &descriptor_b,
                                               tensor_b.ndim(),
                                               sizes_b.data(),
                                               NULL, //stride
                                               cudaType, CUTENSOR_OP_IDENTITY));

    std::vector<int64_t> sizes_c;
    for(auto mode : modes_c)
        sizes_c.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&stream->cutensor_handle_,
                                               &descriptor_c,
                                               tensor_c.ndim(),
                                               sizes_c.data(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));

    // aligment
    uint32_t alignment_req_a;
    uint32_t alignment_req_b;
    uint32_t alignment_req_c;

    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&stream->cutensor_handle_,
                                                  tensor_a_ptr,
                                                  &descriptor_a,
                                                  &alignment_req_a));

    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&stream->cutensor_handle_,
                                                  tensor_b_ptr,
                                                  &descriptor_b,
                                                  &alignment_req_b));

    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&stream->cutensor_handle_,
                                                  tensor_c_ptr,
                                                  &descriptor_c,
                                                  &alignment_req_c));

    // Contraction descriptor
    cutensorContractionPlan_t plan;
    cutensorContractionDescriptor_t descriptor_contraction;
    cutensorContractionFind_t find;

    CUTENSOR_CALL(cutensorInitContractionDescriptor(
                  &stream->cutensor_handle_,
                  &descriptor_contraction,
                  &descriptor_a, modes_a.data(), alignment_req_a,
                  &descriptor_b, modes_b.data(), alignment_req_b,
                  &descriptor_c, modes_c.data(), alignment_req_c,
                  &descriptor_c, modes_c.data(), alignment_req_c,
                  cutensorType));

    CUTENSOR_CALL(cutensorInitContractionFind(&stream->cutensor_handle_,
                                              &find, algo));
    // workspace to allow optimizations
    size_t workspace_size = 0;
    CUTENSOR_CALL(cutensorContractionGetWorkspace(&stream->cutensor_handle_,
                                                  &descriptor_contraction,
                                                  &find,
                                                  CUTENSOR_WORKSPACE_MAX,
                                                  &workspace_size));
    Tensor<gpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), stream);

    // Contraction Plan
    CUTENSOR_CALL(cutensorInitContractionPlan(&stream->cutensor_handle_,
                                              &plan,
                                              &descriptor_contraction,
                                              &find,
                                              workspace_size));
    cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
    nvtxRangePop();
    nvtxRangePush("einsum");

    // run einsum
    typename CuTensorTypeTraits<DType>::ScalarType alpha = 1;
    typename CuTensorTypeTraits<DType>::ScalarType beta = 0;
    CUTENSOR_CALL(cutensorContraction(&stream->cutensor_handle_,
                                      &plan,
                                      (void*) &alpha, tensor_a_ptr, tensor_b_ptr,
                                      (void*) &beta,  tensor_c_ptr, tensor_c_ptr,
                                      workspace.dptr_, workspace_size, mshadow::Stream<gpu>::GetStream(stream)));
    cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
    nvtxRangePop();
  });
}
#endif

inline void NumpyEinsumForwardGpu(const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_CUTENSOR == 1
  // cutensor only available for compute capability larger or equal to 6.0
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);
  EinsumOp& state = state_ptr.get_state<EinsumOp>();
  int num_args = state.num_args;
  //int optimize = state.optimize;
  const char* subscripts = state.subscripts.c_str();
  CHECK_EQ(inputs.size(), num_args);
  CHECK_EQ(outputs.size(), 1U);
  EinsumForwardCutensor(inputs, req, outputs, subscripts, ctx);
#else
  NumpyEinsumForward<gpu>(state_ptr, ctx, inputs, req, outputs);
#endif
}

NNVM_REGISTER_OP(_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumForwardGpu);
NNVM_REGISTER_OP(_backward_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumBackward<gpu>);

}  // namespace op
}  // namespace mxnet

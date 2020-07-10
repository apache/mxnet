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
using ModeType = int32_t;

struct CuTensorEinsumParam : public dmlc::Parameter<CuTensorEinsumParam> {
  int num_args;
  std::string equation;    
  CuTensorEinsumParam(int n, std::string eq):
    num_args(n), equation(eq) {}
  bool operator==(const CuTensorEinsumParam& other) const {
    return this->num_args == other.num_args; //&&
           ! this->equation.compare(other.equation);
  }
};
}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::CuTensorEinsumParam> {
  size_t operator()(const mxnet::op::CuTensorEinsumParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.num_args);
    ret = dmlc::HashCombine(ret, val.equation);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {
/*!
 * \brief The Operator used to perform einsum using cuTensor library.
 */
template<typename DType>
class CuTensorEinsumOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);

 public:
  CuTensorEinsumOp() {
  }
  ~CuTensorEinsumOp() {
  }
  void InitializeModes(std::string equation,
                       const mxnet::TShape& a_shape,
                       const mxnet::TShape& b_shape,
                       std::unordered_map<ModeType, int64_t>& mode_2_size,
                       std::vector<ModeType>& modes_a,
                       std::vector<ModeType>& modes_b,
                       std::vector<ModeType>& modes_c) {
    int comma_pos = equation.find(",");
    int arrow_pos = equation.find("->", comma_pos + 1);
    int a_begin = 0;
    int a_end = comma_pos;
    int b_begin = comma_pos + 1;
    int b_end = arrow_pos;
    int c_begin = arrow_pos + 2;
    int c_end = equation.size();

    assert((a_end - a_begin) == a_shape.ndim());
    for (int i = a_begin; i < a_end; i++) {
      mode_2_size[equation.at(i)] = a_shape[i - a_begin];
    }
    assert((b_end - b_begin) == b_shape.ndim());
    for (int i = b_begin; i < b_end; i++) {
      if (mode_2_size.find(equation.at(i)) == mode_2_size.end()) {
        mode_2_size[equation.at(i)] = b_shape[i - b_begin];
      } else {
         assert(b_shape[i - b_begin] == mode_2_size[equation.at(i)]);
      }
    }
    for (int i = a_end-1; i >= a_begin; i--) {
      modes_a.push_back(equation.at(i));
    }
    for (int i = b_end-1; i >= b_begin; i--) {
      modes_b.push_back(equation.at(i));
    }
    for (int i = c_end-1; i >= c_begin; i--) {
      modes_c.push_back(equation.at(i));
    }
  }

  void Init(const CuTensorEinsumParam& param,
            mxnet::ShapeVector& in_shape,
            mxnet::ShapeVector& out_shape,
            const std::vector<TBlob>& inputs,
            const std::vector<TBlob>& outputs,
            const RunContext& rctx,
            bool add_to_weight) {
    //printf("Initializing\n");
    mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();

    CHECK_EQ(in_shape.size(), 2);
    CHECK_EQ(out_shape.size(), 1);
    mxnet::TShape a_shape = in_shape[0];
    mxnet::TShape b_shape = in_shape[1];
    mxnet::TShape c_shape = out_shape[0];

    cudaType = CuTensorTypeTraits<DType>::cudaType;
    cutensorType = CuTensorTypeTraits<DType>::cutensorType;
    // using defaul algo
    algo = CUTENSOR_ALGO_DEFAULT;
    
    // initialize modes
    InitializeModes(param.equation.c_str(),
                    a_shape, b_shape,
                    mode_2_size, 
                    modes_a, modes_b, modes_c);

    std::vector<int64_t> sizes_a;
    for(auto mode : modes_a)
        sizes_a.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_a,
                                               a_shape.ndim(),
                                               sizes_a.data(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));
    std::vector<int64_t> sizes_b;
    for(auto mode : modes_b)
        sizes_b.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_b,
                                               b_shape.ndim(),
                                               sizes_b.data(),
                                               NULL, //stride
                                               cudaType, CUTENSOR_OP_IDENTITY));
    std::vector<int64_t> sizes_c;
    for(auto mode : modes_c)
        sizes_c.push_back(mode_2_size[mode]);
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_c,
                                               c_shape.ndim(),
                                               sizes_c.data(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));

    const DType* tensor_a_ptr =  inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* tensor_b_ptr =  inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    DType* tensor_c_ptr =  outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&s->cutensor_handle_,
                                                  tensor_a_ptr,
                                                  &descriptor_a,
                                                  &alignment_req_a));

    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&s->cutensor_handle_,
                                                  tensor_b_ptr,
                                                  &descriptor_b,
                                                  &alignment_req_b));

    CUTENSOR_CALL(cutensorGetAlignmentRequirement(&s->cutensor_handle_,
                                                  tensor_c_ptr,
                                                  &descriptor_c,
                                                  &alignment_req_c));

    CUTENSOR_CALL(cutensorInitContractionDescriptor(
                  &s->cutensor_handle_,
                  &descriptor_contraction,
                  &descriptor_a, modes_a.data(), alignment_req_a,
                  &descriptor_b, modes_b.data(), alignment_req_b,
                  &descriptor_c, modes_c.data(), alignment_req_c,
                  &descriptor_c, modes_c.data(), alignment_req_c,
                  cutensorType));

    CUTENSOR_CALL(cutensorInitContractionFind(&s->cutensor_handle_,
                                              &find, algo));

    CUTENSOR_CALL(cutensorContractionGetWorkspace(&s->cutensor_handle_,
                                                  &descriptor_contraction,
                                                  &find,
                                                  CUTENSOR_WORKSPACE_MAX,
                                                  &workspace_size));

    CUTENSOR_CALL(cutensorInitContractionPlan(&s->cutensor_handle_,
                                              &plan,
                                              &descriptor_contraction,
                                              &find,
                                              workspace_size));
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs) {
    //printf("Forward\n");
    mxnet_op::Stream<gpu>* s = ctx.get_stream<gpu>();

    const TBlob &tensor_a = inputs[0];
    const TBlob &tensor_b = inputs[1];
    const TBlob &tensor_c = outputs[0];
    const DType* tensor_a_ptr =  tensor_a.FlatTo2D<gpu, DType>(s).dptr_;
    const DType* tensor_b_ptr =  tensor_b.FlatTo2D<gpu, DType>(s).dptr_;
    DType* tensor_c_ptr =  tensor_c.FlatTo2D<gpu, DType>(s).dptr_;
    
    Tensor<gpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);

    CUTENSOR_CALL(cutensorContraction(&s->cutensor_handle_,
                                      &plan,
                                      (void*) &alpha, tensor_a_ptr, tensor_b_ptr,
                                      (void*) &beta,  tensor_c_ptr, tensor_c_ptr,
                                      workspace.dptr_, 
                                      workspace_size, 
                                      mshadow::Stream<gpu>::GetStream(s)));
  }

  // modes
  std::unordered_map<ModeType, int64_t> mode_2_size;
  std::vector<int> modes_a;
  std::vector<int> modes_b;
  std::vector<int> modes_c;

  // descriptors
  cutensorTensorDescriptor_t descriptor_a;
  cutensorTensorDescriptor_t descriptor_b;
  cutensorTensorDescriptor_t descriptor_c;
  cutensorContractionDescriptor_t descriptor_contraction;
  // aligments
  uint32_t alignment_req_a;
  uint32_t alignment_req_b;
  uint32_t alignment_req_c;

  // contraction plan and algo
  cutensorContractionPlan_t plan;
  cutensorContractionFind_t find;
  cutensorAlgo_t algo;

  // workspace
  size_t workspace_size = 0;  
  
  typename CuTensorTypeTraits<DType>::ScalarType alpha = 1;
  typename CuTensorTypeTraits<DType>::ScalarType beta = 0;

  cudaDataType_t cudaType;
  cutensorComputeType_t cutensorType;
};
// end CuTensorEinsumOp class

typedef ParamOpSign<CuTensorEinsumParam> EinsumSignature;
template<typename DType>
static CuTensorEinsumOp<DType>& GetCuTensorEinsumOp(const CuTensorEinsumParam& param,
                                                    const std::vector<TBlob>& inputs,
                                                    const std::vector<TBlob>& outputs,
                                                    const RunContext& rctx,
                                                    bool add_to_weight) {
  //printf("GetCuTensorEinsumOp\n");
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<EinsumSignature,
                                         std::shared_ptr<CuTensorEinsumOp<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<EinsumSignature,
                                            std::shared_ptr<CuTensorEinsumOp<DType> >,
                                            OpHash> ops;
#endif
  EinsumSignature key(param);
  size_t ndim = 0;
  mxnet::ShapeVector in_shape(inputs.size());
  mxnet::ShapeVector out_shape(1, outputs[0].shape_);
  for (size_t i = 0; i < in_shape.size(); i++)
    in_shape[i] = inputs[i].shape_;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(ndim + // for in and out shapes
              1 + // for dev_id
              1 ); // for add_to_weight
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(rctx.ctx.dev_id);
  key.AddSign(add_to_weight ? 1 : 0);
  /// !!!! I think we need to check Aligment as well, which will lead to:
  // InitializeModes, Initialize Tesor Descriptors & cutensorGetAlignmentRequirement
  // still will avoid: cutensorInitContractionDescriptor, cutensorInitContractionFind,
  // cutensorContractionGetWorkspace & cutensorInitContractionPlan

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuTensorEinsumOp<DType>> op(new CuTensorEinsumOp<DType>());
    auto ins_ret = ops.insert(std::pair<EinsumSignature, std::shared_ptr<CuTensorEinsumOp<DType>>>(
                              key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, in_shape, out_shape, 
                     inputs, outputs,
                     rctx, add_to_weight);
  }
  return *it->second;
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
  const EinsumOp& state = state_ptr.get_state<EinsumOp>();
  CuTensorEinsumParam cutensor_param(state.num_args, state.subscripts);
  auto add_to_weight = false;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    CuTensorEinsumOp<DType> &op = GetCuTensorEinsumOp<DType>
        (cutensor_param, inputs, outputs,
         ctx.run_ctx, add_to_weight);
    op.Forward(ctx, inputs, req, outputs);
  });
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

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

// Round a value 'x' up to the next multiple of 'multiple'
size_t RoundToMultiple(size_t x, size_t multiple) {
  size_t retVal = ((x + multiple - 1) / multiple) * multiple;
  return retVal;
}
}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::EinsumOp> {
  size_t operator()(const mxnet::op::EinsumOp& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.num_args);
    ret = dmlc::HashCombine(ret, val.subscripts);
    ret = dmlc::HashCombine(ret, val.optimize);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename ComputeType,
         typename IntType, int kMaxNumModes_>
struct Einsum
{
    Einsum(const std::string &equation,
           const mxnet::TShape &A_shape,
           const mxnet::TShape &B_shape) :
        numModesA_(A_shape.ndim()),
        numModesB_(B_shape.ndim()),
        numModesC_(0),
        isInitialized_(false)
    {
        const auto arrow_pos = equation.find("->");
        const auto comma_pos = equation.find(",");
        const auto dots = equation.find("...");
        const bool isBroadcast = (dots != std::string::npos);
        const bool isImplicit = (arrow_pos == std::string::npos);
        if (isBroadcast) // TODO
        {
            return;
        }
        const bool usesB = (comma_pos != std::string::npos);

        size_t a_start = 0;
        size_t a_end = isImplicit ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos) : 
                                    ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
        size_t b_start = usesB ? comma_pos + 1 : 0;
        size_t b_end   = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
        size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
        size_t c_end = equation.size();


        char modeA[kMaxNumModes_ + 2];
        uint32_t numModesA = 0;
        for (int i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeA[numModesA++] = equation.at(i);
            }
        }

        char modeB[kMaxNumModes_ + 2];
        uint32_t numModesB = 0;
        for (int i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeB[numModesB++] = equation.at(i);
            }
        }

        char modeC[kMaxNumModes_ + 2];
        uint32_t numModesC = 0;
        for (int i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeC[numModesC++] = equation.at(i);
            }
        }

        if ((numModesA != numModesA_) || (numModesB != numModesB_))
        {
            // substring size and shape don't match
            return;
        }
        if (numModesA_ > kMaxNumModes_ || numModesB_ > kMaxNumModes_)
        {
            // too many modes
            return;
        }

        /**
         * Copy all modes from modeA to modeC if they don't appear in modeB
         */
        auto copyModesIf = [](const char* modeA, uint32_t numModesA,
                const char* modeB, uint32_t numModesB,
                char* modeC, uint32_t &numModesC)
        {
            for (uint32_t i = 0; i < numModesA; i++)
            {
                auto mode = modeA[i];
                bool found = false;
                for(uint32_t j=0; j < numModesB; ++j){
                    if(mode == modeB[j])
                    {
                        found = true;
                        break;
                    }
                }

                if (!found) // is non-contracted mode
                {
                    modeC[numModesC++] = mode;
                    if (numModesC > kMaxNumModes_)
                    {
                        // too many modes
                        return false;
                    }
                }
            }
            return true;
        };


        std::array<char, kMaxNumModes_+1> implicitModeC;
        char* redirectModeC;
        if (isImplicit)
        {
            // we have to copy all non-contracted modes from A over to C
            if (copyModesIf(modeA, numModesA_, modeB, numModesB_, implicitModeC.data(), numModesC_) == false)
            {
                return;
            }
            // we have to copy all non-contracted modes from B over to C
            if (copyModesIf(modeB, numModesB_, modeA, numModesA_, implicitModeC.data(), numModesC_) == false)
            {
                return;
            }
            std::sort(implicitModeC.begin(), std::next(implicitModeC.begin(), numModesC_)); // modes are sorted w.r.t. lexical order
            implicitModeC[numModesC_] = '\0';
            redirectModeC = implicitModeC.data();
        }
        else
        {
            redirectModeC = modeC;
            numModesC_ = numModesC;
        }

        for (uint32_t i = 0; i < numModesA_; i++)
        {
            modesA_[i] = modeA[numModesA_ - i - 1];
            extentA_[i] = A_shape[numModesA_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesB_; i++)
        {
            modesB_[i] = modeB[numModesB_ - i - 1];
            extentB_[i] = B_shape[numModesB_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesC_; i++)
        {
            const auto mode = redirectModeC[numModesC_ - i - 1];
            modesC_[i] = mode;
            bool found = false;
            for (uint32_t j=0; j < numModesA_; ++j)
            {
                if (modesA_[j] == mode)
                {
                    extentC_[i] = extentA_[j];
                    found = true;
                    break;
                }
            }
            for (uint32_t j=0; !found && j < numModesB_; ++j)
            {
                if (modesB_[j] == mode)
                {
                    extentC_[i] = extentB_[j];
                    break;
                }
            }
        }

        isInitialized_ = true;
    }

    std::vector<IntType> getOutputShape() const
    {
        if (!isInitialized_) return {};
        std::vector<IntType> extentC(numModesC_);
        for (int i=0; i < numModesC_; ++i)
        {
            extentC[i] = extentC_.at(numModesC_ - i - 1);
        }

        return extentC;
    }

    bool isInitialized() const { return isInitialized_; }

    const int64_t* getExtentsA() const { return extentA_.data(); }
    const int64_t* getExtentsB() const { return extentB_.data(); }
    const int64_t* getExtentsC() const { return extentC_.data(); }

    const int* getModesA() const { return modesA_.data(); }
    const int* getModesB() const { return modesB_.data(); }
    const int* getModesC() const { return modesC_.data(); }

    private:
    uint32_t numModesA_;
    uint32_t numModesB_;
    uint32_t numModesC_;
    bool isInitialized_;
    std::array<int, kMaxNumModes_> modesA_;
    std::array<int, kMaxNumModes_> modesB_;
    std::array<int, kMaxNumModes_> modesC_;
    std::array<int64_t, kMaxNumModes_> extentA_;
    std::array<int64_t, kMaxNumModes_> extentB_;
    std::array<int64_t, kMaxNumModes_> extentC_;
};

/*!
 * \brief The Operator used to perform einsum using cuTensor library.
 */
template<typename DType>
class CuTensorEinsum {
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);
  static_assert(CUTENSOR_MAJOR >= 1 && CUTENSOR_MINOR >= 2 && CUTENSOR_PATCH >= 0, "minimal cuTENSOR 1.2.0 is required.");
 public:
  CuTensorEinsum() {
  }
  ~CuTensorEinsum() {
  }

  size_t Init(std::string equation,
              const mxnet::ShapeVector& in_shape,
              const mxnet::ShapeVector& out_shape,
              const OpContext& ctx,
              bool req_write,
              size_t prev_workspace_size,
              size_t alignment) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(in_shape.size(), 2);
    CHECK_EQ(out_shape.size(), 1);

    constexpr cudaDataType_t cudaType = CuTensorTypeTraits<DType>::cudaType;
    constexpr cutensorComputeType_t cutensorType = CuTensorTypeTraits<DType>::cutensorType;
    constexpr cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    Einsum<DType, int, kMaxTensorRank> myEinsum(equation, in_shape[0], in_shape[1]);
    if (!myEinsum.isInitialized()) {
        CUTENSOR_CALL(CUTENSOR_STATUS_NOT_SUPPORTED);
    }

    cutensorTensorDescriptor_t descriptor_a;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_a,
                                               in_shape[0].ndim(),
                                               myEinsum.getExtentsA(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descriptor_b;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_b,
                                               in_shape[1].ndim(),
                                               myEinsum.getExtentsB(),
                                               NULL, //stride
                                               cudaType, CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descriptor_c;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_c,
                                               out_shape[0].ndim(),
                                               myEinsum.getExtentsC(),
                                               NULL, //stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));

    CUTENSOR_CALL(cutensorInitContractionDescriptor(
                  &s->cutensor_handle_,
                  &descriptor_contraction,
                  &descriptor_a, myEinsum.getModesA(), alignment,
                  &descriptor_b, myEinsum.getModesB(), alignment,
                  &descriptor_c, myEinsum.getModesC(), alignment,
                  &descriptor_c, myEinsum.getModesC(), alignment,
                  cutensorType));

    CUTENSOR_CALL(cutensorInitContractionFind(&s->cutensor_handle_,
                                              &find, algo));

    if (s->cutensor_cachelines_ != nullptr)
    {
        const cutensorAutotuneMode_t autotuneMode = CUTENSOR_AUTOTUNE_INCREMENTAL;
        CUTENSOR_CALL(cutensorContractionFindSetAttribute(
                    &s->cutensor_handle_,
                    &find,
                    CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
                    &autotuneMode,
                    sizeof(cutensorAutotuneMode_t)));

        const uint32_t incCount = 5;
        CUTENSOR_CALL(cutensorContractionFindSetAttribute(
                    &s->cutensor_handle_,
                    &find,
                    CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
                    &incCount,
                    sizeof(uint32_t)));
    } 

    previous_workspace_size = prev_workspace_size * sizeof(DType);
    CUTENSOR_CALL(cutensorContractionGetWorkspace(&s->cutensor_handle_,
                                                  &descriptor_contraction,
                                                  &find,
                                                  CUTENSOR_WORKSPACE_MAX,
                                                  &my_workspace_size));
    if (s->cutensor_cachelines_ == nullptr){
        CUTENSOR_CALL(cutensorInitContractionPlan(&s->cutensor_handle_,
                                              &plan,
                                              &descriptor_contraction,
                                              &find,
                                              my_workspace_size));
    }

    return my_workspace_size;
  }

  void Compute(const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               bool req_write,
               const std::vector<TBlob> &outputs,
               char* workspace) {
    Stream<gpu>* s = ctx.get_stream<gpu>();

    if (s->cutensor_cachelines_ != nullptr)
    {
        CUTENSOR_CALL(cutensorInitContractionPlan(&s->cutensor_handle_,
                                              &plan,
                                              &descriptor_contraction,
                                              &find,
                                              my_workspace_size));
    }

    const TBlob &tensor_a = inputs[0];
    const TBlob &tensor_b = inputs[1];
    const TBlob &tensor_c = outputs[0];
    const DType* tensor_a_ptr =  tensor_a.FlatTo2D<gpu, DType>(s).dptr_;
    const DType* tensor_b_ptr =  tensor_b.FlatTo2D<gpu, DType>(s).dptr_;
    DType* tensor_c_ptr =  tensor_c.FlatTo2D<gpu, DType>(s).dptr_;
    char* my_workspace(&workspace[previous_workspace_size]);
    CUTENSOR_CALL(cutensorContraction(&s->cutensor_handle_,
                                      &plan,
                                      (void*) &alpha, tensor_a_ptr, tensor_b_ptr,
                                      (void*) &beta,  tensor_c_ptr, tensor_c_ptr,
                                      my_workspace,
                                      my_workspace_size,
                                      s->stream_));
  }

  static const uint32_t kMaxTensorRank = 12; // maximal tensor rank that is supported by cuTENSOR

  cutensorContractionDescriptor_t descriptor_contraction; // encodes the strucutre of the contraction
  cutensorContractionPlan_t plan; // encodes the execution plan
  cutensorContractionFind_t find; // limits the search space (of viable candidates/implementations)

  // workspace
  size_t previous_workspace_size = 0;
  size_t my_workspace_size = 0;
  size_t total_workspace_size = 0;
  
  typename CuTensorTypeTraits<DType>::ScalarType alpha = 1;
  typename CuTensorTypeTraits<DType>::ScalarType beta = 0;
};
// end CuTensorEinsum class

template<typename DType>
class EinsumOpGPU {

 public:
  EinsumOpGPU() {
  }
  ~EinsumOpGPU() {
  }

  void InitCuTensorGrad(std::string equation,
                        const mxnet::ShapeVector& in_shape,
                        const mxnet::ShapeVector& out_shape,
                        const OpContext &ctx,
                        size_t &pos_cutensor_op,
                        size_t temp_grad_size_aligned){
    bool req_write = true;
    int comma_pos = equation.find(",");
    int arrow_pos = equation.find("->", comma_pos + 1);
    int len_op2 = arrow_pos - comma_pos - 1;

    // gradient for first operand
    mxnet::ShapeVector grad_op1_input_shapes;
    mxnet::ShapeVector grad_op1_output_shapes;
    grad_op1_input_shapes.push_back(in_shape[0]);
    grad_op1_input_shapes.push_back(in_shape[2]);
    grad_op1_output_shapes.push_back(out_shape[0]);
    std::string grad_operand1_equation = equation.substr(arrow_pos + 2);
    grad_operand1_equation += ",";
    grad_operand1_equation += equation.substr(comma_pos + 1, len_op2);
    grad_operand1_equation += "->";
    grad_operand1_equation += equation.substr(0, comma_pos);
    bwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
    size_t req_workspace =
      bwd_cutensor_ops[pos_cutensor_op].Init(grad_operand1_equation,
                                             grad_op1_input_shapes,
                                             grad_op1_output_shapes,
                                             ctx, req_write,
                                             temp_grad_size_aligned,
                                             dptr_alignment);
    if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;
    pos_cutensor_op++;

    // gradient for second operand
    mxnet::ShapeVector grad_op2_input_shapes;
    mxnet::ShapeVector grad_op2_output_shapes;
    grad_op2_input_shapes.push_back(in_shape[1]);
    grad_op2_input_shapes.push_back(in_shape[0]);
    grad_op2_output_shapes.push_back(out_shape[1]);
    std::string grad_operand2_equation = equation.substr(0, comma_pos);
    grad_operand2_equation += ",";
    grad_operand2_equation += equation.substr(arrow_pos + 2);
    grad_operand2_equation += "->";
    grad_operand2_equation += equation.substr(comma_pos + 1, len_op2);
    bwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
    req_workspace =
      bwd_cutensor_ops[pos_cutensor_op].Init(grad_operand2_equation,
                                             grad_op2_input_shapes,
                                             grad_op2_output_shapes,
                                             ctx, req_write,
                                             temp_grad_size_aligned,
                                             dptr_alignment);
    if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;
    pos_cutensor_op++;
  }

  void Init(const EinsumOp& state,
            const mxnet::ShapeVector& in_shape,
            const mxnet::ShapeVector& out_shape,
            const OpContext& ctx,
            bool req_write,
            bool is_backward) {
    if (!is_backward) {
      // forward
      if (state.num_args == 2) {
        fwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
        max_workspace_cutensor = fwd_cutensor_ops[0].Init(state.subscripts,
                                                          in_shape, out_shape,
                                                          ctx, req_write,
                                                          0, dptr_alignment);
      } else {
        // more than 2 operands, optimal path
        paths_len = state.paths.size();
        max_workspace_cutensor = 0;
        mxnet::ShapeVector operands_shape(in_shape);
        for (int i = 0; i < paths_len; ++i) {
          bool handle_out = (i == paths_len - 1);
          mxnet::ShapeVector tmp_in_shape;
          mxnet::ShapeVector tmp_out_shape;
          // remove inds from right to left
          for (const int& p : state.paths[i].contract_inds) {
            tmp_in_shape.push_back(operands_shape[p]);
            operands_shape.erase(operands_shape.begin() + p);
          }
          if (handle_out) tmp_out_shape.push_back(out_shape[0]);
          else tmp_out_shape.push_back(state.paths[i].oshape);
          fwd_cutensor_ops.push_back(CuTensorEinsum<DType>());

          size_t req_workspace = fwd_cutensor_ops[i].Init(state.paths[i].einsum_str,
                                                          tmp_in_shape,
                                                          tmp_out_shape,
                                                          ctx, req_write,
                                                          0, dptr_alignment);
          if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;
          temp_ouputs_size += state.paths[i].oshape.Size();
          if (!handle_out) {
            operands_shape.push_back(state.paths[i].oshape);
          }
        }
      }
    } else {
      // backward
      max_workspace_cutensor = 0;
      size_t pos_cutensor_bwd_op = 0;
      if (state.num_args == 2) {
        InitCuTensorGrad(state.subscripts,
                         in_shape, out_shape,
                         ctx, pos_cutensor_bwd_op, 0);
        total_workspace = max_workspace_cutensor/sizeof(DType);
      } else {
        // more than 2 operands, optimal path
        paths_len = state.paths.size();
        // replay the forward process
        bwd_op_idx.resize(paths_len + 1);
        for (int i = 0; i <= paths_len; ++i) {
          if (i == 0) {
            bwd_op_idx[i].reserve(state.num_args);
            for (int j = 0; j < state.num_args; ++j) {
              bwd_op_idx[i].push_back(j + 1);
            }
          } else {
            bwd_op_idx[i] = bwd_op_idx[i - 1];
            // remove inds from right to left
            for (const int& p : state.paths[i - 1].contract_inds) {
              bwd_op_idx[i].erase(bwd_op_idx[i].begin() + p);
            }
            bwd_op_idx[i].push_back(-static_cast<int>(i - 1));
          }
        }
        // calculate amount mem for temporal grads
        for (int i = 0; i + 1 < paths_len; ++i) {
          temp_grads_size += state.paths[i].oshape.Size();
        }
        temp_grads_size_aligned = RoundToMultiple(temp_grads_size, dptr_alignment);
        // go through the paths in the reversed order
        mxnet::ShapeVector temp_in_shape, temp_out_shape;
        for (int i = paths_len - 1; i >= 0; i--) {
          temp_in_shape.clear();
          temp_out_shape.clear();
          bool handle_out = (i == paths_len - 1);
          if (handle_out) {
            // grad_out
            temp_in_shape.push_back(in_shape[0]);
          } else {
            temp_in_shape.push_back(state.paths[i].oshape);
          }
          for (auto p : state.paths[i].contract_inds) {
            int idx = bwd_op_idx[i][p];
            if (idx >= 1) {
              temp_in_shape.push_back(in_shape[idx]);
              temp_out_shape.push_back(out_shape[idx - 1]);
            } else {
              temp_in_shape.push_back(state.paths[-idx].oshape);
              temp_out_shape.push_back(state.paths[-idx].oshape);
            }
          }
          CHECK_EQ(temp_in_shape.size(), 3U);
          CHECK_EQ(temp_out_shape.size(), 2U);

          InitCuTensorGrad(state.paths[i].einsum_str,
                           temp_in_shape, temp_out_shape,
                           ctx, pos_cutensor_bwd_op,
                           temp_grads_size_aligned);
        }
        total_workspace = max_workspace_cutensor/sizeof(DType) +
                          temp_grads_size;
      }
    }
  }

  void Forward(const EinsumOp& state,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    bool req_write = false;
    // cuTensor workspace
    Tensor<gpu, 1, char> cutensor_workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(max_workspace_cutensor), s);

    if (state.num_args == 2) {
      fwd_cutensor_ops[0].Compute(ctx, inputs, req_write, outputs, cutensor_workspace.dptr_);
    } else {
      // more than 2 operands, compute optimal path
      std::vector<TBlob> operands(inputs);
      std::vector<TBlob> tmp_operands;

      // temporal space shared with backward: stateful
      std::vector<TBlob> temp_space_vec(paths_len - 1);
      Tensor<gpu, 1, DType> temp_space = state.tempspace->data().FlatTo1D<gpu, DType>();
      size_t begin = 0;
      for (int i = 0; i < paths_len - 1; ++i) {
        TBlob tblob = TBlob(temp_space.Slice(begin, begin + state.paths[i].oshape.Size()));
        temp_space_vec[i] = tblob.reshape(state.paths[i].oshape);
        begin = begin + state.paths[i].oshape.Size();
      }
      for (int i = 0; i < paths_len; ++i) {
        bool handle_out = (i == paths_len - 1);
        tmp_operands.clear();
        // remove inds from right to left
        for (const int& p : state.paths[i].contract_inds) {
          tmp_operands.push_back(operands[p]);
          operands.erase(operands.begin() + p);
        }
        fwd_cutensor_ops[i].Compute(ctx, tmp_operands, req_write,
                                    handle_out ? outputs :
                                                 std::vector<TBlob>{temp_space_vec[i]},
                                    cutensor_workspace.dptr_);
        if (!handle_out) {
          operands.push_back(temp_space_vec[i]);
        }
      }
    }
  }

  void ComputeGradients(const std::string &equation,
                        const std::vector<TBlob> &inputs,
                        const std::vector<TBlob> &outputs,
                        const OpContext &ctx,
                        size_t &pos_cutensor_op,
                        Tensor<gpu, 1, DType> &workspace){
    char* workspace_ptr = reinterpret_cast<char*>(workspace.dptr_);
    bool req_write = true;
    // gradient for first operand
    std::vector<TBlob> grad_operand1_inputs;
    std::vector<TBlob> grad_operand1_outputs;
    grad_operand1_inputs.push_back(inputs[0]);
    grad_operand1_inputs.push_back(inputs[2]);
    grad_operand1_outputs.push_back(outputs[0]);
    bwd_cutensor_ops[pos_cutensor_op].Compute(ctx, 
                                              grad_operand1_inputs, 
                                              req_write,
                                              grad_operand1_outputs,
                                              workspace_ptr);
    pos_cutensor_op ++;
    // gradient for second operand
    std::vector<TBlob> grad_operand2_inputs;
    std::vector<TBlob> grad_operand2_outputs;
    grad_operand2_inputs.push_back(inputs[1]);
    grad_operand2_inputs.push_back(inputs[0]);
    grad_operand2_outputs.push_back(outputs[1]);
    bwd_cutensor_ops[pos_cutensor_op].Compute(ctx, 
                                              grad_operand2_inputs, 
                                              req_write,
                                              grad_operand2_outputs,
                                              workspace_ptr);
    pos_cutensor_op++;
  }

  void Backward(const EinsumOp& state,
                const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    auto req_write = req[0] == kWriteTo;
    size_t pos_cutensor_op = 0;
    if (state.num_args == 2) {
      // workspace for cuTensor
      Tensor<gpu, 1, DType> temp_space =
        ctx.requested[0].get_space_typed<gpu, 1, DType>(Shape1(total_workspace), s);
      // inputs: out_grad, operand1, operand2
      // outputs: grad_operand1, grad_operand2
      ComputeGradients(state.subscripts,
                       inputs, outputs,
                       ctx, pos_cutensor_op,
                       temp_space);
    } else {
      // more than 2 operands, compute optimal path
      // outputs from forward pass, no need to be re-computed, take from state
      Tensor<gpu, 1, DType> ndarray_space = state.tempspace->data().FlatTo1D<gpu, DType>();
      std::vector<TBlob> temp_data(paths_len - 1);
      size_t begin = 0;
      for (int i = 0; i + 1 < paths_len; ++i) {
        TBlob tblob = TBlob(ndarray_space.Slice(begin, begin + state.paths[i].oshape.Size()));
        temp_data[i] = tblob.reshape(state.paths[i].oshape);
        begin = begin + state.paths[i].oshape.Size();
      }
      // workspace (temporal grad + cuTensor)
      std::vector<TBlob> temp_grad(paths_len - 1);
      Tensor<gpu, 1, DType> temp_space =
        ctx.requested[0].get_space_typed<gpu, 1, DType>(Shape1(total_workspace), s);
      begin = 0;
      for (int i = 0; i + 1 < paths_len; ++i) {
        TBlob tblob = TBlob(temp_space.Slice(begin, begin + state.paths[i].oshape.Size()));
        temp_grad[i] = tblob.reshape(state.paths[i].oshape);
        begin = begin + state.paths[i].oshape.Size();
      }
      // go through the paths in the reversed order
      std::vector<TBlob> temp_inputs, temp_outputs;
      //std::vector<OpReqType> temp_req;
      for (int i = paths_len - 1; i >= 0; i--) {
        temp_inputs.clear();
        temp_outputs.clear();
        //temp_req.clear();
        bool handle_out = (i == paths_len - 1);
        if (handle_out) {
          // grad_out
          temp_inputs.push_back(inputs[0]);
        } else {
          temp_inputs.push_back(temp_grad[i]);
        }
        for (auto p : state.paths[i].contract_inds) {
          int idx = bwd_op_idx[i][p];
          if (idx >= 1) {
            temp_inputs.push_back(inputs[idx]);
            temp_outputs.push_back(outputs[idx - 1]);
            //temp_req.push_back(req[idx - 1]);
          } else {
            temp_inputs.push_back(temp_data[-idx]);
            temp_outputs.push_back(temp_grad[-idx]);
            //temp_req.push_back(OpReqType::kWriteTo);
          }
        }
        CHECK_EQ(temp_inputs.size(), 3U);
        CHECK_EQ(temp_outputs.size(), 2U);
        //CHECK_EQ(temp_req.size(), 2U);

        ComputeGradients(state.paths[i].einsum_str,
                         temp_inputs, temp_outputs,
                         ctx, pos_cutensor_op,
                         temp_space);
      }
    }
  }

  int paths_len = 0;
  // cutensor ops for the forward and backward passs:
  std::vector<CuTensorEinsum<DType>> fwd_cutensor_ops;
  std::vector<CuTensorEinsum<DType>> bwd_cutensor_ops;
  std::vector<std::vector<int> > bwd_op_idx;
  
  const size_t dptr_alignment = 512;
  size_t temp_ouputs_size = 0; 
  size_t temp_grads_size = 0;
  size_t temp_grads_size_aligned = 0;
  size_t max_workspace_cutensor = 0;
  size_t total_workspace = 0;
};

typedef ParamOpSign<EinsumOp> EinsumSignature;
template<typename DType>
static EinsumOpGPU<DType>& GetEinsumOpGPU(const EinsumOp& state,
                                          const mxnet::ShapeVector& in_shape,
                                          const mxnet::ShapeVector& out_shape,
                                          const OpContext& ctx,
                                          bool req_write,
                                          bool is_backward) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<EinsumSignature,
                                         std::shared_ptr<EinsumOpGPU<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<EinsumSignature,
                                            std::shared_ptr<EinsumOpGPU<DType> >,
                                            OpHash> ops;
#endif
  EinsumSignature key(state);
  size_t ndim = 0;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(ndim + // for in and out shapes
              1 + // for dev_id
              1 + // for req_write
              1 ); // is_backward
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(ctx.run_ctx.ctx.dev_id);
  key.AddSign(req_write ? 1 : 0);
  key.AddSign(is_backward ? 1 : 0);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<EinsumOpGPU<DType>> op(new EinsumOpGPU<DType>());
    auto ins_ret = ops.insert(std::pair<EinsumSignature, std::shared_ptr<EinsumOpGPU<DType>>>(
                              key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(state,
                     in_shape, out_shape,
                     ctx, req_write, is_backward);
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
  EinsumOp& state = state_ptr.get_state<EinsumOp>();
  auto req_write = false;
  if (state.num_args <= 1) {
    NumpyEinsumForward<gpu>(state_ptr, ctx, inputs, req, outputs);
  } else {
    if (state.num_args > 2) {
      std::vector<Step>& paths = state.paths;
      std::vector<std::vector<int> > pos;
      std::string string_repr;
      paths = einsum_path(state.subscripts, inputs, true, ctx.run_ctx, &pos, &string_repr);
    }
    mxnet::ShapeVector in_shape(inputs.size());
    mxnet::ShapeVector out_shape(1, outputs[0].shape_);
    for (size_t i = 0; i < in_shape.size(); i++)
      in_shape[i] = inputs[i].shape_;

    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      EinsumOpGPU<DType> &op = GetEinsumOpGPU<DType>
          (state, in_shape, out_shape,
           ctx, req_write, false);
      state.tempspace.reset<NDArray>(new NDArray(TShape(Shape1(op.temp_ouputs_size)),
                                               ctx.run_ctx.ctx,
                                               false,
                                               outputs[0].type_flag_));
      op.Forward(state, ctx, inputs, req, outputs);
    });
  }
#else
  NumpyEinsumForward<gpu>(state_ptr, ctx, inputs, req, outputs);
#endif
}

inline void NumpyEinsumBackwardGpu(const OpStatePtr& state_ptr,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
#if MXNET_USE_CUTENSOR == 1
  // cutensor only available for compute capability larger or equal to 6.0
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);
  const EinsumOp& state = state_ptr.get_state<EinsumOp>();
  auto req_write = req[0] == kWriteTo;
  if (state.num_args <= 1) {
    NumpyEinsumBackward<gpu>(state_ptr, ctx, inputs, req, outputs);
  } else {
    mxnet::ShapeVector in_shape(inputs.size());
    mxnet::ShapeVector out_shape(outputs.size());
    for (size_t i = 0; i < in_shape.size(); i++)
      in_shape[i] = inputs[i].shape_;
    for (size_t i = 0; i < out_shape.size(); i++)
      out_shape[i] = outputs[i].shape_;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      EinsumOpGPU<DType> &op = GetEinsumOpGPU<DType>
          (state, in_shape, out_shape,
           ctx, req_write, true);
      op.Backward(state, ctx, inputs, req, outputs);
    });
  }
#else
  NumpyEinsumBackward<gpu>(state_ptr, ctx, inputs, req, outputs);
#endif
}

NNVM_REGISTER_OP(_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumForwardGpu);

NNVM_REGISTER_OP(_backward_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumBackwardGpu);

}  // namespace op
}  // namespace mxnet

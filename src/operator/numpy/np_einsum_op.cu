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
// maximal tensor rank that is supported by cuTENSOR
static const uint32_t kMaxTensorRank = 12;

template<typename U>
struct CuTensorTypeTraits;
template<>
struct CuTensorTypeTraits<double> {
  static const cudaDataType_t cudaType = CUDA_R_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef double ScalarType;
};
template<>
struct CuTensorTypeTraits<float> {
  static const cudaDataType_t cudaType = CUDA_R_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};
template<>
struct CuTensorTypeTraits<mshadow::half::half_t> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_16F;
  typedef float ScalarType;
};

// Round num elements 'x' to be mem aligned according to 'multiple' and 'dtype_size'
size_t RoundToMultiple(size_t x, size_t multiple, size_t dtype_size) {
  size_t retVal = ((x*dtype_size + multiple - 1) / multiple) * multiple;
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

template<typename ComputeType, typename IntType, int kMaxNumModes_>
struct Einsum {
  Einsum(const std::string &equation,
         const mxnet::TShape &a_shape,
         const mxnet::TShape &b_shape):
         num_modes_a_(a_shape.ndim()),
         num_modes_b_(b_shape.ndim()),
         num_modes_c_(0),
         is_initialized_(false) {
    const auto arrow_pos = equation.find("->");
    const auto comma_pos = equation.find(",");
    const bool is_implicit = (arrow_pos == std::string::npos);
    const bool uses_b = (comma_pos != std::string::npos);

    size_t a_start = 0;
    size_t a_end = is_implicit ? (uses_b ? comma_pos : equation.size()):
                                 (uses_b ? comma_pos : arrow_pos);
    size_t b_start = uses_b ? comma_pos + 1 : 0;
    size_t b_end   = uses_b ? (is_implicit ? equation.size() : arrow_pos) : 0;
    size_t c_start = is_implicit ? equation.size() : arrow_pos + 2;
    size_t c_end = equation.size();

    char mode_a[kMaxNumModes_ + 2];
    uint32_t num_modes_a = 0;
    for (int i = a_start; i < a_end && num_modes_a < kMaxNumModes_ + 2; ++i) {
      mode_a[num_modes_a++] = equation.at(i);
    }
    char mode_b[kMaxNumModes_ + 2];
    uint32_t num_modes_b = 0;
    for (int i = b_start; i < b_end && num_modes_b < kMaxNumModes_ + 2; ++i) {
      mode_b[num_modes_b++] = equation.at(i);
    }
    char mode_c[kMaxNumModes_ + 2];
    uint32_t num_modes_c = 0;
    for (int i = c_start; i < c_end && num_modes_c < kMaxNumModes_ + 2; ++i) {
      mode_c[num_modes_c++] = equation.at(i);
    }

    if ((num_modes_a != num_modes_a_) || (num_modes_b != num_modes_b_)) {
      // substring size and shape don't match
      return;
    }
    if (num_modes_a_ > kMaxNumModes_ || num_modes_b_ > kMaxNumModes_) {
      // too many modes
      return;
    }

    /**
    * Copy all modes from mode_a to mode_c if they don't appear in mode_b
    */
    auto CopyModesIf = [](const char* mode_a, uint32_t num_modes_a,
                          const char* mode_b, uint32_t num_modes_b,
                          char* mode_c, uint32_t &num_modes_c) {
      for (uint32_t i = 0; i < num_modes_a; i++) {
        auto mode = mode_a[i];
        bool found = false;
        for (uint32_t j = 0; j < num_modes_b; ++j) {
          if (mode == mode_b[j]) {
            found = true;
            break;
          }
        }
        if (!found) {  // is non-contracted mode
          mode_c[num_modes_c++] = mode;
          if (num_modes_c > kMaxNumModes_) {
            // too many modes
            return false;
          }
        }
      }
      return true;
    };

    std::array<char, kMaxNumModes_+1> implicit_mode_c;
    char* redirect_mode_c;
    if (is_implicit) {
      // we have to copy all non-contracted modes from a over to c
      if (CopyModesIf(mode_a, num_modes_a_, mode_b, num_modes_b_,
                      implicit_mode_c.data(), num_modes_c_) == false) {
        return;
      }
      // we have to copy all non-contracted modes from b over to c
      if (CopyModesIf(mode_b, num_modes_b_, mode_a, num_modes_a_,
                      implicit_mode_c.data(), num_modes_c_) == false) {
        return;
      }
      std::sort(implicit_mode_c.begin(), std::next(implicit_mode_c.begin(), num_modes_c_));
      // modes are sorted w.r.t. lexical order
      implicit_mode_c[num_modes_c_] = '\0';
      redirect_mode_c = implicit_mode_c.data();
    } else {
      redirect_mode_c = mode_c;
      num_modes_c_ = num_modes_c;
    }

    for (uint32_t i = 0; i < num_modes_a_; i++) {
      modes_a_[i] = mode_a[num_modes_a_ - i - 1];
      extent_a_[i] = a_shape[num_modes_a_ - i - 1];
    }
    for (uint32_t i = 0; i < num_modes_b_; i++) {
      modes_b_[i] = mode_b[num_modes_b_ - i - 1];
      extent_b_[i] = b_shape[num_modes_b_ - i - 1];
    }
    for (uint32_t i = 0; i < num_modes_c_; i++) {
      const auto mode = redirect_mode_c[num_modes_c_ - i - 1];
      modes_c_[i] = mode;
      bool found = false;
      for (uint32_t j=0; j < num_modes_a_; ++j) {
        if (modes_a_[j] == mode) {
          extent_c_[i] = extent_a_[j];
          found = true;
          break;
        }
      }
      for (uint32_t j=0; !found && j < num_modes_b_; ++j) {
        if (modes_b_[j] == mode) {
          extent_c_[i] = extent_b_[j];
          break;
        }
      }
    }

    is_initialized_ = true;
  }

  std::vector<IntType> GetOutputShape() const {
    if (!is_initialized_) return {};
    std::vector<IntType> extent_c(num_modes_c_);
    for (int i=0; i < num_modes_c_; ++i) {
      extent_c[i] = extent_c_.at(num_modes_c_ - i - 1);
    }
    return extent_c;
  }

  bool IsInitialized() const { return is_initialized_; }

  const int64_t* GetExtentsA() const { return extent_a_.data(); }
  const int64_t* GetExtentsB() const { return extent_b_.data(); }
  const int64_t* GetExtentsC() const { return extent_c_.data(); }

  const int* GetModesA() const { return modes_a_.data(); }
  const int* GetModesB() const { return modes_b_.data(); }
  const int* GetModesC() const { return modes_c_.data(); }

  int GetNumModesA() const { return num_modes_a_; }
  int GetNumModesB() const { return num_modes_b_; }
  int GetNumModesC() const { return num_modes_c_; }

 private:
  uint32_t num_modes_a_;
  uint32_t num_modes_b_;
  uint32_t num_modes_c_;
  bool is_initialized_;
  std::array<int, kMaxNumModes_> modes_a_;
  std::array<int, kMaxNumModes_> modes_b_;
  std::array<int, kMaxNumModes_> modes_c_;
  std::array<int64_t, kMaxNumModes_> extent_a_;
  std::array<int64_t, kMaxNumModes_> extent_b_;
  std::array<int64_t, kMaxNumModes_> extent_c_;
};

/*!
 * \brief The Operator used to perform einsum using cuTensor library.
 */
template<typename DType>
class CuTensorEinsum {
  STATIC_ASSERT_CUDNN_VERSION_GE(6000);
  static_assert(CUTENSOR_MAJOR >= 1 && CUTENSOR_MINOR >= 2 &&
                CUTENSOR_PATCH >= 0, "minimal cuTENSOR 1.2.0 is required.");

 public:
  CuTensorEinsum() {
  }
  ~CuTensorEinsum() {
  }

  size_t Init(std::string equation,
              const mxnet::ShapeVector& in_shape,
              const mxnet::ShapeVector& out_shape,
              const OpContext& ctx,
              size_t prev_workspace_size,
              size_t alignment) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(in_shape.size(), 2);
    CHECK_EQ(out_shape.size(), 1);

    constexpr cudaDataType_t cudaType = CuTensorTypeTraits<DType>::cudaType;
    constexpr cutensorComputeType_t cutensorType = CuTensorTypeTraits<DType>::cutensorType;
    constexpr cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    Einsum<DType, int, kMaxTensorRank> my_einsum(equation, in_shape[0], in_shape[1]);
    if (!my_einsum.IsInitialized()) {
        CUTENSOR_CALL(CUTENSOR_STATUS_NOT_SUPPORTED);
    }

    cutensorTensorDescriptor_t descriptor_a;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_a,
                                               my_einsum.GetNumModesA(),
                                               my_einsum.GetExtentsA(),
                                               NULL,  // stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descriptor_b;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_b,
                                               my_einsum.GetNumModesB(),
                                               my_einsum.GetExtentsB(),
                                               NULL,  // stride
                                               cudaType, CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descriptor_c;
    CUTENSOR_CALL(cutensorInitTensorDescriptor(&s->cutensor_handle_,
                                               &descriptor_c,
                                               my_einsum.GetNumModesC(),
                                               my_einsum.GetExtentsC(),
                                               NULL,  // stride
                                               cudaType,
                                               CUTENSOR_OP_IDENTITY));

    CUTENSOR_CALL(cutensorInitContractionDescriptor(
                  &s->cutensor_handle_,
                  &descriptor_contraction,
                  &descriptor_a, my_einsum.GetModesA(), alignment,
                  &descriptor_b, my_einsum.GetModesB(), alignment,
                  &descriptor_c, my_einsum.GetModesC(), alignment,
                  &descriptor_c, my_einsum.GetModesC(), alignment,
                  cutensorType));

    CUTENSOR_CALL(cutensorInitContractionFind(&s->cutensor_handle_,
                                              &find, algo));

    if (s->cutensor_cachelines_ != nullptr) {
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
    if (s->cutensor_cachelines_ == nullptr) {
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
               const std::vector<TBlob> &outputs,
               bool req_write,
               char* workspace) {
    Stream<gpu>* s = ctx.get_stream<gpu>();
    if (!req_write) {
      beta = 1.0;  // kAddTo
    }
    if (s->cutensor_cachelines_ != nullptr) {
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
                                      reinterpret_cast<void*>(&alpha),
                                      tensor_a_ptr, tensor_b_ptr,
                                      reinterpret_cast<void*>(&beta),
                                      tensor_c_ptr, tensor_c_ptr,
                                      my_workspace,
                                      my_workspace_size,
                                      s->stream_));
  }

  cutensorContractionDescriptor_t descriptor_contraction;  // strucutre of the contraction
  cutensorContractionPlan_t plan;  // encodes the execution plan
  cutensorContractionFind_t find;  // limits the search space (of viable candidates)

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
                        size_t pos_cutensor_op,
                        size_t temp_grad_size_aligned) {
    CHECK_EQ(in_shape.size(), 3U);
    CHECK_EQ(out_shape.size(), 2U);

    int comma_pos = equation.find(",");
    int arrow_pos = equation.find("->", comma_pos + 1);
    int len_op2 = arrow_pos - comma_pos - 1;
    const bool isImplicit = (arrow_pos == std::string::npos);
    std::string my_equation;
    if (isImplicit) {
      // get explicit equation
      Einsum<DType, int, kMaxTensorRank> my_einsum(equation, in_shape[1], in_shape[2]);
      if (!my_einsum.IsInitialized()) {
        CUTENSOR_CALL(CUTENSOR_STATUS_NOT_SUPPORTED);
      }
      const int* modes_c = my_einsum.GetModesC();
      my_equation = equation;
      my_equation.append("->");
      for (int i = my_einsum.GetNumModesC() - 1; i >= 0; --i) {
        my_equation += static_cast<char>(modes_c[i]);
      }
      arrow_pos = my_equation.find("->", comma_pos + 1);
      len_op2 = arrow_pos - comma_pos - 1;
    } else {
      my_equation = equation;
    }

    // gradient for first operand
    mxnet::ShapeVector grad_op1_input_shapes;
    mxnet::ShapeVector grad_op1_output_shapes;
    grad_op1_input_shapes.push_back(in_shape[0]);
    grad_op1_input_shapes.push_back(in_shape[2]);
    grad_op1_output_shapes.push_back(out_shape[0]);
    std::string grad_operand1_equation = my_equation.substr(arrow_pos + 2);
    grad_operand1_equation += ",";
    grad_operand1_equation += my_equation.substr(comma_pos + 1, len_op2);
    grad_operand1_equation += "->";
    grad_operand1_equation += my_equation.substr(0, comma_pos);
    bwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
    size_t req_workspace =
      bwd_cutensor_ops[pos_cutensor_op].Init(grad_operand1_equation,
                                             grad_op1_input_shapes,
                                             grad_op1_output_shapes,
                                             ctx,
                                             temp_grad_size_aligned,
                                             dptr_alignment);
    if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;

    // gradient for second operand
    mxnet::ShapeVector grad_op2_input_shapes;
    mxnet::ShapeVector grad_op2_output_shapes;
    grad_op2_input_shapes.push_back(in_shape[1]);
    grad_op2_input_shapes.push_back(in_shape[0]);
    grad_op2_output_shapes.push_back(out_shape[1]);
    std::string grad_operand2_equation = my_equation.substr(0, comma_pos);
    grad_operand2_equation += ",";
    grad_operand2_equation += my_equation.substr(arrow_pos + 2);
    grad_operand2_equation += "->";
    grad_operand2_equation += my_equation.substr(comma_pos + 1, len_op2);
    bwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
    req_workspace =
      bwd_cutensor_ops[pos_cutensor_op+1].Init(grad_operand2_equation,
                                               grad_op2_input_shapes,
                                               grad_op2_output_shapes,
                                               ctx,
                                               temp_grad_size_aligned,
                                               dptr_alignment);
    if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;
  }

  void Init(const EinsumOp& state,
            const std::vector<TBlob>& inputs,
            const mxnet::ShapeVector& in_shape,
            const mxnet::ShapeVector& out_shape,
            const OpContext& ctx,
            bool is_backward) {
    if (!is_backward) {
      std::vector<std::vector<int> > pos;
      std::string string_repr;
      mypaths = einsum_path(state.subscripts, inputs, true, ctx.run_ctx, &pos, &string_repr);
      paths_len = mypaths.size();
      max_workspace_cutensor = 0;
      mxnet::ShapeVector operands_shape(in_shape);
      for (int i = 0; i < paths_len; ++i) {
        bool handle_out = (i == paths_len - 1);
        mxnet::ShapeVector tmp_in_shape;
        mxnet::ShapeVector tmp_out_shape;
        // remove inds from right to left
        for (const int& p : mypaths[i].contract_inds) {
          tmp_in_shape.push_back(operands_shape[p]);
          operands_shape.erase(operands_shape.begin() + p);
        }
        if (handle_out) {
          tmp_out_shape.push_back(out_shape[0]);
        } else {
          tmp_out_shape.push_back(mypaths[i].oshape);
        }
        fwd_cutensor_ops.push_back(CuTensorEinsum<DType>());
        if (mypaths[i].do_cutensor) {
          size_t req_workspace = fwd_cutensor_ops[i].Init(mypaths[i].einsum_str,
                                                          tmp_in_shape,
                                                          tmp_out_shape,
                                                          ctx,
                                                          0, dptr_alignment);

          if (req_workspace > max_workspace_cutensor) max_workspace_cutensor = req_workspace;
        }
        if (i != paths_len - 1) {
          size_t new_aligned_mem_required = RoundToMultiple(mypaths[i].oshape.Size(),
                                                            dptr_alignment, sizeof(DType));
          temp_ouputs_size_aligned += new_aligned_mem_required;
        }
        if (!handle_out) {
          operands_shape.push_back(mypaths[i].oshape);
        }
      }
    } else {
      // backward
      std::vector<TBlob> temp_inputs;  // inputs ignoring grad
      for (int i = 1; i < inputs.size(); ++i) {
          temp_inputs.push_back(inputs[i]);
      }
      std::vector<std::vector<int> > pos;
      std::string string_repr;
      mypaths = einsum_path(state.subscripts, temp_inputs, true,
                            ctx.run_ctx, &pos, &string_repr);
      max_workspace_cutensor = 0;
      size_t pos_cutensor_bwd_op = 0;
      paths_len = mypaths.size();
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
          for (const int& p : mypaths[i - 1].contract_inds) {
            bwd_op_idx[i].erase(bwd_op_idx[i].begin() + p);
          }
          bwd_op_idx[i].push_back(-static_cast<int>(i - 1));
        }
      }
      // calculate amount mem for temporal grads
      for (int i = 0; i + 1 < paths_len; ++i) {
        size_t new_aligned_mem_required = RoundToMultiple(mypaths[i].oshape.Size(),
                                                          dptr_alignment, sizeof(DType));
        temp_grads_offsets.push_back(new_aligned_mem_required);
        temp_grads_size_aligned += new_aligned_mem_required;
      }
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
          temp_in_shape.push_back(mypaths[i].oshape);
        }
        for (auto p : mypaths[i].contract_inds) {
          int idx = bwd_op_idx[i][p];
          if (idx >= 1) {
            temp_in_shape.push_back(in_shape[idx]);
            temp_out_shape.push_back(out_shape[idx - 1]);
          } else {
            temp_in_shape.push_back(mypaths[-idx].oshape);
            temp_out_shape.push_back(mypaths[-idx].oshape);
          }
        }
        CHECK_EQ(temp_in_shape.size(), 3U);
        CHECK_EQ(temp_out_shape.size(), 2U);

        if (mypaths[i].do_cutensor) {
          InitCuTensorGrad(mypaths[i].einsum_str,
                           temp_in_shape, temp_out_shape,
                           ctx, pos_cutensor_bwd_op,
                           temp_grads_size_aligned);
          pos_cutensor_bwd_op = pos_cutensor_bwd_op + 2;
        }
      }
      total_workspace = max_workspace_cutensor/sizeof(DType) +
                        temp_grads_size_aligned;
    }
  }

  void Forward(const EinsumOp& state,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<TBlob>& outputs) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    auto req = kWriteTo;
    // cuTensor workspace
    Tensor<gpu, 1, char> cutensor_workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(max_workspace_cutensor), s);

    std::vector<TBlob> operands(inputs);
    std::vector<TBlob> tmp_operands;

    // temporal space shared with backward: stateful
    std::vector<TBlob> temp_space_vec(paths_len - 1);
    Tensor<gpu, 1, DType> temp_space = state.tempspace->data().FlatTo1D<gpu, DType>();
    size_t begin = 0;
    for (int i = 0; i < paths_len - 1; ++i) {
      TBlob tblob = TBlob(temp_space.Slice(begin, begin + mypaths[i].oshape.Size()));
      temp_space_vec[i] = tblob.reshape(mypaths[i].oshape);
      size_t aligned_mem_required = RoundToMultiple(mypaths[i].oshape.Size(),
                                                    dptr_alignment, sizeof(DType));
      begin = begin + aligned_mem_required;
    }
    for (int i = 0; i < paths_len; ++i) {
      bool handle_out = (i == paths_len - 1);
      tmp_operands.clear();
      // remove inds from right to left
      for (const int& p : mypaths[i].contract_inds) {
        tmp_operands.push_back(operands[p]);
        operands.erase(operands.begin() + p);
      }
      if (mypaths[i].do_cutensor) {
        fwd_cutensor_ops[i].Compute(ctx, tmp_operands,
                                    handle_out ? outputs :
                                                 std::vector<TBlob>{temp_space_vec[i]},
                                    req,
                                    cutensor_workspace.dptr_);
      } else {
        // special cases do not use cuTensor: diagonal, trace,
        // implicit summation, dimension collapse, broadcasting
        NumpyEinsumProcess<gpu, 0>(tmp_operands,
        std::vector<OpReqType>{OpReqType::kWriteTo},
        handle_out ? outputs : std::vector<TBlob>{temp_space_vec[i]},
        mypaths[i].einsum_str.c_str(), tmp_operands.size(), ctx);
      }
      if (!handle_out) {
        operands.push_back(temp_space_vec[i]);
      }
    }
  }

  void ComputeGradients(const std::string &equation,
                        const std::vector<TBlob> &inputs,
                        const std::vector<TBlob> &outputs,
                        const std::vector<OpReqType> &req,
                        const OpContext &ctx,
                        size_t pos_cutensor_op,
                        Tensor<gpu, 1, DType> *workspace) {
    char* workspace_ptr = reinterpret_cast<char*>(workspace->dptr_);
    // gradient for first operand
    std::vector<TBlob> grad_operand1_inputs;
    std::vector<TBlob> grad_operand1_outputs;
    grad_operand1_inputs.push_back(inputs[0]);
    grad_operand1_inputs.push_back(inputs[2]);
    grad_operand1_outputs.push_back(outputs[0]);
    bwd_cutensor_ops[pos_cutensor_op].Compute(ctx,
                                              grad_operand1_inputs,
                                              grad_operand1_outputs,
                                              req[0] == kWriteTo,
                                              workspace_ptr);
    // gradient for second operand
    std::vector<TBlob> grad_operand2_inputs;
    std::vector<TBlob> grad_operand2_outputs;
    grad_operand2_inputs.push_back(inputs[1]);
    grad_operand2_inputs.push_back(inputs[0]);
    grad_operand2_outputs.push_back(outputs[1]);
    bwd_cutensor_ops[pos_cutensor_op+1].Compute(ctx,
                                                grad_operand2_inputs,
                                                grad_operand2_outputs,
                                                req[1] == kWriteTo,
                                                workspace_ptr);
  }

  void Backward(const EinsumOp& state,
                const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<TBlob>& outputs,
                const std::vector<OpReqType> &req) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t pos_cutensor_op = 0;

    // outputs from forward pass, no need to be re-computed, take from state
    Tensor<gpu, 1, DType> ndarray_space = state.tempspace->data().FlatTo1D<gpu, DType>();
    std::vector<TBlob> temp_data(paths_len - 1);
    size_t begin = 0;
    for (int i = 0; i < paths_len -1; ++i) {
      TBlob tblob = TBlob(ndarray_space.Slice(begin, begin + mypaths[i].oshape.Size()));
      temp_data[i] = tblob.reshape(mypaths[i].oshape);
      size_t aligned_mem_required = RoundToMultiple(mypaths[i].oshape.Size(),
                                                    dptr_alignment, sizeof(DType));
      begin = begin + aligned_mem_required;
    }
    // workspace (temporal grad + cuTensor)
    std::vector<TBlob> temp_grad(paths_len - 1);
    Tensor<gpu, 1, DType> temp_space =
      ctx.requested[0].get_space_typed<gpu, 1, DType>(Shape1(total_workspace), s);
    begin = 0;
    for (int i = 0; i < paths_len - 1; ++i) {
      TBlob tblob = TBlob(temp_space.Slice(begin, begin + mypaths[i].oshape.Size()));
      temp_grad[i] = tblob.reshape(mypaths[i].oshape);
      begin = begin + temp_grads_offsets[i];
    }
    // go through the paths in the reversed order
    std::vector<TBlob> temp_inputs, temp_outputs;
    std::vector<OpReqType> temp_req;
    for (int i = paths_len - 1; i >= 0; i--) {
      temp_inputs.clear();
      temp_outputs.clear();
      temp_req.clear();
      bool handle_out = (i == paths_len - 1);
      if (handle_out) {
        // grad_out
        temp_inputs.push_back(inputs[0]);
      } else {
        temp_inputs.push_back(temp_grad[i]);
      }
      for (auto p : mypaths[i].contract_inds) {
        int idx = bwd_op_idx[i][p];
        if (idx >= 1) {
          temp_inputs.push_back(inputs[idx]);
          temp_outputs.push_back(outputs[idx - 1]);
          temp_req.push_back(req[idx - 1]);
        } else {
          temp_inputs.push_back(temp_data[-idx]);
          temp_outputs.push_back(temp_grad[-idx]);
          temp_req.push_back(OpReqType::kWriteTo);
        }
      }
      CHECK_EQ(temp_inputs.size(), 3U);
      CHECK_EQ(temp_outputs.size(), 2U);
      CHECK_EQ(temp_req.size(), 2U);

      if (mypaths[i].do_cutensor) {
        ComputeGradients(mypaths[i].einsum_str,
                         temp_inputs, temp_outputs, temp_req,
                         ctx, pos_cutensor_op,
                         &temp_space);
        pos_cutensor_op = pos_cutensor_op + 2;
      } else {
        // special cases do not use cuTensor: diagonal, trace,
        // implicit summation, dimension collapse, broadcasting
        NumpyEinsumProcess<gpu, 1>(temp_inputs, temp_req, temp_outputs,
                                   mypaths[i].einsum_str.c_str(),
                                   temp_outputs.size(),
                                   ctx);
      }
    }
  }
  int paths_len = 0;
  std::vector<Step> mypaths;
  // cutensor ops for the forward and backward passs:
  std::vector<CuTensorEinsum<DType>> fwd_cutensor_ops;
  std::vector<CuTensorEinsum<DType>> bwd_cutensor_ops;
  std::vector<std::vector<int> > bwd_op_idx;

  const size_t dptr_alignment = 128;
  size_t temp_ouputs_size_aligned = 0;  // temporal outputs saved in FWD
  size_t temp_grads_size_aligned = 0;
  std::vector<size_t> temp_grads_offsets;
  size_t max_workspace_cutensor = 0;
  size_t total_workspace = 0;
};

typedef ParamOpSign<EinsumOp> EinsumSignature;
template<typename DType>
static EinsumOpGPU<DType>& GetEinsumOpGPU(const EinsumOp& state,
                                          const std::vector<TBlob>& inputs,
                                          const std::vector<TBlob>& outputs,
                                          const std::vector<OpReqType> &req,
                                          const OpContext& ctx,
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
  mxnet::ShapeVector in_shape(inputs.size());
  mxnet::ShapeVector out_shape(outputs.size());
  for (size_t i = 0; i < in_shape.size(); i++)
    in_shape[i] = inputs[i].shape_;
  for (size_t i = 0; i < out_shape.size(); i++)
    out_shape[i] = outputs[i].shape_;
  EinsumSignature key(state);
  size_t ndim = 0;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(ndim +  // for in and out shapes
              1 +  // for dev_id
              req.size() +
              1);  // for is_backward
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(ctx.run_ctx.ctx.dev_id);
  for (int i = 0; i < req.size(); ++i) {
    key.AddSign(req[i]);
  }
  key.AddSign(is_backward ? 1 : 0);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<EinsumOpGPU<DType>> op(new EinsumOpGPU<DType>());
    auto ins_ret = ops.insert(std::pair<EinsumSignature, std::shared_ptr<EinsumOpGPU<DType>>>(
                              key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(state, inputs,
                     in_shape, out_shape,
                     ctx, is_backward);
  }
  return *it->second;
}

bool IsCutensorCompatible(const EinsumOp state,
                          const std::vector<TBlob>& inputs,
                          const std::vector<TBlob>& outputs,
                          const OpContext& ctx) {
  if (state.num_args <= 1) return false;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (!_tensordot_type_check(inputs[i].type_flag_, ctx.run_ctx))
      return false;
    for (size_t j = 0; j < inputs[i].ndim(); j++) {
      if (inputs[i].size(j) <= 0) return false;
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (!_tensordot_type_check(outputs[i].type_flag_, ctx.run_ctx))
      return false;
    for (size_t j = 0; j < outputs[i].ndim(); j++) {
      if (outputs[i].size(j) <= 0) return false;
    }
  }
  return true;
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
  bool use_cutensor = IsCutensorCompatible(state, inputs, outputs, ctx);
  if (!use_cutensor) {
    NumpyEinsumForward<gpu>(state_ptr, ctx, inputs, req, outputs);
  } else {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      EinsumOpGPU<DType> &op = GetEinsumOpGPU<DType>
          (state, inputs, outputs,
           req, ctx, false);
      state.tempspace.reset<NDArray>(new NDArray(TShape(Shape1(op.temp_ouputs_size_aligned)),
                                                 ctx.run_ctx.ctx, false,
                                                 outputs[0].type_flag_));
      op.Forward(state, ctx, inputs, outputs);
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
  bool use_cutensor = IsCutensorCompatible(state, inputs, outputs, ctx);
  if (!use_cutensor) {
    NumpyEinsumBackward<gpu>(state_ptr, ctx, inputs, req, outputs);
  } else {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      EinsumOpGPU<DType> &op = GetEinsumOpGPU<DType>
          (state, inputs, outputs,
           req, ctx, true);
      op.Backward(state, ctx, inputs, outputs, req);
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

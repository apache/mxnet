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
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
#include <cub/cub.cuh>
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief Compute the number of elements of every row.
 */
struct SliceMarkCsrIndPtr {
  /*!
   * \brief
   * \param i           the i-th row of the output csr ndarray
   * \param prefix_sum  indptr array of the output csr ndarray
   * \param in_idx      indices array of the input csr ndarray
   * \param in_indptr   indptr array of the input csr ndarray
   * \param begin_col   starting indice
   * \param end_col     ending indice
   */
  template <typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i,
                                  RType* prefix_sum,
                                  const IType* in_idx,
                                  const RType* in_indptr,
                                  const int begin_col,
                                  const int end_col) {
    if (i == 0) {
      prefix_sum[0] = 0;
    }
    RType size = 0;
    for (RType j = in_indptr[i]; j < in_indptr[i + 1]; j++) {
      // indices of CSRNDArray are in ascending order per row
      if (in_idx[j] >= end_col) {
        break;
      } else if (in_idx[j] >= begin_col) {
        size++;
      }
    }
    prefix_sum[i + 1] = size;
  }
};

template <>
void SliceDimTwoCsrImpl<gpu>(const mxnet::TShape& begin,
                             const mxnet::TShape& end,
                             const OpContext& ctx,
                             const NDArray& in,
                             const NDArray& out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;

  Stream<gpu>* s = ctx.get_stream<gpu>();

  nnvm::dim_t begin_row = begin[0], end_row = end[0];
  nnvm::dim_t begin_col = begin[1], end_col = end[1];
  nnvm::dim_t indptr_len = end_row - begin_row + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        RType* in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        IType* in_idx    = in.aux_data(kIdx).dptr<IType>();
        DType* in_data   = in.data().dptr<DType>();

        RType* out_indptr = out.aux_data(kIndPtr).dptr<RType>();

        Kernel<SliceMarkCsrIndPtr, gpu>::Launch(
            s, indptr_len - 1, out_indptr, in_idx, in_indptr + begin_row, begin_col, end_col);
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
        d_temp_storage = workspace.dptr_;

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        // retrieve nnr
        RType nnr = 0;
        CUDA_CALL(cudaMemcpyAsync(&nnr,
                                  &out_indptr[indptr_len - 1],
                                  sizeof(RType),
                                  cudaMemcpyDeviceToHost,
                                  mshadow::Stream<gpu>::GetStream(s)));
        CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));

        // returns zeros in csr format if nnr = 0
        if (nnr == 0) {
          out.set_aux_shape(kIdx, Shape1(0));
          return;
        }
        out.CheckAndAllocAuxData(kIdx, Shape1(nnr));
        out.CheckAndAllocData(Shape1(nnr));
        IType* out_idx  = out.aux_data(kIdx).dptr<IType>();
        DType* out_data = out.data().dptr<DType>();

        Kernel<SliceDimTwoCsrAssign, gpu>::Launch(s,
                                                  indptr_len - 1,
                                                  out_idx,
                                                  out_data,
                                                  out_indptr,
                                                  in_idx,
                                                  in_data,
                                                  in_indptr + begin_row,
                                                  begin_col,
                                                  end_col);
      });
    });
  });
}

template <typename DType>
struct split_tensor_data {
  static const int MaxSections = 128;
  size_t num_sections;
  DType* outputs[MaxSections];
  size_t indices[MaxSections + 1];
  DType* inputs[1];
};

template <bool split_last_axis, typename LType, typename DType>
__global__ void split_tensor_kernel(size_t input_size,
                                    const split_tensor_data<DType> params,
                                    size_t split_axis_size,
                                    size_t tail_size,
                                    size_t last_axis_size,
                                    size_t blocks_last_axis) {
  const int entries_per_load = sizeof(LType) / sizeof(DType);
  const LType* in_aligned    = reinterpret_cast<const LType*>(params.inputs[0]);
  const size_t last_axis_size_aligned =
      entries_per_load > 0 ? last_axis_size / entries_per_load : last_axis_size;
  if (split_last_axis) {
    size_t input_offset_leading = (blockIdx.x / blocks_last_axis) * last_axis_size_aligned;
    size_t position_last_axis   = (blockIdx.x % blocks_last_axis) * blockDim.x * entries_per_load +
                                params.indices[0] + threadIdx.x * entries_per_load;
    if (position_last_axis < params.indices[params.num_sections]) {
      size_t position_last_axis_aligned =
          entries_per_load > 0 ? position_last_axis / entries_per_load : position_last_axis;
      LType input_data = in_aligned[input_offset_leading + position_last_axis_aligned];
      // Binary search to find section of each thread
      size_t lower = 0;
      size_t upper = params.num_sections - 1;
      while (lower < upper) {
        size_t mid = (lower + upper + 1) / 2;
        if (position_last_axis >= params.indices[mid])
          lower = mid;
        else
          upper = mid - 1;
      }
      size_t section      = upper;
      size_t section_size = params.indices[section + 1] - params.indices[section];
      LType* out_aligned  = reinterpret_cast<LType*>(params.outputs[section]);
      size_t section_size_aligned =
          entries_per_load > 0 ? section_size / entries_per_load : section_size;
      size_t index_aligned = entries_per_load > 0 ? params.indices[section] / entries_per_load :
                                                    params.indices[section];
      size_t output_offset_leading = (blockIdx.x / blocks_last_axis) * section_size_aligned;
      size_t output_position = output_offset_leading + position_last_axis_aligned - index_aligned;
      out_aligned[output_position] = input_data;
    }
  } else {
    size_t split_axis_size_iter   = params.indices[params.num_sections] - params.indices[0];
    size_t blocks_per_leading_dim = (split_axis_size_iter * tail_size * blocks_last_axis);
    // input offsets: leading (axes pre-split-axis), at split-axis, tail, and blocks_last_axis
    size_t input_offset_leading = (blockIdx.x / blocks_per_leading_dim) * split_axis_size *
                                  tail_size * last_axis_size_aligned;
    size_t pos_in_split_axis =
        (blockIdx.x / (tail_size * blocks_last_axis)) % split_axis_size_iter + params.indices[0];
    size_t input_offset_split_axis = pos_in_split_axis * tail_size * last_axis_size_aligned;
    size_t offset_tail  = ((blockIdx.x / blocks_last_axis) % tail_size) * last_axis_size_aligned;
    size_t input_offset = input_offset_leading + input_offset_split_axis + offset_tail +
                          (blockIdx.x % blocks_last_axis) * blockDim.x;
    // Binary search to find section for this block
    size_t lower = 0;
    size_t upper = params.num_sections - 1;
    while (lower < upper) {
      size_t mid = (lower + upper + 1) / 2;
      if (pos_in_split_axis >= params.indices[mid])
        lower = mid;
      else
        upper = mid - 1;
    }
    size_t section      = upper;
    size_t section_size = params.indices[section + 1] - params.indices[section];
    LType* out_aligned  = reinterpret_cast<LType*>(params.outputs[section]);
    // output offsets: leading (axes pre-split-axis), at split-axis,and blocks_last_axis
    size_t output_offset_leading =
        (blockIdx.x / blocks_per_leading_dim) * section_size * tail_size * last_axis_size_aligned;
    size_t output_offset_split_axis =
        ((blockIdx.x % blocks_per_leading_dim) / blocks_last_axis -
         ((params.indices[section] - params.indices[0]) * tail_size)) *
        last_axis_size_aligned;
    size_t output_offset = output_offset_leading + output_offset_split_axis +
                           (blockIdx.x % blocks_last_axis) * blockDim.x;
    if (threadIdx.x < last_axis_size_aligned) {
      LType input_data                         = in_aligned[input_offset + threadIdx.x];
      out_aligned[output_offset + threadIdx.x] = input_data;
    }
  }
}

template <typename DType>
int get_load_type_split(size_t last_axis_size,
                        bool splitting_last_axis,
                        size_t n_sections,
                        size_t* indices) {
  using namespace mshadow;
  int sections_largest_multiple = 8;
  if (splitting_last_axis) {
    for (size_t i = 0; i < n_sections; ++i) {
      size_t size_section = indices[i + 1] - indices[i];
      if (size_section * sizeof(DType) % 8)
        sections_largest_multiple = std::min(sections_largest_multiple, 4);
      if (size_section * sizeof(DType) % 4)
        sections_largest_multiple = std::min(sections_largest_multiple, 2);
      if (size_section * sizeof(DType) % 2)
        sections_largest_multiple = std::min(sections_largest_multiple, 1);
    }
  }
  if (last_axis_size * sizeof(DType) % 8 == 0 && sections_largest_multiple == 8) {
    return kFloat64;
  } else if (last_axis_size * sizeof(DType) % 4 == 0 && sections_largest_multiple >= 4) {
    return kFloat32;
  } else if (last_axis_size * sizeof(DType) % 2 == 0 && sections_largest_multiple >= 2) {
    return kFloat16;
  } else {
    return kUint8;
  }
}

inline void SplitOpForwardGPU(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), (param.sections > 0) ? param.sections : param.indices.ndim());
  const TBlob& input_data = inputs[split_enum::kData];
  int real_axis           = param.axis;
  if (real_axis < 0) {
    real_axis += input_data.ndim();
  }
  size_t last_axis_size  = input_data.shape_[inputs[0].ndim() - 1];
  size_t split_axis_size = input_data.shape_[real_axis];
  size_t tail_size       = 1;  // does not include last dim
  for (int i = real_axis + 1; i < input_data.ndim() - 1; ++i) {
    tail_size *= input_data.shape_[i];
  }
  if (last_axis_size < 128) {
    // custom kernel will not be efficient with less than 128 elemnts in last axis
    SplitOpForwardImpl<gpu>(attrs, ctx, inputs, req, outputs, real_axis);
  } else {
    Stream<gpu>* s = ctx.get_stream<gpu>();
    CHECK_LT(real_axis, input_data.ndim());
    const mxnet::TShape& ishape = input_data.shape_;
    const mxnet::TShape split_pts =
        (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
    std::vector<size_t> indices;
    for (const auto& split_pos : split_pts) {
      indices.push_back(split_pos);
    }
    if (param.sections == 0) {
      indices.push_back(ishape[real_axis]);
    }
    size_t n_sections        = indices.size() - 1;
    bool splitting_last_axis = (real_axis == inputs[0].ndim() - 1);

    for (size_t sections_processed = 0; sections_processed < n_sections;) {
      size_t remaining_sections = n_sections - sections_processed;
      MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
        // set parameters
        split_tensor_data<DType> params{};
        params.num_sections = std::min<size_t>(remaining_sections, params.MaxSections);
        params.inputs[0]    = input_data.dptr<DType>();
        for (size_t i = 0; i < params.num_sections; ++i) {
          params.outputs[i] = outputs[sections_processed + i].dptr<DType>();
          params.indices[i] = indices[sections_processed + i];
        }
        params.indices[params.num_sections] = indices[sections_processed + params.num_sections];
        // load type: we need to check that last axis size is multiple of ltype
        // and if splitting_last_axis, all section sizes as well
        int ltype = get_load_type_split<DType>(
            last_axis_size, splitting_last_axis, params.num_sections, params.indices);
        MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
          CHECK_LE(sizeof(DType), sizeof(LType));
          const size_t entries_per_load = sizeof(LType) / sizeof(DType);
          size_t block_size             = 32;
          size_t max_threads_block      = 256;
          size_t last_axis_elements =
              entries_per_load > 0 ? (last_axis_size / entries_per_load) : 0;
          if (splitting_last_axis) {
            // may not be possible to include whole axis if too many sections
            last_axis_elements =
                entries_per_load > 0 ?
                    ((params.indices[params.num_sections] - params.indices[0]) / entries_per_load) :
                    0;
          }
          while (block_size < last_axis_elements && (block_size < max_threads_block)) {
            block_size += 32;
          }
          size_t blocks_last_axis = (last_axis_elements + block_size - 1) / block_size;
          size_t n_blocks         = blocks_last_axis;
          for (int i = 0; i < input_data.ndim() - 1; ++i) {
            if (i == real_axis) {
              // may not be possible to include all sections if too many
              n_blocks *= (params.indices[params.num_sections] - params.indices[0]);
            } else {
              n_blocks *= input_data.shape_[i];
            }
          }
          if (splitting_last_axis) {
            split_tensor_kernel<true, LType>
                <<<n_blocks, block_size, 0, s->stream_>>>(input_data.Size(),
                                                          params,
                                                          split_axis_size,
                                                          tail_size,
                                                          last_axis_size,
                                                          blocks_last_axis);
          } else {
            split_tensor_kernel<false, LType>
                <<<n_blocks, block_size, 0, s->stream_>>>(input_data.Size(),
                                                          params,
                                                          split_axis_size,
                                                          tail_size,
                                                          last_axis_size,
                                                          blocks_last_axis);
          }
        });
        sections_processed += params.num_sections;
      });
    }
  }
}

NNVM_REGISTER_OP(Reshape).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose).set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
    .set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", SliceEx<gpu>);

NNVM_REGISTER_OP(_backward_slice).set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign).set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
    .set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis).set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis).set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(slice_like).set_attr<FCompute>("FCompute<gpu>", SliceLikeForward<gpu>);

NNVM_REGISTER_OP(_backward_slice_like).set_attr<FCompute>("FCompute<gpu>", SliceLikeBackward<gpu>);

NNVM_REGISTER_OP(clip)
    .set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip).set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat).set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat).set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile).set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile).set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack).set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack).set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);

NNVM_REGISTER_OP(squeeze).set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_backward_squeeze)
    .set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(depth_to_space).set_attr<FCompute>("FCompute<gpu>", DepthToSpaceOpForward<gpu>);

NNVM_REGISTER_OP(space_to_depth).set_attr<FCompute>("FCompute<gpu>", SpaceToDepthOpForward<gpu>);

NNVM_REGISTER_OP(_split_v2)
    // Incompatible due to Copy(xpu_tensor, cpu_tensor) in SplitOpForwardImpl
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", SplitOpForwardGPU);

NNVM_REGISTER_OP(_split_v2_backward)
    // Incompatible due to Copy(xpu_tensor, cpu_tensor) in SplitOpBackwardImpl
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", SplitOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet

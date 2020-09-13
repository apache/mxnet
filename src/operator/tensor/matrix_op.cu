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
 *  Copyright (c) 2015 by Contributors
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
  template<typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i,
                                  RType* prefix_sum,
                                  const IType* in_idx,
                                  const RType* in_indptr,
                                  const int begin_col, const int end_col) {
    if (i == 0) {
      prefix_sum[0] = 0;
    }
    RType size = 0;
    for (RType j = in_indptr[i]; j < in_indptr[i+1]; j++) {
      // indices of CSRNDArray are in ascending order per row
      if (in_idx[j] >= end_col) {
        break;
      } else if (in_idx[j] >= begin_col) {
        size++;
      }
    }
    prefix_sum[i+1] = size;
  }
};


template<>
void SliceDimTwoCsrImpl<gpu>(const mxnet::TShape &begin, const mxnet::TShape &end,
                             const OpContext& ctx, const NDArray &in, const NDArray &out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;

  Stream<gpu> *s = ctx.get_stream<gpu>();

  nnvm::dim_t begin_row = begin[0], end_row = end[0];
  nnvm::dim_t begin_col = begin[1], end_col = end[1];
  nnvm::dim_t indptr_len = end_row - begin_row + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        RType *in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        IType *in_idx = in.aux_data(kIdx).dptr<IType>();
        DType *in_data = in.data().dptr<DType>();

        RType *out_indptr = out.aux_data(kIndPtr).dptr<RType>();

        Kernel<SliceMarkCsrIndPtr, gpu>::Launch(s, indptr_len - 1,
                                                out_indptr,
                                                in_idx,
                                                in_indptr + begin_row,
                                                begin_col, end_col);
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
        d_temp_storage = workspace.dptr_;

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        // retrieve nnr
        RType nnr = 0;
        CUDA_CALL(cudaMemcpyAsync(&nnr, &out_indptr[indptr_len-1], sizeof(RType),
                                  cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
        CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));

        // returns zeros in csr format if nnr = 0
        if (nnr == 0) {
          out.set_aux_shape(kIdx, Shape1(0));
          return;
        }
        out.CheckAndAllocAuxData(kIdx, Shape1(nnr));
        out.CheckAndAllocData(Shape1(nnr));
        IType *out_idx = out.aux_data(kIdx).dptr<IType>();
        DType *out_data = out.data().dptr<DType>();

        Kernel<SliceDimTwoCsrAssign, gpu>::Launch(s, indptr_len - 1, out_idx, out_data,
                                                  out_indptr, in_idx, in_data,
                                                  in_indptr + begin_row,
                                                  begin_col, end_col);
      });
    });
  });
}

constexpr size_t split_max_sections = 128;
template <typename DType>
struct split_tensor_data {
  size_t num_sections;
  DType* outputs[split_max_sections];
  size_t indices[split_max_sections];
  DType* inputs[1];
};
    
template <bool split_last_axis, typename LType, typename DType>
__global__ void split_tensor_kernel(size_t input_size,
                                    const split_tensor_data<DType> params,
                                    size_t axis_size,
                                    size_t last_dim,
                                    size_t trailing) {
  const int entries_per_load = sizeof(LType)/sizeof(DType);
  /*extern __shared__ int position2section[];
  //initialize position2section
  if (threadIdx.x==0) {
    size_t section = 0;
    for (size_t i=0; i<params.num_sections; ++i) {
      size_t start = params.indices[i];
      size_t end = params.indices[i+1];
      if (split_last_axis) {
        start = entries_per_load > 0 ? start / entries_per_load: start;
        end = entries_per_load > 0 ? end / entries_per_load: end;
      }
      for (int j=start; j<end; ++j) {
        position2section[j] = section;
      }
      section++;
    }
  }
  __syncthreads();*/

  // silence warning division by 0
  const int axis_size_aligned =  entries_per_load > 0 ? axis_size / entries_per_load : axis_size;
  const int last_dim_aligned =  entries_per_load > 0 ? last_dim / entries_per_load : 0;
  const LType* in_aligned = reinterpret_cast<const LType*>(params.inputs[0]);
  size_t input_offset = blockIdx.x * last_dim_aligned;
  if (split_last_axis) {
    for (index_t i = threadIdx.x; i < axis_size_aligned; i += blockDim.x) {
      LType input_data = in_aligned[input_offset+i];
      size_t section = 0;
      size_t section_size = params.indices[1] - params.indices[0];
      for (; section < params.num_sections && params.indices[section+1] <= i*entries_per_load;) {
        section++;
        section_size = params.indices[section+1] - params.indices[section];
      }
      //size_t section = position2section[i];
      //size_t section_size = params.indices[section+1] - params.indices[section];
      LType* out_aligned = reinterpret_cast<LType*>(params.outputs[section]);
      // silence warning division by 0
      size_t section_size_aligned = entries_per_load > 0 ? section_size / entries_per_load :
                                                         section_size;
      size_t index_aligned = entries_per_load > 0 ? params.indices[section] / entries_per_load :
                                                    params.indices[section];
      size_t output_position = blockIdx.x * section_size_aligned + i - index_aligned;
      out_aligned[output_position] = input_data;
    }
  } else {
    size_t position_in_axis = (blockIdx.x / trailing) % axis_size;
    size_t section = 0;
    size_t section_size = params.indices[1] - params.indices[0];
    for (; section < params.num_sections && params.indices[section+1] <= position_in_axis;) {
      section++;
      section_size = params.indices[section+1] - params.indices[section];
    }
    //size_t section = position2section[position_in_axis];
    //size_t section_size = params.indices[section+1] - params.indices[section];
    LType* out_aligned = reinterpret_cast<LType*>(params.outputs[section]);
    size_t head_id = blockIdx.x / (trailing * axis_size);
    size_t head_module = blockIdx.x % (trailing * axis_size);
    size_t offset_head_sector = head_module - (params.indices[section] * trailing);
    size_t position_in_sector = (head_id * section_size * trailing +
                                 offset_head_sector) * last_dim_aligned;
    for (index_t i = threadIdx.x; i < last_dim_aligned; i += blockDim.x) {
      LType input_data = in_aligned[input_offset + i];
      out_aligned[position_in_sector + i] = input_data;
    }
  }
}

template <typename DType>
int get_load_type_split(size_t last_dim,
                        bool splitting_last_axis,
                        std::vector<size_t> indices) {
  using namespace mshadow;
  int sectors_largest_multiple = 8;
  if(splitting_last_axis) {
    for(size_t i = 0; i < indices.size()-1; ++i){
      size_t size_sector = indices[i+1] - indices[i];
      if (size_sector * sizeof(DType) % 8)
        sectors_largest_multiple = std::min(sectors_largest_multiple, 4);
      if (size_sector * sizeof(DType) % 4)
        sectors_largest_multiple = std::min(sectors_largest_multiple, 2);
      if (size_sector * sizeof(DType) % 2)
        sectors_largest_multiple = std::min(sectors_largest_multiple, 1);
    }
  }
  if (last_dim * sizeof(DType) % 8 == 0 && sectors_largest_multiple == 8) {
    return kFloat64;
  } else if (last_dim * sizeof(DType) % 4 == 0 && sectors_largest_multiple >= 4) {
    return kFloat32;
  } else if (last_dim * sizeof(DType) % 2 == 0 && sectors_largest_multiple >= 2) {
    return kFloat16;
  } else {
    return kUint8;
  }
}

template<>
inline void SplitOpForwardImpl<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs,
                                    const int real_axis) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  const TBlob& input_data = inputs[split_enum::kData];
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
  size_t axis_size = input_data.shape_[real_axis];

  if (outputs.size() > split_max_sections) {
    size_t trailing = 1;
    for (int i = real_axis + 1; i < input_data.ndim(); ++i) {
      trailing *= input_data.shape_[i];
    }
    size_t workspace_size = 0;
    workspace_size += indices.size() * sizeof(size_t);
    MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
      std::vector<DType*> output_data;
      for (const TBlob& data : outputs) {
        output_data.push_back(data.dptr<DType>());
      }
      workspace_size += output_data.size() * sizeof(DType*);
      Tensor<gpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
      Tensor<cpu, 1, size_t> indices_cpu_tensor(indices.data(), Shape1(indices.size()));
      Tensor<gpu, 1, size_t> indices_xpu_tensor(
        reinterpret_cast<size_t*>(workspace.dptr_), Shape1(indices.size()));
      Tensor<cpu, 1, DType*> ptrs_cpu_tensor(output_data.data(), Shape1(output_data.size()));
      Tensor<gpu, 1, DType*> ptrs_xpu_tensor(
        reinterpret_cast<DType**>(workspace.dptr_ + indices.size() * sizeof(size_t)),
        Shape1(output_data.size()));
      mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
      mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
      Kernel<SplitKernel, gpu>::Launch(
        s, input_data.Size(), input_data.dptr<DType>(), ptrs_xpu_tensor.dptr_,
        indices_xpu_tensor.dptr_, indices.size() - 1, axis_size, trailing);
    });

  } else {
    size_t trailing = 1;
    // trailing ignores last dimension
    for (int i = real_axis + 1; i < input_data.ndim()-1; ++i) {
      trailing *= input_data.shape_[i];
    }
    bool splitting_last_axis = (real_axis == inputs[0].ndim()-1);
    size_t last_dim = input_data.shape_[input_data.ndim()-1];
    MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
      // load type: we need to check that last axis size is multiple of ltype 
      // and if splitting_last_axis, all sector sizes as well 
      int ltype = get_load_type_split<DType>(last_dim, splitting_last_axis, indices);
      // set parameters
      split_tensor_data<DType> params{};
      params.num_sections = indices.size() - 1;
      params.inputs[0] = input_data.dptr<DType>();
      params.indices[0] = indices[0]; 
      for (int i=0; i < params.num_sections; i++) {
        params.outputs[i] = outputs[i].dptr<DType>();
        params.indices[i+1] = indices[i+1];
      }
      
      MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
        CHECK_LE(sizeof(DType), sizeof(LType));
        int nblocks = 1;
        // each block of threads computes one instance of last dimension
        for (int i = 0 ; i < input_data.ndim()-1; ++i) {
          nblocks *= input_data.shape_[i];
        }
        const int entries_per_load = sizeof(LType)/sizeof(DType);
        int block_size = 32;
        int max_threads_block = 512;
        size_t block_n_elems = entries_per_load > 0 ? (last_dim/entries_per_load): 0;
        while (block_size < block_n_elems && (block_size < max_threads_block))
          block_size += 32;
        //size_t required_shared = last_dim * sizeof(size_t) / entries_per_load;
        size_t required_shared = 0;
        if (splitting_last_axis) {
          split_tensor_kernel<true, LType><<<nblocks, block_size,
                              required_shared, s->stream_>>>
            (input_data.Size(), params, axis_size, last_dim, trailing);
        } else {
          split_tensor_kernel<false, LType><<<nblocks, block_size,
                              required_shared, s->stream_>>>
            (input_data.Size(), params, axis_size, last_dim, trailing);
        }
      });
    }); 
  }
}

NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SliceEx<gpu>);

NNVM_REGISTER_OP(_backward_slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(slice_like)
.set_attr<FCompute>("FCompute<gpu>", SliceLikeForward<gpu>);

NNVM_REGISTER_OP(_backward_slice_like)
.set_attr<FCompute>("FCompute<gpu>", SliceLikeBackward<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);

NNVM_REGISTER_OP(squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_backward_squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(depth_to_space)
.set_attr<FCompute>("FCompute<gpu>", DepthToSpaceOpForward<gpu>);

NNVM_REGISTER_OP(space_to_depth)
.set_attr<FCompute>("FCompute<gpu>", SpaceToDepthOpForward<gpu>);

NNVM_REGISTER_OP(_split_v2)
.set_attr<FCompute>("FCompute<gpu>", SplitOpForward<gpu>);

NNVM_REGISTER_OP(_split_v2_backward)
.set_attr<FCompute>("FCompute<gpu>", SplitOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet

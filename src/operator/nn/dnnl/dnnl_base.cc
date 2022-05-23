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

#if MXNET_USE_ONEDNN == 1

#include <atomic>

#include "../../../common/exec_utils.h"
#include "operator/operator_common.h"
#include "dnnl_base-inl.h"

namespace mxnet {

DNNLStream* DNNLStream::Get() {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local DNNLStream stream;
#else
  static MX_THREAD_LOCAL DNNLStream stream;
#endif
  return &stream;
}

namespace op {
void DNNLMemorySum(const dnnl::memory& arr1, const dnnl::memory& arr2, const dnnl::memory& out) {
  std::vector<dnnl::memory::desc> input_pds(2);
  std::vector<float> scales(2, 1);
  input_pds[0] = arr1.get_desc();
  input_pds[1] = arr2.get_desc();
  CHECK(input_pds[0] == input_pds[0]);
  const dnnl::memory* in_mem1 = &arr1;
  const dnnl::memory* in_mem2 = &arr2;
  auto output_pd              = out.get_desc();
  if (input_pds[0] != output_pd) {
    auto tmp_memory1 = TmpMemMgr::Get()->Alloc(output_pd);
    auto tmp_memory2 = TmpMemMgr::Get()->Alloc(output_pd);
    DNNLMemoryCopy(arr1, tmp_memory1);
    DNNLMemoryCopy(arr2, tmp_memory2);
    input_pds[0] = tmp_memory1->get_desc();
    input_pds[1] = tmp_memory2->get_desc();
    in_mem1      = tmp_memory1;
    in_mem2      = tmp_memory2;
  }
  dnnl::sum::primitive_desc sum_pd(output_pd, scales, input_pds, CpuEngine::Get()->get_engine());
  dnnl_args_map_t args = {
      {DNNL_ARG_MULTIPLE_SRC, *in_mem1},
      {DNNL_ARG_MULTIPLE_SRC + 1, *in_mem2},
      {DNNL_ARG_DST, out},
  };
  DNNLStream::Get()->RegisterPrimArgs(dnnl::sum(sum_pd), args);
}
}  // namespace op

void* AlignMem(void* mem, size_t size, size_t alignment, size_t* space) {
  if (size > *space)
    return nullptr;
  intptr_t addr = reinterpret_cast<intptr_t>(mem);
  // If the address has been aligned, don't do anything.
  intptr_t last_chunk = addr % alignment;
  if (last_chunk == 0)
    return mem;
  intptr_t padding = alignment - last_chunk;
  // If the buffer doesn't have enough space, we should return null here.
  if (padding + size > *space)
    return nullptr;
  addr += padding;
  *space -= padding;
  CHECK_EQ(addr % alignment, 0);
  return reinterpret_cast<void*>(addr);
}

dnnl::memory* TmpMemMgr::Alloc(const dnnl::memory::desc& md) {
  // We need to include the size of the memory used for alignment.
  this->est_size += md.get_size() + alignment;
  void* mem = AlignMem(this->curr_mem, md.get_size(), alignment, &this->curr_size);
  if (mem) {
    // The memory is allocated from the temporary memory space in the
    // operator. It'll only become invalid after we exit from the operator.
    dnnl_mem_ptr ret(new dnnl::memory(md, CpuEngine::Get()->get_engine(), mem));
    DNNLStream::Get()->RegisterMem(ret);
    CHECK_EQ(mem, mem);
    this->curr_size -= md.get_size();
    this->curr_mem = static_cast<char*>(mem) + md.get_size();
    return ret.get();
  } else {
    // If curr_mem has been initialized and we still reach here, it means the current
    // allocated memory isn't enough. But it doesn't matter for multiple invokes of a
    // operator, as the TmpMemMgr could estimate the space at the first iteration and
    // then re-requests abundant space from MXNet resource. DNNL could allocate
    // the space by itself. Thus, we just let it continue for estimating the maximum
    // required space size. It will be allocated at next call.
    if (this->curr_mem && dmlc::GetEnv("MXNET_ONEDNN_DEBUG", false)) {
      LOG(WARNING) << "oneDNN debug message: The rest of the temporary space is not "
                   << "adequate for allocating " << md.get_size() << " bytes. Thus, oneDNN "
                   << "allocate the space by itself.";
    }
    dnnl_mem_ptr ret(new dnnl::memory(md, CpuEngine::Get()->get_engine()));
    DNNLStream::Get()->RegisterMem(ret);
    return ret.get();
  }
}

void DNNLMemoryCopy(const dnnl::memory& mem, const dnnl::memory* this_mem) {
  DNNLStream* stream                = DNNLStream::Get();
  dnnl::memory::desc from_desc      = mem.get_desc();
  dnnl::memory::desc this_desc      = this_mem->get_desc();
  dnnl_format_tag_t from_def_format = GetDefaultFormat(from_desc);
  dnnl_format_tag_t this_def_format = GetDefaultFormat(this_desc);

  if (!same_shape(this_desc, from_desc) && IsDefaultFormat(from_desc)) {
    // In this case, we can simply create a new DNNL memory for the required
    // shape.
    dnnl::memory::dims dims(this_desc.data.dims, this_desc.data.dims + this_desc.data.ndims);
    auto this_dtype = static_cast<dnnl::memory::data_type>(this_desc.data.data_type);
    dnnl::memory::desc data_md(
        dims, this_dtype, static_cast<dnnl::memory::format_tag>(this_def_format));

    dnnl_mem_ptr tmp_mem(new dnnl::memory(data_md, mem.get_engine(), mem.get_data_handle()));
    stream->RegisterMem(tmp_mem);
    std::unordered_map<int, dnnl::memory> args(
        {{DNNL_ARG_FROM, *tmp_mem}, {DNNL_ARG_TO, *this_mem}});
    stream->RegisterPrimArgs(dnnl::reorder(*tmp_mem, *this_mem), args);
  } else if (!same_shape(this_desc, from_desc)) {
    // In this case, the source memory stores data in a customized layout. We
    // need to reorganize the data in memory before we can reshape.
    dnnl::memory::desc def_desc = GetDesc(from_desc, from_def_format);
    dnnl::memory* def_mem       = TmpMemMgr::Get()->Alloc(def_desc);
    std::unordered_map<int, dnnl::memory> args({{DNNL_ARG_FROM, mem}, {DNNL_ARG_TO, *def_mem}});
    stream->RegisterPrimArgs(dnnl::reorder(mem, *def_mem), args);

    // Now we can reshape it
    dnnl_mem_ptr tmp_mem(new dnnl::memory(this_desc, mem.get_engine(), def_mem->get_data_handle()));
    stream->RegisterMem(tmp_mem);
    args = {{DNNL_ARG_FROM, *tmp_mem}, {DNNL_ARG_TO, *this_mem}};
    stream->RegisterPrimArgs(dnnl::reorder(*tmp_mem, *this_mem), args);
  } else if (this_desc == from_desc) {
    std::unordered_map<int, dnnl::memory> args({{DNNL_ARG_FROM, mem}, {DNNL_ARG_TO, *this_mem}});
    // If the layout is the same, we can just copy data.
    stream->RegisterPrimArgs(dnnl::reorder(mem, *this_mem), args);
  } else {
    // If both are not using the default layouts. There isn't much we can do,
    // other than reorder data layout directly.
    if (!IsDefaultFormat(this_desc) && !IsDefaultFormat(from_desc)) {
      std::unordered_map<int, dnnl::memory> args({{DNNL_ARG_FROM, mem}, {DNNL_ARG_TO, *this_mem}});
      stream->RegisterPrimArgs(dnnl::reorder(mem, *this_mem), args);
    } else if (IsDefaultFormat(this_desc)) {
      // If the dest mem uses the default memory layout, we can simply use
      // the default format of the source memory to improve perf of reorder.
      dnnl::memory::desc desc = GetDesc(from_desc, from_def_format);
      dnnl_mem_ptr tmp_mem(new dnnl::memory(desc, mem.get_engine(), this_mem->get_data_handle()));
      stream->RegisterMem(tmp_mem);
      std::unordered_map<int, dnnl::memory> args({{DNNL_ARG_FROM, mem}, {DNNL_ARG_TO, *tmp_mem}});
      stream->RegisterPrimArgs(dnnl::reorder(mem, *tmp_mem), args);
    } else {
      // If the src mem uses the default memory layout, we can use
      // the default format of the source memory to improve perf.
      dnnl::memory::desc desc = GetDesc(this_desc, this_def_format);
      dnnl_mem_ptr tmp_mem(new dnnl::memory(desc, this_mem->get_engine(), mem.get_data_handle()));
      stream->RegisterMem(tmp_mem);
      std::unordered_map<int, dnnl::memory> args(
          {{DNNL_ARG_FROM, *tmp_mem}, {DNNL_ARG_TO, *this_mem}});
      stream->RegisterPrimArgs(dnnl::reorder(*tmp_mem, *this_mem), args);
    }
  }
}

bool CanWriteTo(const NDArray& out_arr, const NDArray& in_arr, const dnnl::memory::desc& desc) {
  auto in_mem     = in_arr.GetDNNLData();
  bool add_same   = in_mem->get_data_handle() == out_arr.GetDNNLData()->get_data_handle();
  bool pdesc_same = out_arr.GetDNNLData()->get_desc() == desc && in_mem->get_desc() == desc;
  return add_same && pdesc_same;
}

dnnl_output_t CreateDNNLMem(const NDArray& out_arr,
                            const dnnl::memory::desc& desc,
                            OpReqType req,
                            const NDArray* in_arr) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return dnnl_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req && in_arr != nullptr && CanWriteTo(out_arr, *in_arr, desc)) {
    dnnl::memory* mem = const_cast<NDArray&>(out_arr).CreateDNNLData(&desc);
    // mem is nullptr if out_arr is view and desc is DNNL format.
    // need to Reorder2Default before calling CreateDNNLMem
    CHECK(mem != nullptr);
    return dnnl_output_t(OutDataOp::Noop, mem);
  } else if (kWriteInplace == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return dnnl_output_t(OutDataOp::CopyBack, tmp);
  } else if (kWriteTo == req) {
    dnnl::memory* mem = const_cast<NDArray&>(out_arr).CreateDNNLData(&desc);
    if (nullptr == mem) {
      auto tmp = TmpMemMgr::Get()->Alloc(desc);
      return dnnl_output_t(OutDataOp::CopyBack, tmp);
    }
    return dnnl_output_t(OutDataOp::Noop, mem);
  }
  auto tmp = TmpMemMgr::Get()->Alloc(desc);
  return dnnl_output_t(OutDataOp::Noop, tmp);
}

dnnl_output_t CreateDNNLWeightGrad(const NDArray& out_arr,
                                   const dnnl::memory::desc& desc,
                                   OpReqType req) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return dnnl_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return dnnl_output_t(OutDataOp::CopyBack, tmp);
  } else {
    dnnl::memory* mem = nullptr;
    if (IsDefaultFormat(desc)) {
      mem = const_cast<NDArray&>(out_arr).CreateDNNLData(&desc);
    }
    if (mem == nullptr) {
      auto tmp = TmpMemMgr::Get()->Alloc(desc);
      return dnnl_output_t(OutDataOp::CopyBack, tmp);
    } else {
      return dnnl_output_t(OutDataOp::Noop, mem);
    }
  }
}

void CommitOutput(const NDArray& arr, const dnnl_output_t& res) {
  if (res.first == CopyBack) {
    const_cast<NDArray&>(arr).CopyFrom(*res.second);
  } else if (res.first == AddBack) {
    auto res_memory = res.second;
    auto target_pd  = arr.GetDNNLData()->get_desc();
    auto res_desc   = res.second->get_desc();
    auto mem        = arr.GetDNNLData(&res_desc);
    if (mem == nullptr) {
      auto tmp_memory = TmpMemMgr::Get()->Alloc(target_pd);
      DNNLMemoryCopy(*res_memory, tmp_memory);
      res_memory = tmp_memory;
      mem        = arr.GetDNNLData();
    }
    op::DNNLMemorySum(*mem, *res_memory, *mem);
  }
}

const dnnl::memory* GetWeights(const NDArray& arr, int num_groups) {
  const auto type = get_dnnl_type(arr.dtype());
  auto tz         = dnnl::memory::dims{0};
  auto format_tag = dnnl::memory::format_tag::undef;
  auto engine     = CpuEngine::Get()->get_engine();
  const int ndim  = arr.shape().ndim();
  int O = 0, I = 1, H = 2, W = 3;
  int D = -1;
  if (ndim == 5) {
    D = 2;
    H = 3;
    W = 4;
  }
  if (ndim == 2) {
    tz         = dnnl::memory::dims{arr.shape()[O], arr.shape()[I]};
    format_tag = dnnl::memory::format_tag::oi;
  } else if (ndim == 3) {
    tz = num_groups > 1 ?
             dnnl::memory::dims{
                 num_groups, arr.shape()[O] / num_groups, arr.shape()[I], arr.shape()[H]} :
             dnnl::memory::dims{arr.shape()[O], arr.shape()[I], arr.shape()[H]};
    format_tag = num_groups > 1 ? dnnl::memory::format_tag::goiw : dnnl::memory::format_tag::oiw;
  } else if (ndim == 4) {
    tz = num_groups > 1 ?
             dnnl::memory::dims{num_groups,
                                arr.shape()[O] / num_groups,
                                arr.shape()[I],
                                arr.shape()[H],
                                arr.shape()[W]} :
             dnnl::memory::dims{arr.shape()[O], arr.shape()[I], arr.shape()[H], arr.shape()[W]};
    format_tag = num_groups > 1 ? dnnl::memory::format_tag::goihw : dnnl::memory::format_tag::oihw;
  } else if (ndim == 5) {
    tz = num_groups > 1 ?
             dnnl::memory::dims{num_groups,
                                arr.shape()[O] / num_groups,
                                arr.shape()[I],
                                arr.shape()[D],
                                arr.shape()[H],
                                arr.shape()[W]} :
             dnnl::memory::dims{
                 arr.shape()[O], arr.shape()[I], arr.shape()[D], arr.shape()[H], arr.shape()[W]};
    format_tag =
        num_groups > 1 ? dnnl::memory::format_tag::goidhw : dnnl::memory::format_tag::oidhw;
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
  }
  const auto md = dnnl::memory::desc{tz, type, format_tag};
  return arr.GetDNNLData(&md);
}

const dnnl::memory* GetWeights(const NDArray& arr,
                               const dnnl::memory::desc& target_desc,
                               int num_groups) {
  const dnnl::memory* mem = arr.GetDNNLData(&target_desc);
  // If the weight array already uses the target layout, simply return it directly.
  if (mem)
    return mem;
  mem = GetWeights(arr, num_groups);
  if (mem == nullptr)
    mem = arr.GetDNNLDataReorder(&target_desc);
  if (mem->get_desc() == target_desc)
    return mem;

  auto ret = TmpMemMgr::Get()->Alloc(target_desc);
  std::unordered_map<int, dnnl::memory> args({{DNNL_ARG_FROM, *mem}, {DNNL_ARG_TO, *ret}});
  DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*mem, *ret), args);
  return ret;
}

// default: block and dims' stride increase monotonically
// dnnl: 1.winograd 2.rnn packed 3. block and dims'stride is not increase monotonically
bool IsDNNL(const dnnl::memory::desc& desc) {
  bool rslt = true;
  if (desc.data.format_kind == dnnl_blocked) {
    if (desc.data.format_desc.blocking.inner_nblks == 0) {
      int i = 0;
      for (i = 0; i < desc.data.ndims - 1; i++) {
        if (desc.data.format_desc.blocking.strides[i] <
            desc.data.format_desc.blocking.strides[i + 1]) {
          break;
        }
      }
      if (i == desc.data.ndims - 1) {
        rslt = false;
      }
    }
  }
  return rslt;
}

dnnl_format_tag_t GetDefaultFormat(int num_dims) {
  switch (num_dims) {
    case 1:
      return dnnl_a;
    case 2:
      return dnnl_ab;
    case 3:
      return dnnl_abc;
    case 4:
      return dnnl_abcd;
    case 5:
      return dnnl_abcde;
    case 6:
      return dnnl_abcdef;
    case 7:
      return dnnl_abcdefg;
    case 8:
      return dnnl_abcdefgh;
    case 9:
      return dnnl_abcdefghi;
    case 10:
      return dnnl_abcdefghij;
    case 11:
      return dnnl_abcdefghijk;
    case 12:
      return dnnl_abcdefghijkl;
    default:
      LOG(FATAL) << "Not implemented dimension (" << num_dims << ") for oneDNN";
      return dnnl_format_tag_undef;
  }
}

dnnl_format_tag_t GetPermutedFormat(int num_dims) {
  switch (num_dims) {
    case 1:
      return dnnl_a;
    case 2:
      return dnnl_ba;
    case 3:
      return dnnl_acb;
    case 4:
      return dnnl_abdc;
    case 5:
      return dnnl_abced;
    case 6:
      return dnnl_abcdfe;
    case 7:
      return dnnl_abcdegf;
    case 8:
      return dnnl_abcdefhg;
    case 9:
      return dnnl_abcdefgih;
    case 10:
      return dnnl_abcdefghji;
    case 11:
      return dnnl_abcdefghikj;
    case 12:
      return dnnl_abcdefghijlk;
    default:
      LOG(FATAL) << "Not implemented dimension (" << num_dims << ") for oneDNN";
      return dnnl_format_tag_undef;
  }
}

dnnl_format_tag_t GetDefaultFormat(const dnnl::memory::desc& desc) {
  return GetDefaultFormat(desc.data.ndims);
}

bool IsDefaultFormat(const dnnl::memory::desc& desc) {
  bool rslt = false;
  if (desc.data.format_kind == dnnl_blocked) {
    if (desc.data.format_desc.blocking.inner_nblks == 0) {
      int i = 0;
      for (i = 0; i < desc.data.ndims - 1; i++) {
        if (desc.data.format_desc.blocking.strides[i] <
            desc.data.format_desc.blocking.strides[i + 1]) {
          break;
        }
      }
      if (i == desc.data.ndims - 1) {
        rslt = true;
      }
    }
  }
  return rslt;
}

dnnl::memory::desc GetDesc(const dnnl::memory::desc& desc, const dnnl_format_tag_t& format) {
  dnnl::memory::dims dims(desc.data.ndims);
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = desc.data.dims[i];
  dnnl::memory::format_tag cpp_format = static_cast<dnnl::memory::format_tag>(format);
  dnnl::memory::data_type cpp_type    = static_cast<dnnl::memory::data_type>(desc.data.data_type);
  return dnnl::memory::desc(dims, cpp_type, cpp_format);
}

// reorder dnnl src to dst format dtype
void ReorderTo(const dnnl::memory* src, const dnnl::memory* dst) {
  dnnl::stream s(CpuEngine::Get()->get_engine());
  auto new_src = *src;
  auto new_dst = *dst;
  dnnl::reorder(new_src, new_dst).execute(s, new_src, new_dst);
}

template <typename Compute, typename AttrState>
void FallBackCompute(Compute fn,
                     const AttrState& attrs_states,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  std::vector<TBlob> in_blobs(inputs.size());
  std::vector<NDArray> in_bufs;
  std::vector<OpReqType> new_req = req;
  for (size_t i = 0; i < in_blobs.size(); i++) {
    // If the input data isn't stored in the default format, we shouldn't
    // call data() directly, which will change the layout of the NDArray.
    // Instead, we should save the converted data in another NDArray.
    // TODO(zhengda) we should use temp space to save the converted data.
    if (inputs[i].IsDefaultData() && inputs[i].dtype() != mshadow::kBfloat16) {
      in_blobs[i] = inputs[i].data();
    } else {
      if (in_bufs.empty())
        in_bufs.reserve(inputs.size());
      if (inputs[i].dtype() != mshadow::kBfloat16) {
        in_bufs.push_back(inputs[i].Reorder2Default());
      } else {
        in_bufs.push_back(inputs[i].Reorder2DefaultFloatFormat());
      }
      in_blobs[i] = in_bufs.back().data();
    }
  }
  DNNLStream::Get()->Submit();

  std::vector<TBlob> out_blobs(outputs.size());
  std::vector<NDArray> temp_src, temp_dst;
  std::vector<NDArray> temp_bf16_src, temp_bf16_dst;
  for (size_t i = 0; i < out_blobs.size(); i++) {
    NDArray output = outputs[i];
    // for bf16, fisrt change it to f32
    if (outputs[i].dtype() == mshadow::kBfloat16) {
      NDArray temp = outputs[i].Reorder2DefaultFloatFormat();
      temp_bf16_src.emplace_back(temp);
      temp_bf16_dst.emplace_back(outputs[i]);
      output = temp;
      if (req[i] == kWriteInplace) {
        new_req[i] = kWriteTo;
      }
    } else {
      // ensure output does not use dnnl mem.
      // for inplace, we already converted & copied input above.
      if ((req[i] == kWriteTo) || (req[i] == kWriteInplace)) {
        const_cast<NDArray&>(output).InvalidateDNNLData();
        if (req[i] == kWriteInplace) {
          new_req[i] = kWriteTo;
        }
      } else if (req[i] == kAddTo && output.IsDNNLData()) {
        NDArray temp = outputs[i].Reorder2Default();
        temp_src.emplace_back(temp);
        temp_dst.emplace_back(outputs[i]);
        output = temp;
      }
    }
    CHECK(output.IsDefaultData());
    out_blobs[i] = output.data();
  }
  fn(attrs_states, ctx, in_blobs, new_req, out_blobs);
  for (size_t i = 0, bf16_pos = 0; i < out_blobs.size(); i++) {
    if (outputs[i].dtype() == mshadow::kBfloat16) {
      auto src_mem = temp_bf16_src[bf16_pos].GetDNNLData();
      auto dst_mem = temp_bf16_dst[bf16_pos].GetDNNLData();
      bf16_pos++;
      ReorderTo(src_mem, dst_mem);
    } else if (req[i] == kAddTo && outputs[i].IsDNNLData()) {
      mxnet::common::CastNonDefaultStorage(temp_src, temp_dst, ctx, false);
    }
  }
}

template <typename DType>
void print_diff(const mxnet::NDArray& arr1, const mxnet::NDArray& arr2) {
  DType* data1 = reinterpret_cast<DType*>(arr1.data().dptr_);
  DType* data2 = reinterpret_cast<DType*>(arr2.data().dptr_);
  for (size_t i = 0; i < arr1.shape().Size(); i++)
    std::cout << data1[i] - data2[i] << ", ";
  std::cout << std::endl;
}

template <typename DType>
static bool SimilarArray(const mxnet::NDArray& arr1,
                         const mxnet::NDArray& arr2,
                         DType rtol,
                         DType atol) {
  if (arr1.shape().Size() != arr2.shape().Size())
    return false;

  // This function should be used outside an DNNL operator.
  // There shouldn't be any operators in the stream.
  CHECK(!DNNLStream::Get()->HasOps());
  // We need to reorder data in the arrays to the default layout.
  // But we shouldn't reorder data in the original array.
  NDArray buf1, buf2;
  if (arr1.IsDNNLData()) {
    buf1     = NDArray(arr1.shape(), arr1.ctx(), false, arr1.dtype());
    auto mem = arr1.GetDNNLData();
    buf1.CopyFrom(*mem);
  }
  if (arr2.IsDNNLData()) {
    buf2     = NDArray(arr2.shape(), arr2.ctx(), false, arr2.dtype());
    auto mem = arr2.GetDNNLData();
    buf2.CopyFrom(*mem);
  }
  DNNLStream::Get()->Submit();

  DType* data1 =
      reinterpret_cast<DType*>(arr1.IsDNNLData() ? buf1.data().dptr_ : arr1.data().dptr_);
  DType* data2 =
      reinterpret_cast<DType*>(arr2.IsDNNLData() ? buf2.data().dptr_ : arr2.data().dptr_);
  std::atomic<bool> success(true);
#pragma omp parallel for
#ifdef _MSC_VER
  for (int64_t i = 0; i < arr1.shape().Size(); i++)
#else
  for (size_t i = 0; i < arr1.shape().Size(); i++)
#endif
  {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
    if (std::abs(data1[i] - data2[i]) > atol + rtol * std::abs(data2[i]))
      success.store(false);
#pragma clang diagnostic pop
  }
  return success.load();
}

template void FallBackCompute(void (*)(nnvm::NodeAttrs const&,
                                       OpContext const&,
                                       std::vector<TBlob, std::allocator<TBlob> > const&,
                                       std::vector<OpReqType, std::allocator<OpReqType> > const&,
                                       std::vector<TBlob, std::allocator<TBlob> > const&),
                              nnvm::NodeAttrs const&,
                              OpContext const&,
                              std::vector<NDArray, std::allocator<NDArray> > const&,
                              std::vector<OpReqType, std::allocator<OpReqType> > const&,
                              std::vector<NDArray, std::allocator<NDArray> > const&);

template void FallBackCompute(void (*)(OpStatePtr const&,
                                       OpContext const&,
                                       std::vector<TBlob, std::allocator<TBlob> > const&,
                                       std::vector<OpReqType, std::allocator<OpReqType> > const&,
                                       std::vector<TBlob, std::allocator<TBlob> > const&),
                              OpStatePtr const&,
                              OpContext const&,
                              std::vector<NDArray, std::allocator<NDArray> > const&,
                              std::vector<OpReqType, std::allocator<OpReqType> > const&,
                              std::vector<NDArray, std::allocator<NDArray> > const&);

void OpCheck::Init(const std::vector<mxnet::NDArray>& inputs_,
                   const std::vector<mxnet::NDArray>& outputs_) {
  auto ctx = inputs_[0].ctx();
  CHECK(!DNNLStream::Get()->HasOps());
  for (size_t i = 0; i < inputs_.size(); i++) {
    NDArray data = inputs_[i];
    inputs.emplace_back(data.shape(), ctx, false, data.dtype());
    if (data.IsDNNLData() && data.IsView())
      data = data.Reorder2Default();
    auto mem = data.GetDNNLData();
    inputs[i].CopyFrom(*mem);
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs.emplace_back(outputs_[i].shape(), ctx, false, outputs_[i].dtype());
    if (backward) {
      auto mem = outputs_[i].GetDNNLData();
      outputs[i].CopyFrom(*mem);
    }
  }
  DNNLStream::Get()->Submit();
}

void OpCheck::Run(mxnet::FCompute fn,
                  const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<mxnet::NDArray>& inputs_,
                  const std::vector<mxnet::OpReqType>& req,
                  const std::vector<mxnet::NDArray>& outputs_) {
  static auto& is_excluded = Op::GetAttr<bool>("TExcludeDNNLDebug");
  if (is_excluded.get(attrs.op, false)) {
    LOG(WARNING) << attrs.op->name << " not checked. TExcludeDNNLDebug flag present";
    return;
  }
  std::vector<mxnet::TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();
  std::vector<mxnet::TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  fn(attrs, ctx, in_blobs, req, out_blobs);
  if (dmlc::GetEnv("MXNET_ONEDNN_DEBUG", false))
    LOG(INFO) << "test " << attrs.op->name;
  size_t num = std::min(outputs.size(), outputs_.size());
  num        = std::min(num_checks, num);
  for (size_t i = 0; i < num; i++) {
    // We don't need to compare if it doesn't need to output data.
    if (req[i] == kNullOp)
      continue;
    MSHADOW_TYPE_SWITCH(outputs[i].dtype(), DType, {
      bool similar = SimilarArray<DType>(
          outputs[i], outputs_[i], static_cast<DType>(1e-2), static_cast<DType>(1e-2));
      if (!similar) {
        LOG(ERROR) << attrs.op->name << " fails";
      }
      CHECK(similar);
    });
  }
}

void OpCheck::CopyResult(const std::vector<mxnet::NDArray>& outputs_,
                         const std::vector<size_t>& indice) {
  CHECK(!DNNLStream::Get()->HasOps());
  auto non_const_outputs_ = const_cast<std::vector<mxnet::NDArray>&>(outputs_);
  for (auto i = indice.begin(); i != indice.end(); ++i) {
    auto mem = outputs[*i].GetDNNLData();
    non_const_outputs_[*i].CopyFrom(*mem);
  }
  DNNLStream::Get()->Submit();
}

bool DNNLStorageType(const nnvm::NodeAttrs& attrs,
                     const int dev_mask,
                     bool support_dnnl,
                     DispatchMode* dispatch_mode,
                     std::vector<int>* in_attrs,
                     std::vector<int>* out_attrs) {
  for (int& v : *in_attrs)
    if (v == -1)
      v = kDefaultStorage;

  DispatchMode wanted_mode;
#if MXNET_USE_ONEDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !DNNLEnvSet())
    wanted_mode = DispatchMode::kFComputeFallback;
  else if (dev_mask == mshadow::cpu::kDevMask && support_dnnl)
    wanted_mode = DispatchMode::kFComputeEx;
  else
#endif
    wanted_mode = DispatchMode::kFCompute;

  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched =
        op::storage_type_assign(out_attrs, mxnet::kDefaultStorage, dispatch_mode, wanted_mode);
  }
  if (!dispatched) {
    dispatched = op::dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

inline static const std::vector<NDArray> GetDNNLInputArray(const std::vector<NDArray>& inputs) {
  std::vector<NDArray> ret;
  ret.reserve(inputs.size());
  for (const auto& in : inputs) {
    if (in.IsView() && in.IsDNNLData()) {
      ret.push_back(in.Reorder2Default());
    } else {
      ret.push_back(in);
    }
  }
  return ret;
}

void DNNLRun(mxnet::FComputeEx fn,
             const nnvm::NodeAttrs& attrs,
             const mxnet::OpContext& ctx,
             const std::vector<mxnet::NDArray>& inputs,
             const std::vector<mxnet::OpReqType>& req,
             const std::vector<mxnet::NDArray>& outputs) {
  if (CheckDNNLInputArrayIsView(inputs)) {
    const auto dnnl_inputs = GetDNNLInputArray(inputs);
    fn(attrs, ctx, dnnl_inputs, req, outputs);
  } else {
    fn(attrs, ctx, inputs, req, outputs);
  }
}

void DNNLRun(FComputeExUnary fn,
             const nnvm::NodeAttrs& attrs,
             const mxnet::OpContext& ctx,
             const mxnet::NDArray& input,
             const mxnet::OpReqType& req,
             const mxnet::NDArray& output) {
  auto dnnl_input = input;
  if (input.IsView() && input.IsDNNLData()) {
    dnnl_input = input.Reorder2Default();
    fn(attrs, ctx, dnnl_input, req, output);
  } else {
    fn(attrs, ctx, input, req, output);
  }
}

}  // namespace mxnet

#endif

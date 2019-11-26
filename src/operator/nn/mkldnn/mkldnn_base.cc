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

#if MXNET_USE_MKLDNN == 1

#include <atomic>
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"
#include "../../../common/exec_utils.h"
#include "../../operator_common.h"

namespace mxnet {

MKLDNNStream *MKLDNNStream::Get() {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local MKLDNNStream stream;
#else
  static MX_THREAD_LOCAL MKLDNNStream stream;
#endif
  return &stream;
}

void *AlignMem(void *mem, size_t size, size_t alignment, size_t *space) {
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
  return reinterpret_cast<void *>(addr);
}

mkldnn::memory *TmpMemMgr::Alloc(const mkldnn::memory::desc &md) {
  // We need to include the size of the memory used for alignment.
  this->est_size += md.get_size() + alignment;
  void *mem = AlignMem(this->curr_mem, md.get_size(), alignment, &this->curr_size);
  if (mem) {
    // The memory is allocated from the temporary memory space in the
    // operator. It'll only become invalid after we exit from the operator.
    mkldnn_mem_ptr ret(new mkldnn::memory(md, CpuEngine::Get()->get_engine(), mem));
    MKLDNNStream::Get()->RegisterMem(ret);
    CHECK_EQ(mem, mem);
    this->curr_size -= md.get_size();
    this->curr_mem = static_cast<char *>(mem) + md.get_size();
    return ret.get();
  } else {
    // If curr_mem has been initialized and we still reach here, it means the current
    // allocated memory isn't enough. But it doesn't matter for multiple invokes of a
    // operator, as the TmpMemMgr could estimate the space at the first iteration and
    // then re-requests abundant space from MXNet resource. MKL-DNN could allocate
    // the space by itself. Thus, we just let it continue for estimating the maximum
    // required space size. It will be allocated at next call.
    if (this->curr_mem && dmlc::GetEnv("MXNET_MKLDNN_DEBUG", false)) {
      LOG(WARNING) << "mkl-dnn debug message: The rest of the temporary space is not "
          << "adequate for allocating " << md.get_size() << " bytes. Thus, mkl-dnn "
          << "allocate the space by itself.";
    }
    mkldnn_mem_ptr ret(new mkldnn::memory(md, CpuEngine::Get()->get_engine()));
    MKLDNNStream::Get()->RegisterMem(ret);
    return ret.get();
  }
}

void MKLDNNCopy(const mkldnn::memory &mem, const mkldnn::memory* this_mem) {
  MKLDNNStream *stream = MKLDNNStream::Get();
  mkldnn::memory::desc from_desc = mem.get_desc();
  mkldnn::memory::desc this_desc = this_mem->get_desc();
  mkldnn_format_tag_t from_def_format = GetDefaultFormat(from_desc);
  mkldnn_format_tag_t this_def_format = GetDefaultFormat(this_desc);

  if (!same_shape(this_desc, from_desc) && IsDefaultFormat(from_desc)) {
    // In this case, we can simply create a new MKLDNN memory for the required
    // shape.
    mkldnn::memory::dims dims(this_desc.data.dims,
                              this_desc.data.dims + this_desc.data.ndims);
    auto this_dtype = static_cast<mkldnn::memory::data_type>(this_desc.data.data_type);
    mkldnn::memory::desc data_md(dims, this_dtype,
                                 static_cast<mkldnn::memory::format_tag>(this_def_format));

    mkldnn_mem_ptr tmp_mem(new mkldnn::memory(data_md, mem.get_engine(), mem.get_data_handle()));
    stream->RegisterMem(tmp_mem);
    std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *tmp_mem},
                                                 {MKLDNN_ARG_TO, *this_mem}});
    stream->RegisterPrimArgs(mkldnn::reorder(*tmp_mem, *this_mem), args);
  } else if (!same_shape(this_desc, from_desc)) {
    // In this case, the source memory stores data in a customized layout. We
    // need to reorganize the data in memory before we can reshape.
    mkldnn::memory::desc def_desc = GetDesc(from_desc, from_def_format);
    mkldnn::memory *def_mem = TmpMemMgr::Get()->Alloc(def_desc);
    std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, mem},
                                                 {MKLDNN_ARG_TO, *def_mem}});
    stream->RegisterPrimArgs(mkldnn::reorder(mem, *def_mem), args);

    // Now we can reshape it
    mkldnn_mem_ptr tmp_mem(new mkldnn::memory(this_desc,
        mem.get_engine(), def_mem->get_data_handle()));
    stream->RegisterMem(tmp_mem);
    args = {{MKLDNN_ARG_FROM, *tmp_mem}, {MKLDNN_ARG_TO, *this_mem}};
    stream->RegisterPrimArgs(mkldnn::reorder(*tmp_mem, *this_mem), args);
} else if (this_desc == from_desc) {
    std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, mem},
                                                 {MKLDNN_ARG_TO, *this_mem}});
    // If the layout is the same, we can just copy data.
    stream->RegisterPrimArgs(mkldnn::reorder(mem, *this_mem), args);
} else {
    // If both are not using the default layouts. There isn't much we can do,
    // other than reorder data layout directly.
    if (!IsDefaultFormat(this_desc) && !IsDefaultFormat(from_desc)) {
      std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, mem},
                                                   {MKLDNN_ARG_TO, *this_mem}});
      stream->RegisterPrimArgs(mkldnn::reorder(mem, *this_mem), args);
    } else if (IsDefaultFormat(this_desc)) {
      // If the dest mem uses the default memory layout, we can simply use
      // the default format of the source memory to improve perf of reorder.
      mkldnn::memory::desc desc = GetDesc(from_desc, from_def_format);
      mkldnn_mem_ptr tmp_mem(new mkldnn::memory(desc,
          mem.get_engine(), this_mem->get_data_handle()));
      stream->RegisterMem(tmp_mem);
      std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, mem},
                                                   {MKLDNN_ARG_TO, *tmp_mem}});
      stream->RegisterPrimArgs(mkldnn::reorder(mem, *tmp_mem), args);
    } else {
      // If the src mem uses the default memory layout, we can use
      // the default format of the source memory to improve perf.
      mkldnn::memory::desc desc = GetDesc(this_desc, this_def_format);
      mkldnn_mem_ptr tmp_mem(new mkldnn::memory(desc,
          this_mem->get_engine(), mem.get_data_handle()));
      stream->RegisterMem(tmp_mem);
      std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *tmp_mem},
                                                   {MKLDNN_ARG_TO, *this_mem}});
      stream->RegisterPrimArgs(mkldnn::reorder(*tmp_mem, *this_mem), args);
    }
  }
}

bool CanWriteTo(const NDArray &out_arr,
                const NDArray &in_arr,
                const mkldnn::memory::desc &desc) {
  auto in_mem = in_arr.GetMKLDNNData();
  bool add_same = in_mem->get_data_handle() == out_arr.GetMKLDNNData()->get_data_handle();
  bool pdesc_same = out_arr.GetMKLDNNData()->get_desc() == desc &&
      in_mem->get_desc() == desc;
  return add_same && pdesc_same;
}

mkldnn_output_t CreateMKLDNNMem(const NDArray &out_arr,
                                const mkldnn::memory::desc &desc,
                                OpReqType req,
                                const NDArray* in_arr) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req && in_arr != nullptr && CanWriteTo(out_arr, *in_arr, desc)) {
    mkldnn::memory *mem = const_cast<NDArray &>(out_arr).CreateMKLDNNData(desc);
    // mem is nullptr if out_arr is view and desc is MKLDNN format.
    // need to Reorder2Default before calling CreateMKLDNNMem
    CHECK(mem != nullptr);
    return mkldnn_output_t(OutDataOp::Noop, mem);
  } else if (kWriteInplace == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::CopyBack, tmp);
  } else if (kWriteTo == req) {
    mkldnn::memory *mem = const_cast<NDArray &>(out_arr).CreateMKLDNNData(desc);
    if (nullptr == mem) {
      auto tmp = TmpMemMgr::Get()->Alloc(desc);
      return mkldnn_output_t(OutDataOp::CopyBack, tmp);
    }
    return mkldnn_output_t(OutDataOp::Noop, mem);
  }
  auto tmp = TmpMemMgr::Get()->Alloc(desc);
  return mkldnn_output_t(OutDataOp::Noop, tmp);
}

mkldnn_output_t CreateMKLDNNWeightGrad(const NDArray &out_arr,
                                       const mkldnn::memory::desc &desc,
                                       OpReqType req) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::CopyBack, tmp);
  } else {
    mkldnn::memory *mem = nullptr;
    if (IsDefaultFormat(desc)) {
      mem = const_cast<NDArray &>(out_arr).CreateMKLDNNData(desc);
    }
    if (mem == nullptr) {
      auto tmp = TmpMemMgr::Get()->Alloc(desc);
      return mkldnn_output_t(OutDataOp::CopyBack, tmp);
    } else {
      return mkldnn_output_t(OutDataOp::Noop, mem);
    }
  }
}

void CommitOutput(const NDArray &arr, const mkldnn_output_t &res) {
  if (res.first == CopyBack) {
    const_cast<NDArray &>(arr).CopyFrom(*res.second);
  } else if (res.first == AddBack) {
    auto res_memory = res.second;
    auto target_pd = arr.GetMKLDNNData()->get_desc();
    auto mem = arr.GetMKLDNNData(res.second->get_desc());
    if (mem == nullptr) {
      auto tmp_memory = TmpMemMgr::Get()->Alloc(target_pd);
      MKLDNNCopy(*res_memory, tmp_memory);
      res_memory = tmp_memory;
      mem = arr.GetMKLDNNData();
    }
    op::MKLDNNSum(*mem, *res_memory, *mem);
  }
}

const mkldnn::memory *GetWeights(const NDArray &arr, int num_groups) {
  const auto type = get_mkldnn_type(arr.dtype());
  auto tz = mkldnn::memory::dims{0};
  auto format_tag = mkldnn::memory::format_tag::undef;
  auto engine = CpuEngine::Get()->get_engine();
  const int O = 0, I = 1, H = 2, W = 3;
  if (arr.shape().ndim() == 2) {
    tz = mkldnn::memory::dims{static_cast<int>(arr.shape()[O]), static_cast<int>(arr.shape()[I])};
    format_tag = mkldnn::memory::format_tag::oi;
  } else if (arr.shape().ndim() == 3) {
    tz = num_groups > 1
             ? mkldnn::memory::dims{num_groups, static_cast<int>(arr.shape()[O] / num_groups),
                                    static_cast<int>(arr.shape()[I]),
                                    static_cast<int>(arr.shape()[H])}
             : mkldnn::memory::dims{static_cast<int>(arr.shape()[O]),
                                    static_cast<int>(arr.shape()[I]),
                                    static_cast<int>(arr.shape()[H])};
    format_tag = num_groups > 1 ? mkldnn::memory::format_tag::goiw
                                : mkldnn::memory::format_tag::oiw;
  } else if (arr.shape().ndim() == 4) {
    tz = num_groups > 1
             ? mkldnn::memory::dims{num_groups, static_cast<int>(arr.shape()[O] / num_groups),
                                    static_cast<int>(arr.shape()[I]),
                                    static_cast<int>(arr.shape()[H]),
                                    static_cast<int>(arr.shape()[W])}
             : mkldnn::memory::dims{
                   static_cast<int>(arr.shape()[O]), static_cast<int>(arr.shape()[I]),
                   static_cast<int>(arr.shape()[H]), static_cast<int>(arr.shape()[W])};
    format_tag = num_groups > 1 ? mkldnn::memory::format_tag::goihw
                                : mkldnn::memory::format_tag::oihw;
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
  }
  const auto md = mkldnn::memory::desc{tz, type, format_tag};
  return arr.GetMKLDNNData(md);
}

const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::memory::desc &target_desc, int num_groups) {
  const mkldnn::memory *mem = arr.GetMKLDNNData(target_desc);
  // If the weight array already uses the target layout, simply return it directly.
  if (mem) return mem;
  mem = GetWeights(arr, num_groups);
  if (mem == nullptr) mem = arr.GetMKLDNNDataReorder(target_desc);
  if (mem->get_desc() == target_desc) return mem;

  auto ret = TmpMemMgr::Get()->Alloc(target_desc);
  std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *mem},
                                               {MKLDNN_ARG_TO, *ret}});
  MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::reorder(*mem, *ret), args);
  return ret;
}


// default: block and dims' stride increase monotonically
// mkldnn: 1.winograd 2.rnn packed 3. block and dims'stride is not increase monotonically
bool IsMKLDNN(const mkldnn::memory::desc &desc) {
  bool rslt = true;
  if (desc.data.format_kind == mkldnn_blocked) {
    if (desc.data.format_desc.blocking.inner_nblks == 0) {
      int i = 0;
      for (i = 0; i < desc.data.ndims-1; i++) {
        if (desc.data.format_desc.blocking.strides[i]
            < desc.data.format_desc.blocking.strides[i + 1]) {
          break;
        }
      }
      if (i == desc.data.ndims-1) {
        rslt = false;
      }
    }
  }
  return rslt;
}

mkldnn_format_tag_t GetDefaultFormat(int num_dims) {
  switch (num_dims) {
    case 1: return mkldnn_a;
    case 2: return mkldnn_ab;
    case 3: return mkldnn_abc;
    case 4: return mkldnn_abcd;
    case 5: return mkldnn_abcde;
    case 6: return mkldnn_abcdef;
    default:
      LOG(FATAL) << "Not implemented dimension (" << num_dims << ") for MKLDNN";
      return mkldnn_format_tag_undef;
  }
}

mkldnn_format_tag_t GetDefaultFormat(const mkldnn::memory::desc &desc) {
  return GetDefaultFormat(desc.data.ndims);
}

bool IsDefaultFormat(const mkldnn::memory::desc &desc) {
  bool rslt = false;
  if (desc.data.format_kind == mkldnn_blocked) {
    if (desc.data.format_desc.blocking.inner_nblks == 0) {
      int i = 0;
      for (i = 0; i < desc.data.ndims-1; i++) {
        if (desc.data.format_desc.blocking.strides[i]
            < desc.data.format_desc.blocking.strides[i + 1]) {
          break;
        }
      }
      if (i == desc.data.ndims-1) {
        rslt = true;
      }
    }
  }
  return rslt;
}

mkldnn::memory::desc GetDesc(const mkldnn::memory::desc &desc,
                             const mkldnn_format_tag_t &format) {
  mkldnn::memory::dims dims(desc.data.ndims);
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = desc.data.dims[i];
  mkldnn::memory::format_tag cpp_format = static_cast<mkldnn::memory::format_tag>(format);
  mkldnn::memory::data_type cpp_type = static_cast<mkldnn::memory::data_type>(
      desc.data.data_type);
  mkldnn::memory::desc data_md(dims, cpp_type, cpp_format);
  return mkldnn::memory::desc(dims, cpp_type, cpp_format);
}

template <typename Compute, typename AttrState>
void FallBackCompute(Compute fn, const AttrState &attrs_states,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs) {
  std::vector<TBlob> in_blobs(inputs.size());
  std::vector<NDArray> in_bufs;
  std::vector<OpReqType> new_req = req;
  for (size_t i = 0; i < in_blobs.size(); i++) {
    // If the input data isn't stored in the default format, we shouldn't
    // call data() directly, which will change the layout of the NDArray.
    // Instead, we should save the converted data in another NDArray.
    // TODO(zhengda) we should use temp space to save the converted data.
    if (inputs[i].IsDefaultData()) {
      in_blobs[i] = inputs[i].data();
    } else {
      if (in_bufs.empty())
        in_bufs.reserve(inputs.size());
      in_bufs.push_back(inputs[i].Reorder2Default());
      in_blobs[i] = in_bufs.back().data();
    }
  }
  MKLDNNStream::Get()->Submit();

  std::vector<TBlob> out_blobs(outputs.size());
  std::vector<NDArray> temp_src, temp_dst;
  for (size_t i = 0; i < out_blobs.size(); i++) {
    NDArray output = outputs[i];
    // ensure output does not use mkldnn mem.
    // for inplace, we already converted & copied input above.
    if ((req[i] == kWriteTo) || (req[i] == kWriteInplace)) {
      const_cast<NDArray &>(output).InvalidateMKLDNNData();
      if (req[i] == kWriteInplace) {
        new_req[i] = kWriteTo;
      }
    } else if (req[i] == kAddTo && output.IsMKLDNNData()) {
      NDArray temp = outputs[i].Reorder2Default();
      temp_src.emplace_back(temp);
      temp_dst.emplace_back(outputs[i]);
      output = temp;
    }
    CHECK(output.IsDefaultData());
    out_blobs[i] = output.data();
  }

  fn(attrs_states, ctx, in_blobs, new_req, out_blobs);
  for (size_t i = 0; i < out_blobs.size(); i++) {
    if (req[i] == kAddTo && outputs[i].IsMKLDNNData())
      mxnet::common::CastNonDefaultStorage(temp_src, temp_dst, ctx, false);
  }
}

template<typename DType>
void print_diff(const mxnet::NDArray &arr1, const mxnet::NDArray &arr2) {
  DType *data1 = reinterpret_cast<DType *>(arr1.data().dptr_);
  DType *data2 = reinterpret_cast<DType *>(arr2.data().dptr_);
  for (size_t i = 0; i < arr1.shape().Size(); i++)
    std::cout << data1[i] - data2[i] << ", ";
  std::cout << std::endl;
}

template<typename DType>
static bool SimilarArray(const mxnet::NDArray &arr1, const mxnet::NDArray &arr2,
                         DType rtol, DType atol) {
  if (arr1.shape().Size() != arr2.shape().Size())
    return false;

  // This function should be used outside an MKLDNN operator.
  // There shouldn't be any operators in the stream.
  CHECK(!MKLDNNStream::Get()->HasOps());
  // We need to reorder data in the arrays to the default layout.
  // But we shouldn't reorder data in the original array.
  NDArray buf1, buf2;
  if (arr1.IsMKLDNNData()) {
    buf1 = NDArray(arr1.shape(), arr1.ctx(), false, arr1.dtype());
    auto mem = arr1.GetMKLDNNData();
    buf1.CopyFrom(*mem);
  }
  if (arr2.IsMKLDNNData()) {
    buf2 = NDArray(arr2.shape(), arr2.ctx(), false, arr2.dtype());
    auto mem = arr2.GetMKLDNNData();
    buf2.CopyFrom(*mem);
  }
  MKLDNNStream::Get()->Submit();

  DType *data1 = reinterpret_cast<DType *>(
      arr1.IsMKLDNNData() ? buf1.data().dptr_: arr1.data().dptr_);
  DType *data2 = reinterpret_cast<DType *>(
      arr2.IsMKLDNNData() ? buf2.data().dptr_: arr2.data().dptr_);
  std::atomic<bool> success(true);
#pragma omp parallel for
#ifdef _MSC_VER
  for (int64_t i = 0; i < arr1.shape().Size(); i++)
#else
  for (size_t i = 0; i < arr1.shape().Size(); i++)
#endif
  {
    if (std::abs(data1[i] - data2[i]) > atol + rtol * std::abs(data2[i]))
      success.store(false);
  }
  return success.load();
}

template void FallBackCompute(void (*)(nnvm::NodeAttrs const &, OpContext const &,
                                       std::vector<TBlob, std::allocator<TBlob> > const &,
                                       std::vector<OpReqType, std::allocator<OpReqType> > const &,
                                       std::vector<TBlob, std::allocator<TBlob> > const &),
                              nnvm::NodeAttrs const &, OpContext const &,
                              std::vector<NDArray, std::allocator<NDArray> > const &,
                              std::vector<OpReqType, std::allocator<OpReqType> > const &,
                              std::vector<NDArray, std::allocator<NDArray> > const &);

template void FallBackCompute(void (*)(OpStatePtr const &, OpContext const &,
                                       std::vector<TBlob, std::allocator<TBlob> > const &,
                                       std::vector<OpReqType, std::allocator<OpReqType> > const &,
                                       std::vector<TBlob, std::allocator<TBlob> > const &),
                              OpStatePtr const &, OpContext const &,
                              std::vector<NDArray, std::allocator<NDArray> > const &,
                              std::vector<OpReqType, std::allocator<OpReqType> > const &,
                              std::vector<NDArray, std::allocator<NDArray> > const &);

void OpCheck::Init(const std::vector<mxnet::NDArray> &inputs_,
                   const std::vector<mxnet::NDArray> &outputs_) {
  auto ctx = inputs_[0].ctx();
  CHECK(!MKLDNNStream::Get()->HasOps());
  for (size_t i = 0; i < inputs_.size(); i++) {
    NDArray data = inputs_[i];
    inputs.emplace_back(data.shape(), ctx, false, data.dtype());
    if (data.IsMKLDNNData() && data.IsView())
        data = data.Reorder2Default();
    auto mem = data.GetMKLDNNData();
    inputs[i].CopyFrom(*mem);
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs.emplace_back(outputs_[i].shape(), ctx,
                         false, outputs_[i].dtype());
    if (backward) {
      auto mem = outputs_[i].GetMKLDNNData();
      outputs[i].CopyFrom(*mem);
    }
  }
  MKLDNNStream::Get()->Submit();
}

void OpCheck::Run(mxnet::FCompute fn, const nnvm::NodeAttrs &attrs,
                  const mxnet::OpContext &ctx,
                  const std::vector<mxnet::NDArray> &inputs_,
                  const std::vector<mxnet::OpReqType> &req,
                  const std::vector<mxnet::NDArray> &outputs_) {
  static auto& is_excluded = Op::GetAttr<bool>("TExcludeMKLDNNDebug");
  if (is_excluded.get(attrs.op, false)) {
    LOG(WARNING) << attrs.op->name << " not checked. TExcludeMKLDNNDebug flag present";
    return;
  }
  std::vector<mxnet::TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) in_blobs[i] = inputs[i].data();
  std::vector<mxnet::TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  fn(attrs, ctx, in_blobs, req, out_blobs);
  if (dmlc::GetEnv("MXNET_MKLDNN_DEBUG", false))
    LOG(INFO) << "test " << attrs.op->name;
  size_t num = std::min(outputs.size(), outputs_.size());
  num = std::min(num_checks, num);
  for (size_t i = 0; i < num; i++) {
    // We don't need to compare if it doesn't need to output data.
    if (req[i] == kNullOp)
      continue;
    MSHADOW_TYPE_SWITCH(outputs[i].dtype(), DType, {
      bool similar = SimilarArray<DType>(outputs[i], outputs_[i], 1e-2, 1e-2);
      if (!similar) {
        LOG(ERROR) << attrs.op->name << " fails";
      }
      CHECK(similar);
    });
  }
}

void OpCheck::CopyResult(const std::vector<mxnet::NDArray> &outputs_,
                         const std::vector<size_t> &indice) {
  CHECK(!MKLDNNStream::Get()->HasOps());
  auto non_const_outputs_ = const_cast<std::vector<mxnet::NDArray> &>(outputs_);
  for (auto i = indice.begin(); i != indice.end(); ++i) {
    auto mem = outputs[*i].GetMKLDNNData();
    non_const_outputs_[*i].CopyFrom(*mem);
  }
  MKLDNNStream::Get()->Submit();
}

bool MKLDNNStorageType(const nnvm::NodeAttrs &attrs,
                       const int dev_mask,
                       bool support_mkldnn,
                       DispatchMode *dispatch_mode,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  for (int& v : *in_attrs)
    if (v == - 1) v = kDefaultStorage;

  DispatchMode wanted_mode;
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !MKLDNNEnvSet())
    wanted_mode = DispatchMode::kFComputeFallback;
  else if (dev_mask == mshadow::cpu::kDevMask && support_mkldnn)
    wanted_mode = DispatchMode::kFComputeEx;
  else
#endif
    wanted_mode = DispatchMode::kFCompute;

  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                         dispatch_mode, wanted_mode);
  }
  if (!dispatched) {
    dispatched = op::dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

inline static const std::vector<NDArray> GetMKLDNNInputArray(const std::vector<NDArray> &inputs) {
  std::vector<NDArray> ret;
  ret.reserve(inputs.size());
  for (const auto &in : inputs) {
    if (in.IsView() && in.IsMKLDNNData()) {
      ret.push_back(in.Reorder2Default());
    } else {
      ret.push_back(in);
    }
  }
  return ret;
}

void MKLDNNRun(mxnet::FComputeEx fn,
               const nnvm::NodeAttrs &attrs,
               const mxnet::OpContext &ctx,
               const std::vector<mxnet::NDArray> &inputs,
               const std::vector<mxnet::OpReqType> &req,
               const std::vector<mxnet::NDArray> &outputs) {
  if (CheckMKLDNNInputArrayIsView(inputs)) {
    const auto mkldnn_inputs = GetMKLDNNInputArray(inputs);
    fn(attrs, ctx, mkldnn_inputs, req, outputs);
  } else {
    fn(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace mxnet

#endif

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

mkldnn::memory *TmpMemMgr::Alloc(const mkldnn::memory::primitive_desc &pd) {
  // We need to include the size of the memory used for alignment.
  this->est_size += pd.get_size() + alignment;
  void *mem = AlignMem(this->curr_mem, pd.get_size(), alignment, &this->curr_size);
  if (mem) {
    // The memory is allocated from the temporary memory space in the
    // operator. It'll only become invalid after we exit from the operator.
    mkldnn_mem_ptr ret(new mkldnn::memory(pd, mem));
    MKLDNNStream::Get()->RegisterMem(ret);
    CHECK_EQ(mem, mem);
    this->curr_size -= pd.get_size();
    this->curr_mem = static_cast<char *>(mem) + pd.get_size();
    return ret.get();
  } else {
    // If curr_mem has been initialized and we still reach here. It means
    // the current allocated memory isn't enough.
    if (this->curr_mem)
      LOG(WARNING) << "Allocate " << pd.get_size()
          << " bytes with malloc directly";
    mkldnn_mem_ptr ret(new mkldnn::memory(pd));
    MKLDNNStream::Get()->RegisterMem(ret);
    return ret.get();
  }
}

mkldnn_output_t CreateMKLDNNMem(const NDArray &arr,
                                const mkldnn::memory::primitive_desc &desc,
                                OpReqType req) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req) {
    // MKLDNN ops may not support the case that the input and the output uses
    // the same memory. Let's use an extra copy to make sure it always works.
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::CopyBack, tmp);
  } else {
    mkldnn::memory *mem = const_cast<NDArray &>(arr).CreateMKLDNNData(desc);
    if (mem == nullptr) {
      auto tmp = TmpMemMgr::Get()->Alloc(desc);
      return mkldnn_output_t(OutDataOp::CopyBack, tmp);
    } else {
      return mkldnn_output_t(OutDataOp::Noop, mem);
    }
  }
}

mkldnn_output_t CreateMKLDNNWeightGrad(const NDArray &arr,
                                       const mkldnn::memory::primitive_desc &desc,
                                       OpReqType req) {
  if (kAddTo == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::AddBack, tmp);
  } else if (kWriteInplace == req) {
    auto tmp = TmpMemMgr::Get()->Alloc(desc);
    return mkldnn_output_t(OutDataOp::CopyBack, tmp);
  } else {
    auto _desc = desc;
    auto def_format = GetDefaultFormat(_desc.desc());
    mkldnn::memory *mem = nullptr;
    if (def_format == _desc.desc().data.format) {
      mem = const_cast<NDArray &>(arr).CreateMKLDNNData(desc);
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
    auto mem = arr.GetMKLDNNData(res.second->get_primitive_desc());
    CHECK(mem != nullptr);
    // We have to allocate new memory for the sum result.
    auto sum_res = TmpMemMgr::Get()->Alloc(
        res.second->get_primitive_desc());
    op::MKLDNNSum(*res.second, *mem, *sum_res);
    const_cast<NDArray &>(arr).CopyFrom(*sum_res);
  }
}

const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::memory::primitive_desc &target_pd,
                                 int num_groups) {
  const mkldnn::memory *mem = arr.GetMKLDNNData(target_pd);
  // If the weight array already uses the target layout, simply return it
  // directly.
  if (mem)
    return mem;

  mkldnn::memory::data_type type = get_mkldnn_type(arr.dtype());
  auto engine = CpuEngine::Get()->get_engine();
  if (arr.shape().ndim() == 2) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oi};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4 && num_groups == 1) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1]),
          static_cast<int>(arr.shape()[2]), static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{ num_groups,
      static_cast<int>(arr.shape()[0] / num_groups),
      static_cast<int>(arr.shape()[1]),
      static_cast<int>(arr.shape()[2]),
      static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::goihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    mem = arr.GetMKLDNNData(pd);
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
    return nullptr;
  }
  if (mem == nullptr)
    mem = arr.GetMKLDNNDataReorder(target_pd);
  if (mem->get_primitive_desc() == target_pd) return mem;

  auto ret = TmpMemMgr::Get()->Alloc(target_pd);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*mem, *ret));
  return ret;
}

mkldnn_memory_format_t GetDefaultFormat(int num_dims) {
  switch (num_dims) {
    case 1: return mkldnn_x;
    case 2: return mkldnn_nc;
    case 4: return mkldnn_nchw;
    case 5: return mkldnn_goihw;
    default:
      LOG(FATAL) << "Unsupported MKLDNN dimensions: " << num_dims;
      return mkldnn_format_undef;
  }
}

mkldnn_memory_format_t GetDefaultFormat(const mkldnn::memory::desc &desc) {
  if (desc.data.ndims == 1) {
    return desc.data.format;
  } else if (desc.data.ndims == 2) {
    if (desc.data.format == mkldnn_io)
      return mkldnn_oi;
    else
      return desc.data.format;
  } else if (desc.data.ndims == 4) {
    switch (desc.data.format) {
      case mkldnn_nchw:
      case mkldnn_nhwc:
      case mkldnn_chwn:
      case mkldnn_nChw8c:
      case mkldnn_nChw16c:
        return mkldnn_nchw;
      case mkldnn_oihw:
      case mkldnn_ihwo:
      case mkldnn_hwio:
      case mkldnn_OIhw8i8o:
      case mkldnn_OIhw16i16o:
      case mkldnn_OIhw4i16o4i:
      case mkldnn_OIhw8i16o2i:
      case mkldnn_OIhw8o16i2o:
      case mkldnn_OIhw8o8i:
      case mkldnn_OIhw16o16i:
      case mkldnn_IOhw16o16i:
      case mkldnn_Oihw8o:
      case mkldnn_Oihw16o:
      case mkldnn_Ohwi8o:
      case mkldnn_Ohwi16o:
      case mkldnn_OhIw16o4i:
        return mkldnn_oihw;
      default:
        LOG(FATAL) << "Unknown MKLDNN format for 4 dimensions: " << desc.data.format;
        return mkldnn_format_undef;
    }
  } else if (desc.data.ndims == 5) {
    switch (desc.data.format) {
      case mkldnn_goihw:
      case mkldnn_gOIhw8i8o:
      case mkldnn_gOIhw16i16o:
      case mkldnn_gOIhw4i16o4i:
      case mkldnn_gOIhw8i16o2i:
      case mkldnn_gOIhw8o16i2o:
      case mkldnn_gOIhw8o8i:
      case mkldnn_gOIhw16o16i:
      case mkldnn_gIOhw16o16i:
      case mkldnn_gOihw8o:
      case mkldnn_Goihw8g:
      case mkldnn_gOihw16o:
      case mkldnn_Goihw16g:
      case mkldnn_gOhwi8o:
      case mkldnn_gOhwi16o:
      case mkldnn_gOhIw16o4i:
        return mkldnn_goihw;
      default:
        LOG(FATAL) << "Unknown MKLDNN format for 5 dimensions: " << desc.data.format;
        return mkldnn_format_undef;
    }
  } else {
    LOG(FATAL) << "Unsupported dimensions: " << desc.data.ndims;
    return mkldnn_format_undef;
  }
}

mkldnn::memory::primitive_desc GetPrimitiveDesc(mkldnn::memory::primitive_desc pd,
                                                mkldnn_memory_format_t format) {
  mkldnn::memory::dims dims(pd.desc().data.ndims);
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = pd.desc().data.dims[i];
  mkldnn::memory::format cpp_format = static_cast<mkldnn::memory::format>(format);
  mkldnn::memory::data_type cpp_type = static_cast<mkldnn::memory::data_type>(
      pd.desc().data.data_type);
  mkldnn::memory::desc data_md(dims, cpp_type, cpp_format);
  return mkldnn::memory::primitive_desc(data_md, pd.get_engine());
}

void FallBackCompute(FCompute fn, const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs) {
  std::vector<TBlob> in_blobs(inputs.size());
  std::vector<NDArray> in_bufs;
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
      in_bufs.emplace_back(inputs[i].shape(), inputs[i].ctx(),
                           false, inputs[i].dtype());
      const mkldnn::memory *mem = inputs[i].GetMKLDNNData();
      in_bufs.back().CopyFrom(*mem);
      in_blobs[i] = in_bufs.back().data();
    }
  }
  MKLDNNStream::Get()->Submit();

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++) {
    NDArray output = outputs[i];
    // ensure output does not use mkldnn mem.
    // for inplace, we already converted & copied input above.
    if ((req[i] == kWriteTo) || (req[i] == kWriteInplace))
      const_cast<NDArray &>(output).InvalidateMKLDNNData();
    else if (req[i] == kAddTo)
      output = outputs[i].Reorder2Default();
    CHECK(output.IsDefaultData());
    out_blobs[i] = output.data();
  }
  fn(attrs, ctx, in_blobs, req, out_blobs);
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
  for (int64_t i = 0; i < arr1.shape().Size(); i++) {
#else
  for (size_t i = 0; i < arr1.shape().Size(); i++) {
#endif
    if (std::abs(data1[i] - data2[i]) > atol + rtol * std::abs(data2[i]))
      success.store(false);
  }
  return success.load();
}

void OpCheck::Init(const std::vector<mxnet::NDArray> &inputs_,
                   const std::vector<mxnet::NDArray> &outputs_) {
  auto ctx = inputs_[0].ctx();
  CHECK(!MKLDNNStream::Get()->HasOps());
  for (size_t i = 0; i < inputs_.size(); i++) {
    inputs.emplace_back(inputs_[i].shape(), ctx,
                        false, inputs_[i].dtype());
    auto mem = inputs_[i].GetMKLDNNData();
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
  std::vector<mxnet::TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) in_blobs[i] = inputs[i].data();
  std::vector<mxnet::TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  fn(attrs, ctx, in_blobs, req, out_blobs);

  LOG(INFO) << "test " << attrs.op->name;
  size_t num = std::min(outputs.size(), outputs_.size());
  num = std::min(num_checks, num);
  for (size_t i = 0; i < num; i++) {
    // We don't need to compare if it doesn't need to output data.
    if (req[i] == kNullOp)
      continue;
    MSHADOW_TYPE_SWITCH(outputs[i].dtype(), DType, {
      bool similar = SimilarArray<DType>(outputs[i], outputs_[i], 1e-3, 1e-3);
      if (!similar) {
        LOG(ERROR) << attrs.op->name << " fails";
      }
      CHECK(similar);
    });
  }
}

}  // namespace mxnet

#endif

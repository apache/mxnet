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

#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

namespace mxnet {

mkldnn::memory *TmpMemMgr::Alloc(const mkldnn::memory::primitive_desc &pd) {
  // We need to include the size of the memory used for alignment.
  this->est_size += pd.get_size() + alignment;
  void *this_mem = this->curr_mem;
  void *mem = std::align(alignment, pd.get_size(), this_mem, this->curr_size);
  if (mem) {
    // The memory is allocated from the temporary memory space in the
    // operator. It'll only become invalid after we exit from the operator.
    mkldnn_mem_ptr ret(new mkldnn::memory(pd, this_mem));
    MKLDNNStream::Get()->RegisterMem(ret);
    CHECK_EQ(this_mem, mem);
    this->curr_size -= pd.get_size();
    this->curr_mem = static_cast<char *>(this_mem) + pd.get_size();
    return ret.get();
  } else {
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
    op::Sum(*res.second, *mem, *sum_res);
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

mkldnn_memory_format_t GetDefaultFormat(mkldnn::memory::desc desc) {
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
      case mkldnn_gOIhw8i16o2i:
      case mkldnn_gOIhw8o16i2o:
      case mkldnn_gOIhw8o8i:
      case mkldnn_gOIhw16o16i:
      case mkldnn_gIOhw16o16i:
      case mkldnn_gOihw8o:
      case mkldnn_gOihw16o:
      case mkldnn_gOhwi8o:
      case mkldnn_gOhwi16o:
      case mkldnn_gOhIw16o4i:
        return mkldnn_goihw;
      default:
        LOG(FATAL) << "Unknown MKLDNN format for 4 dimensions: " << desc.data.format;
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
  // TODO(zhengda) We should buffer the NDArrays.
  std::vector<NDArray> in_bufs;
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) {
      in_blobs[i] = inputs[i].data();
  }
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++) {
    if (req[i] == kWriteTo)
      const_cast<NDArray &>(outputs[i]).InvalidateData();
    CHECK(outputs[i].IsDefault());
    out_blobs[i] = outputs[i].data();
  }
  fn(attrs, ctx, in_blobs, req, out_blobs);
}

}  // namespace mxnet

#endif

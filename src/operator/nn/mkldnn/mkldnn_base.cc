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
  const mkldnn::memory *mem;
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
  if (mem->get_primitive_desc() == target_pd) return mem;

  auto ret = TmpMemMgr::Get()->Alloc(target_pd);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(*mem, *ret));
  return ret;
}

const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::engine &engine,
                                 int num_groups) {
  mkldnn::memory::data_type type = get_mkldnn_type(arr.dtype());
  if (arr.shape().ndim() == 2) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oi};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    return arr.GetMKLDNNData(pd);
  } else if (arr.shape().ndim() == 4 && num_groups == 1) {
    mkldnn::memory::dims tz = mkldnn::memory::dims{
      static_cast<int>(arr.shape()[0]), static_cast<int>(arr.shape()[1]),
          static_cast<int>(arr.shape()[2]), static_cast<int>(arr.shape()[3])};
    mkldnn::memory::desc md =
        mkldnn::memory::desc{tz, type, mkldnn::memory::format::oihw};
    mkldnn::memory::primitive_desc pd =
        mkldnn::memory::primitive_desc{md, engine};
    return arr.GetMKLDNNData(pd);
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
    return arr.GetMKLDNNData(pd);
  } else {
    LOG(FATAL) << "The weight array has an unsupported number of dimensions";
    return nullptr;
  }
}

}  // namespace mxnet

#endif

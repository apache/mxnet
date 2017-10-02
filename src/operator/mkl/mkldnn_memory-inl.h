/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_memory-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_MKL_MKLDNN_MEMORY_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_MEMORY_INL_H_

#include <string>
#include <vector>
#include <iterator>
#if MXNET_USE_MKLDNN == 1
#include "mkldnn.hpp"
#include "mkldnn_base-inl.h"
#define CHECK_MKL_BUFFER 0
#if CHECK_MKL_BUFFER == 1
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#endif
using namespace mkldnn;

namespace mxnet {

template <typename Dtype>
struct MKLDNNMemoryDescriptorBase : public PrvMemDescr,
 public std::enable_shared_from_this<MKLDNNMemoryDescriptorBase<Dtype> > {
    MKLDNNMemoryDescriptorBase(std::shared_ptr<memory::primitive_desc> usr_memory_pd
        , std::shared_ptr<memory::primitive_desc> prv_memory_pd);

    ~MKLDNNMemoryDescriptorBase() {
    }
    std::shared_ptr<MKLDNNMemoryDescriptorBase<Dtype> > get_shared_ptr() {
      return this->shared_from_this();
    }
    // ---- PrvMemDescr virtual functions -----
    void allocate() {
      if (_prv_memory == nullptr) {
        _prv_memory = std::shared_ptr<memory>(new memory(*_prv_memory_pd));
        _internal_ptr = reinterpret_cast<Dtype *>(_prv_memory->get_data_handle());
        _internal_size = prv_size();
      }
    }
    std::shared_ptr<memory>  get_prv_memory(bool usage_check = false) {
      if (_prv_memory == nullptr) {
        if (usage_check)
          CHECK_EQ(usage_check, true) << "get null prv memory from";
        allocate();
      }
      return _prv_memory;
    }
    inline bool conversion_needed() const {
      if (!_prv_memory_pd_not_null)
        return false;
      if (!_usr_memory_pd_not_null)
        return false;
      if (*_usr_memory_pd != *_prv_memory_pd)
        return true;
      else
        return false;
    }

    void set_prv_memory_pd(std::shared_ptr<memory::primitive_desc> memory_pd) {
      _prv_memory_pd = memory_pd;
      if (_prv_memory_pd)
        _prv_memory_pd_not_null = true;
    }

    void set_usr_memory_pd(std::shared_ptr<memory::primitive_desc> memory_pd) {
      _usr_memory_pd = memory_pd;
      if (_usr_memory_pd)
        _usr_memory_pd_not_null = true;
    }

    virtual void* prv_ptr(bool allocate_when_uninit = true) {
      return _internal_ptr;
    }
    virtual size_t prv_size() { return _prv_memory_pd->get_size(); }
    virtual size_t prv_count() { return prv_size() / sizeof(Dtype); }

    virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other);
    virtual PrvDescrType get_descr_type() { return PRV_DESCR_MKLDNN; }

    std::shared_ptr<memory::primitive_desc>  prv_memory_pd() const {
        return _prv_memory_pd;
    }
    std::shared_ptr<memory::primitive_desc>  usr_memory_pd() const {
        return _usr_memory_pd;
    }

    std::string name;  // for debugging purposes

    void check_usr_with_prv_descriptors();
    void set_prv_memory(std::shared_ptr<memory> memory) {
        _prv_memory = memory;
        if (_prv_memory == nullptr) {
          _internal_ptr = reinterpret_cast<Dtype *>(_prv_memory->get_data_handle());
          _internal_size = prv_size();
        } else {
          VLOG(1) << "Set NULL Prv Memory";
        }
    }

 protected:
    std::shared_ptr<memory::primitive_desc> _usr_memory_pd;
    std::shared_ptr<memory::primitive_desc> _prv_memory_pd;
    bool _usr_memory_pd_not_null;
    bool _prv_memory_pd_not_null;
    std::shared_ptr<memory> _prv_memory;
    Dtype* _internal_ptr;
    int  _internal_size;
    std::shared_ptr<memory> _usr_memory;
    void* _dbg_cpu_ptr;
};

template <typename Dtype>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype> {
 public:
    MKLDNNMemoryDescriptor(std::shared_ptr<memory::primitive_desc> usr_memory_pd
        , std::shared_ptr<memory::primitive_desc> prv_memory_pd);

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_extprv(std::shared_ptr<memory> extprv_memory);
    virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other);
    virtual bool on_to_cpu();

    virtual void create_reorder_from_prv(void* cpu_ptr);
    virtual void create_reorder_to_prv(void* cpu_ptr);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    std::shared_ptr<memory> get_converted_prv(Dtype* cpu_data,
      bool set_prv_ptr, const TBlob &blob);
    void sync_converted_prv(Dtype* cpu_data, bool set_prv_ptr, const TBlob &tblob);
    std::shared_ptr<memory> create_output_memory(Dtype* cpu_data, const TBlob &blob,
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > thisData = nullptr, bool in_place = false);
    void sync_output_memory(const TBlob &blob,
        std::shared_ptr<MKLDNNMemoryDescriptor<Dtype> > thisData = nullptr, bool in_place = false);

    std::shared_ptr<primitive>  reorder_usr2prv() { return _reorder_usr2prv.aprimitive; }
    std::shared_ptr<primitive>  reorder_prv2usr() { return _reorder_prv2usr.aprimitive; }

 private:
    MKLDNNPrimitive<Dtype> _reorder_usr2prv;
    MKLDNNPrimitive<Dtype> _reorder_prv2usr;
};

template <typename Dtype>
class MKLDNNData : public MKLDNNMemoryDescriptor<Dtype> {
 public:
    MKLDNNData(std::shared_ptr<memory::primitive_desc> usr_memory_pd
        , std::shared_ptr<memory::primitive_desc> prv_memory_pd)
        : MKLDNNMemoryDescriptor<Dtype>(usr_memory_pd, prv_memory_pd) {}
};

template <typename Dtype>
std::shared_ptr<MKLDNNData<Dtype> >
get_mkldnn_prv_descriptor(std::shared_ptr<MKLMemHolder> blob);

template <typename Dtype>
inline std::shared_ptr<MKLDNNData<Dtype> > get_mkldnn_prv_descriptor(const TBlob &b) {
  return get_mkldnn_prv_descriptor<Dtype>(b.Mkl_mem_);
}

template<typename DType>
inline std::shared_ptr<memory> mkldnn_prv_memory(const TBlob &b) {
  std::shared_ptr<MKLMemHolder> mkl_mem = b.Mkl_mem_;
  bool mem_valid = (mkl_mem != nullptr) && mkl_mem->head_at_prv();
  if (mem_valid) {
    std::shared_ptr<MKLDNNMemoryDescriptor<DType> > mem_desc
      = get_mkldnn_prv_descriptor<DType>(mkl_mem);
    if (mem_desc != nullptr)
      return mem_desc->get_prv_memory(true);
  }
  return nullptr;
}
template struct MKLDNNData<float>;
template struct MKLDNNData<double>;
template struct MKLDNNData<uint8_t>;
template struct MKLDNNData<int8_t>;
template struct MKLDNNData<int32_t>;

}  // namespace mxnet

#endif
#endif  // MXNET_OPERATOR_MKL_MKLDNN_MEMORY_INL_H_

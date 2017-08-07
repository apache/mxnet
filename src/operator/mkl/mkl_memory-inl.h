/*******************************************************************************
* Copyright 2016 Intel Corporation
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
* \file mkl_memory-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_MEMORY_INL_H_
#define MXNET_OPERATOR_MKL_MKL_MEMORY_INL_H_


#include <string>
#include <vector>
#include <memory>
#include "mkl_cppwrapper.h"

namespace mxnet {

template <typename DType>
struct MKLMemoryDescriptorBase : public PrvMemDescr,
 public std::enable_shared_from_this<MKLMemoryDescriptorBase<DType> > {
    MKLMemoryDescriptorBase() : layout_usr(NULL), layout_int(NULL),
    convert_to_int(NULL), convert_from_int(NULL), convert_prv2prv(NULL),
    name("UNKNOWN"), internal_ptr(NULL) {}
  virtual ~MKLMemoryDescriptorBase() {
    dnnLayoutDelete<DType>(layout_usr);
    dnnLayoutDelete<DType>(layout_int);
    if (internal_ptr != NULL) {
      dnnReleaseBuffer<DType>(internal_ptr);
      internal_ptr = NULL;
    }
    if (convert_to_int != NULL) {
      dnnDelete<DType>(convert_to_int);
      convert_to_int = NULL;
    }
    if (convert_from_int != NULL) {
      dnnDelete<DType>(convert_from_int);
      convert_from_int = NULL;
    }
    if (convert_prv2prv != NULL) {
      dnnDelete<DType>(convert_prv2prv);
      convert_prv2prv = NULL;
    }
  }
  std::shared_ptr<MKLMemoryDescriptorBase<DType> > get_shared_ptr() {
    return this->shared_from_this();
  }

  dnnLayout_t layout_usr;
  dnnLayout_t layout_int;
  dnnPrimitive_t convert_to_int;
  dnnPrimitive_t convert_from_int;
  dnnPrimitive_t convert_prv2prv;
  std::shared_ptr<MKLMemoryDescriptorBase<DType> > descr_prv2prv_conversion;


  std::string name;  // for debugging purposes
  void allocate() {
    if (internal_ptr == NULL) {
      int status = dnnAllocateBuffer<DType>(
              reinterpret_cast<void **>(&internal_ptr), layout_int);
      CHECK_EQ(status, E_SUCCESS)
          << "Failed internal_ptr memory allocation with status "
          << status << "\n";
    }
  }
  virtual void* prv_ptr(bool allocate_when_uninit = true) {
    if (internal_ptr == NULL && allocate_when_uninit)
      allocate();
    return internal_ptr;
  }
  inline bool conversion_needed() {
    return (convert_to_int != NULL);
  }
  void create_conversions();
  void create_internal_layout(const dnnPrimitive_t primitive,
                dnnResourceType_t type);
  void create_user_layout(size_t dimension, const size_t size[],
              const size_t strides[]);
  void create_layouts(
    const dnnPrimitive_t primitive, dnnResourceType_t type,
    size_t dimension, const size_t size[], const size_t strides[]);

  virtual PrvDescrType get_descr_type() {
    return PRV_DESCR_MKL2017;
  }
  virtual size_t prv_size() {
    return dnnLayoutGetMemorySize<DType>(layout_int);
  }
  virtual size_t prv_count() {
    return dnnLayoutGetMemorySize<DType>(layout_int) / sizeof(DType);
  }
  virtual void convert_from_prv(void* cpu_ptr);
  virtual void convert_to_prv(void* cpu_ptr);
  virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other);
  virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other);
 protected:
  DType* internal_ptr;
};

template <typename DType>
struct MKLMemoryDescriptor : MKLMemoryDescriptorBase<DType> {
  // The last get_converted_prv() argument is a hack for reusing
  // in backward a conversion done already in the forward direction.
  DType* get_converted_prv(DType *data_ptr, bool set_prv_ptr,
      const TBlob &blob);
  void* get_output_ptr(DType *data_ptr, std::shared_ptr<MKLMemoryDescriptor<DType> > self_ptr,
    const TBlob &blob, bool in_place = false);
  bool copy_from(std::shared_ptr<MKLMemHolder> dnn_chunk);
  MKLMemoryDescriptor() {}
};

template <typename DType> struct MKLData : MKLMemoryDescriptor<DType> {
  static std::shared_ptr<MKLData<DType> > create() {
    return std::make_shared<MKLData<DType> >();
  }
};

template struct MKLData<float>;
template struct MKLData<double>;

}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_MEMORY_INL_H_

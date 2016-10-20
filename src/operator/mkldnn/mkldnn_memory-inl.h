/*!
 * Copyright (c) 2016 by Contributors
 * \file mkldnn_memory-inl.h
 * \brief
 * \author Chen, Xiaoming
*/

#ifndef MXNET_OPERATOR_MKLDNN_MKLDNN_MEMORY_INL_H_
#define MXNET_OPERATOR_MKLDNN_MKLDNN_MEMORY_INL_H_

#include <dmlc/logging.h>
#include "./mkldnn_cppwrapper.h"
namespace mxnet {
#if MXNET_USE_MKLDNN == 1
template <typename DType>
struct MKLMemoryDescriptorBase {
  MKLMemoryDescriptorBase()
      : layout_usr(NULL),
        layout_int(NULL),
        convert_to_int(NULL),
        convert_from_int(NULL),
        internal_ptr(NULL) {}
  ~MKLMemoryDescriptorBase() {
    dnnLayoutDelete<DType>(layout_usr);
    dnnLayoutDelete<DType>(layout_int);
    dnnReleaseBuffer<DType>(internal_ptr);
    dnnDelete<DType>(convert_to_int);
    dnnDelete<DType>(convert_from_int);
  }

  void allocate() {
    if (internal_ptr == NULL) {
      CHECK_EQ(dnnAllocateBuffer<DType>(reinterpret_cast<void**>(&internal_ptr),
                                        layout_int),
               E_SUCCESS);
    }
  }

  void create_conversions() {
    dnnError_t status;
    if (this->convert_from_int) {
      status = dnnDelete<DType>(this->convert_from_int);
      CHECK_EQ(status, E_SUCCESS);
    }
    if (this->convert_to_int) {
      status = dnnDelete<DType>(this->convert_to_int);
      CHECK_EQ(status, E_SUCCESS);
    }
    if (this->layout_int && !dnnLayoutCompare<DType>(layout_usr, layout_int)) {
      CHECK(layout_usr);
      status =
          dnnConversionCreate<DType>(&convert_to_int, layout_usr, layout_int);
      CHECK_EQ(status, E_SUCCESS);
      status =
          dnnConversionCreate<DType>(&convert_from_int, layout_int, layout_usr);
      CHECK_EQ(status, E_SUCCESS);
    }
  }

  void create_internal_layout(const dnnPrimitive_t primitive,
                              dnnResourceType_t type) {
    dnnError_t status;
    if (this->layout_int) {
      status = dnnLayoutDelete<DType>(this->layout_int);
      CHECK_EQ(status, E_SUCCESS);
    }
    status =
        dnnLayoutCreateFromPrimitive<DType>(&this->layout_int, primitive, type);
    CHECK_EQ(status, E_SUCCESS) << "internal layout create fail with status "
                                << status;

    if (this->layout_usr) this->create_conversions();
  }

  void create_user_layout(size_t dimension, const size_t size[],
                          const size_t strides[]) {
    dnnError_t status;
    if (this->layout_usr) {
      status = dnnLayoutDelete<DType>(this->layout_usr);
      CHECK_EQ(status, E_SUCCESS);
    }

    status =
        dnnLayoutCreate<DType>(&this->layout_usr, dimension, size, strides);
    CHECK_EQ(status, E_SUCCESS) << "user layout create fail";

    if (this->layout_int) this->create_conversions();
  }

  void create_layouts(const dnnPrimitive_t primitive, dnnResourceType_t type,
                      size_t dimension, const size_t size[],
                      const size_t strides[]) {
    this->create_internal_layout(primitive, type);
    this->create_user_layout(dimension, size, strides);
  }

  dnnLayout_t layout_usr;
  dnnLayout_t layout_int;
  dnnPrimitive_t convert_to_int;
  dnnPrimitive_t convert_from_int;
  DType* internal_ptr;
};

template <typename DType>
struct MKLMemoryDescriptor : MKLMemoryDescriptorBase<DType> {
  DType* get_converted_prv(DType* data, bool second_use) {
    if (second_use) {
      if (this->convert_to_int) {
        return this->internal_ptr;
      }
      return data;
    }

    if (this->convert_to_int) {
      dnnError_t status;
      void* convert_resources[dnnResourceNumber];

      this->allocate();
      convert_resources[dnnResourceFrom] = reinterpret_cast<void*>(data);
      convert_resources[dnnResourceTo] =
          reinterpret_cast<void*>(this->internal_ptr);

      status = dnnExecute<DType>(this->convert_to_int, convert_resources);
      CHECK_EQ(status, E_SUCCESS) << "status " << status;

      return this->internal_ptr;
    }

    return data;
  }

  DType* set_output_ptr(DType* data) {
    if (this->convert_to_int) {
      this->allocate();
      return this->internal_ptr;
    }
    return data;
  }

  DType* get_output_ptr(DType* data) {
    if (this->convert_from_int) {
      dnnError_t status;
      void* convert_resources[dnnResourceNumber];

      convert_resources[dnnResourceFrom] =
          reinterpret_cast<void*>(this->internal_ptr);
      convert_resources[dnnResourceTo] = reinterpret_cast<void*>(data);

      status = dnnExecute<DType>(this->convert_from_int, convert_resources);
      CHECK_EQ(status, E_SUCCESS) << "status " << status;
    }

    return data;
  }
};

template <typename DType> struct MKLData : MKLMemoryDescriptor<DType> {
};

template struct MKLData<float>;
template struct MKLData<double>;
#endif
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKLDNN_MKLDNN_MEMORY_INL_H_

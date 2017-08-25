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
* \file mkl_memory.cc
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#include "../operator_common.h"

#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "mkl_memory-inl.h"
#include "mkl_util-inl.h"

namespace mxnet {

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_conversions() {
  int status;
  if (this->convert_from_int) {
    status = dnnDelete<Dtype>(this->convert_from_int);
    CHECK_EQ(status, E_SUCCESS);
    this->convert_from_int = NULL;
  }
  if (this->convert_to_int) {
    status = dnnDelete<Dtype>(this->convert_to_int);
    CHECK_EQ(status, E_SUCCESS);
    this->convert_to_int = NULL;
  }
  if (layout_int
      && !dnnLayoutCompare<Dtype>(layout_usr, layout_int)) {
    CHECK(layout_usr);
    status = dnnConversionCreate<Dtype>(&convert_to_int, layout_usr,
            layout_int);
    CHECK_EQ(status, E_SUCCESS)
            << "Failed creation convert_to_int with status "
            << status << " for buffer: " << this->name << "\n";
    status = dnnConversionCreate<Dtype>(&convert_from_int, layout_int,
            layout_usr);
    CHECK_EQ(status, E_SUCCESS)
            << "Failed creation convert_from_int with status "
            << status << " for buffer: " << this->name << "\n";
  }
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_internal_layout(
    const dnnPrimitive_t primitive, dnnResourceType_t type) {
  int status;
  if (this->layout_int) {
    status = dnnLayoutDelete<Dtype>(this->layout_int);
    CHECK_EQ(status, E_SUCCESS);
  }
  status = dnnLayoutCreateFromPrimitive<Dtype>(
      &this->layout_int, primitive, type);
  CHECK_EQ(status, E_SUCCESS)
      << "Failed dnnLayoutCreateFromPrimitive with status "
      << status << " for buffer: " << this->name << "\n";

  if (this->layout_usr)
    this->create_conversions();
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_user_layout(
    size_t dimension, const size_t size[], const size_t strides[]) {
  int status;
  if (this->layout_usr) {
    status = dnnLayoutDelete<Dtype>(this->layout_usr);
    CHECK_EQ(status, E_SUCCESS);
  }

  status = dnnLayoutCreate<Dtype>(
      &this->layout_usr, dimension, size, strides);
  CHECK_EQ(status, E_SUCCESS) << "Failed dnnLayoutCreate with status "
      << status << " for buffer: " << this->name << "\n";

  if (this->layout_int)
    this->create_conversions();
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_layouts(
    const dnnPrimitive_t primitive, dnnResourceType_t type,
    size_t dimension, const size_t size[], const size_t strides[]) {
  this->create_internal_layout(primitive, type);
  this->create_user_layout(dimension, size, strides);
}


template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_from_prv(void* cpu_ptr) {
  CHECK(cpu_ptr);
  CHECK(this->convert_from_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  convert_resources[dnnResourceFrom] = this->prv_ptr();
  convert_resources[dnnResourceTo]   = cpu_ptr;
  status = dnnExecute<Dtype>(this->convert_from_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_to_prv(void* cpu_ptr) {
  CHECK(cpu_ptr);
  CHECK(this->convert_to_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  convert_resources[dnnResourceFrom] = cpu_ptr;
  convert_resources[dnnResourceTo]   = this->prv_ptr();
  status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}


template <typename Dtype>
bool MKLMemoryDescriptorBase<Dtype>::layout_compare(
  std::shared_ptr<PrvMemDescr> other) {
  CHECK_EQ(other->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);
  std::shared_ptr<MKLMemoryDescriptorBase<Dtype> >other_descr =
    std::static_pointer_cast<MKLMemoryDescriptorBase<Dtype> >
    (other);

  if (dnnLayoutCompare<Dtype>(other_descr->layout_int,
      this->layout_int))
    return true;
  else
    return false;
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_from_other(
  std::shared_ptr<PrvMemDescr> other) {
    std::shared_ptr<MKLMemoryDescriptorBase<Dtype> > other_descr =
        std::static_pointer_cast<MKLMemoryDescriptorBase<Dtype> >
            (other);

  int status;
  dnnPrimitive_t convert;
  status = dnnConversionCreate<Dtype>(&convert,
    other_descr->layout_int, this->layout_int);

  void *convert_resources[dnnResourceNumber];
  convert_resources[dnnResourceFrom] = other_descr->prv_ptr();
  convert_resources[dnnResourceTo]   = this->prv_ptr();
  status = dnnExecute<Dtype>(convert, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from other failed with status "
                      << status;

  dnnDelete<Dtype>(convert);
}


template <typename Dtype>
Dtype* MKLMemoryDescriptor<Dtype>::get_converted_prv(
    Dtype *cpu_ptr, bool set_prv_ptr, const TBlob &blob) {
  Dtype* prv_ptr = NULL;
  std::shared_ptr<MKLMemHolder> dnn_chunk = NULL;
#if MKL_EXPERIMENTAL == 1
  dnn_chunk = blob.Mkl_mem_;
#endif
#if MKL_EXPERIMENTAL == 1
  if (dnn_chunk != NULL)
    prv_ptr = static_cast<Dtype*>(dnn_chunk->prv_data());
#endif

  if (this->convert_to_int != NULL) {
#if MKL_EXPERIMENTAL == 1
    int status;
    void *convert_resources[dnnResourceNumber];
#endif
    if (prv_ptr == NULL) {
      this->allocate();
      this->convert_to_prv(cpu_ptr);
#if MKL_EXPERIMENTAL == 1
      if (set_prv_ptr) {
        dnn_chunk->set_prv_descriptor(this->get_shared_ptr(), true);
      }
#endif
      return this->internal_ptr;
    }
#if MKL_EXPERIMENTAL == 1
    if (prv_ptr != NULL)  {
      std::shared_ptr<MKLData<Dtype> > current_descr =
        op::mkl_get_mem_desc<Dtype>(dnn_chunk);
      if (!dnnLayoutCompare<Dtype>(current_descr->layout_int,
        this->layout_int)) {
        if (this->convert_prv2prv) {
          CHECK_EQ(dnnLayoutCompare<Dtype>(
            this->descr_prv2prv_conversion->layout_int,
            this->layout_int), 0);
          status = 0;
        } else {
          status = dnnConversionCreate<Dtype>(&this->convert_prv2prv,
            current_descr->layout_int, this->layout_int);
          if (status == 0)
            this->descr_prv2prv_conversion = current_descr;
        }
        if (status != 0) {
          this->allocate();
          convert_resources[dnnResourceFrom] = cpu_ptr;
          convert_resources[dnnResourceTo] =
            reinterpret_cast<void*>(this->internal_ptr);
          status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;
        } else {
          this->allocate();
          convert_resources[dnnResourceFrom] = reinterpret_cast<void*>(prv_ptr);
          convert_resources[dnnResourceTo] =
            reinterpret_cast<void*>(this->internal_ptr);
          status = dnnExecute<Dtype>(this->convert_prv2prv, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;
        }
        if (set_prv_ptr) {
          dnn_chunk->set_prv_descriptor(this->get_shared_ptr(), true);
        }
        return this->internal_ptr;
      } else if (current_descr.get() != this) {
        // MKL_DLOG(INFO) << "layout OK                 "
        //  << current_descr->name << " == " << this->name;
      }
    }
#endif
    return const_cast<Dtype *>(prv_ptr);
  } else {
    if (prv_ptr != NULL) {
#if MKL_EXPERIMENTAL == 1
      std::shared_ptr<MKLMemoryDescriptorBase<float> > other_descr =
        std::static_pointer_cast<MKLMemoryDescriptorBase<float> >
        (dnn_chunk->prv_descriptor_);
      dnn_chunk->check_and_prv_to_cpu(cpu_ptr);
#endif
      // printf("get_converted_prv release %s\n", other_descr->name.c_str());
    }
  }
  return cpu_ptr;
}

template <typename Dtype>
void* MKLMemoryDescriptor<Dtype>::get_output_ptr(Dtype *data_ptr,
  std::shared_ptr<MKLMemoryDescriptor<Dtype> > self_ptr, const TBlob &blob, bool in_place) {
#if MKL_EXPERIMENTAL == 1
  std::shared_ptr<MKLMemHolder> dnn_chunk = blob.Mkl_mem_;
#endif
  if (this->conversion_needed()) {
    void * prv_ptr =  this->prv_ptr();
#if MKL_EXPERIMENTAL == 1
    if (!in_place) {
      dnn_chunk->set_prv_descriptor(self_ptr);
    } else {
      Dtype * blob_prv = op::mkl_prv_data<Dtype>(blob);
      if (blob_prv != NULL)
        return blob_prv;
    }
#endif
    return prv_ptr;
  } else {
#if MKL_EXPERIMENTAL == 1
    std::shared_ptr<MKLMemoryDescriptorBase<float> > other_descr =
      std::static_pointer_cast<MKLMemoryDescriptorBase<float> >
      (dnn_chunk->prv_descriptor_);
    dnn_chunk->check_and_prv_to_cpu(data_ptr);
#endif
    return data_ptr;
  }
}

template class MKLMemoryDescriptor<double>;
template class MKLMemoryDescriptor<float>;

template class MKLMemoryDescriptorBase<float>;
template class MKLMemoryDescriptorBase<double>;
}  // namespace mxnet
#endif

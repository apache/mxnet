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
* \file mkl_util-inl.h
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_
#define MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_
#include <vector>
#define MKLDNN_CALL(func)                                                               \
  {                                                                                     \
    dnnError_t status = (func);                                                                \
    CHECK_EQ(status, E_SUCCESS) << "MKL DNN call failed (status: " << status << ").";           \
  }


namespace mxnet {
namespace op {

#if MKL_EXPERIMENTAL == 1
  template<typename DType>
  inline DType * mkl_prv_data(const TBlob &b) {
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
      return reinterpret_cast<DType*>(bottom_data_mem->prv_data());
    }
    return NULL;
  }

  template<typename DType>
  inline int mkl_prv_count(const TBlob &b) {
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
      return bottom_data_mem->prv_count();
    }
    return 0;
  }
#endif
  inline void mkl_set_priv_flag(const TBlob &b) {
#if MKL_EXPERIMENTAL == 1
    std::shared_ptr<MKLMemHolder> bottom_data_mem = b.Mkl_mem_;
    bool mem_valid = (bottom_data_mem != nullptr) && bottom_data_mem->head_at_prv();
    if (mem_valid) {
      bottom_data_mem->disable_prv_2_cpu(true);
    }
#endif
  }
#if MKL_EXPERIMENTAL == 1
  template<typename DType>
  inline std::shared_ptr<MKLData<DType> > mkl_get_mem_desc(
    const std::shared_ptr<MKLMemHolder> data_mem) {
    std::shared_ptr<PrvMemDescr> prv_descriptor =
      data_mem->get_prv_descriptor();
    CHECK_EQ(prv_descriptor->get_descr_type(),
      PrvMemDescr::PRV_DESCR_MKL2017);
    std::shared_ptr<MKLData<DType> > mem_descr
      = std::static_pointer_cast<MKLData<DType>>
      (prv_descriptor);
    CHECK(mem_descr != NULL);
    return mem_descr;
  }
#endif
  template<typename xpu, int dim, typename DType>
  inline  mshadow::Tensor<xpu, dim, DType> mkl_experimental_direct_get(
    const TBlob &b, mshadow::Stream<xpu> *s) {
    mkl_set_priv_flag(b);
    return b.get<xpu, dim, DType>(s);
  }
  template<typename xpu, int dim, typename DType>
  inline  mshadow::Tensor<xpu, dim, DType> mkl_experimental_direct_get_with_shape(
    const TBlob &b, const mshadow::Shape<dim> &shape, mshadow::Stream<xpu> *s) {
    mkl_set_priv_flag(b);
    return b.get_with_shape<xpu, dim, DType>(shape, s);
  }
}  // namespace op
#if MKL_EXPERIMENTAL == 1
inline void mkl_tblobs_prv_to_cpu(const std::vector<TBlob> &data) {
  for (size_t i = 0; i < data.size(); i++) {
    std::shared_ptr<MKLMemHolder> mem_holder = data[i].Mkl_mem_;
    if (mem_holder != nullptr && mem_holder->b_eager_mode) {
      mem_holder->check_and_prv_to_cpu(data[i].dptr_);
    }
  }
}
inline void mkl_set_tblob_eager_mode(const TBlob &data) {
  std::shared_ptr<MKLMemHolder> mem_holder = data.Mkl_mem_;
  if (mem_holder != nullptr) {
    mem_holder->set_eager_mode(true);
  }
}
#endif
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_UTIL_INL_H_

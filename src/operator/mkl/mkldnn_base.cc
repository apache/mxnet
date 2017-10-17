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
* \file mkldnn_base.cc
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/

#include <dmlc/logging.h>
#ifdef MXNET_USE_MKLDNN
#include "mkldnn_base-inl.h"
using namespace mkldnn;
namespace mxnet {

bool EnableMkldnnWarnGenerated() {
  return false;
}
std::shared_ptr<MKLDNNStream> StreamHolder::get_stream() {
    if (this->_current_stream == NULL || !this->_current_stream->ready()) {
        _current_stream.reset(new MKLDNNStream());
    }
    return _current_stream;
}

template <typename Dtype>
std::shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::get_mkldnn_stream() {
    if (mkldnn_stream == NULL)
        mkldnn_stream = StreamHolder::Instance().get_stream();
    else
        StreamHolder::Instance().prepare_mkldnn_stream(mkldnn_stream);
    return mkldnn_stream;
}

template <typename Dtype>
std::shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::submit() {
    CHECK(this->aprimitive);
    try {
        this->get_mkldnn_stream()->submit({*(this->aprimitive)}).wait();
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return mkldnn_stream;
}

template class MKLDNNLayer<double>;
template class MKLDNNLayer<float>;
template class MKLDNNLayer<uint8_t>;
template class MKLDNNLayer<int8_t>;
template class MKLDNNLayer<int32_t>;

template class MKLDNNPrimitive<double>;
template class MKLDNNPrimitive<float>;
template class MKLDNNPrimitive<uint8_t>;
template class MKLDNNPrimitive<int8_t>;
template class MKLDNNPrimitive<int32_t>;
}  // namespace mxnet
#endif  // #ifdef MXNET_USE_MKLDNN

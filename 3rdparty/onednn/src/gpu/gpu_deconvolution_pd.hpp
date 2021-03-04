/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
*******************************************************************************/

#ifndef GPU_GPU_DECONVOLUTION_PD_HPP
#define GPU_GPU_DECONVOLUTION_PD_HPP

#include "common/deconvolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_deconvolution_fwd_pd_t : public deconvolution_fwd_pd_t {
    using deconvolution_fwd_pd_t::deconvolution_fwd_pd_t;
};

struct gpu_deconvolution_bwd_data_pd_t : public deconvolution_bwd_data_pd_t {
    using deconvolution_bwd_data_pd_t::deconvolution_bwd_data_pd_t;
};

struct gpu_deconvolution_bwd_weights_pd_t
    : public deconvolution_bwd_weights_pd_t {
    using deconvolution_bwd_weights_pd_t::deconvolution_bwd_weights_pd_t;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

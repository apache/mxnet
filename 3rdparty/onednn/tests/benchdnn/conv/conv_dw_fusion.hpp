/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CONV_DW_FUSION_HPP
#define CONV_DW_FUSION_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "conv/conv_common.hpp"

namespace conv_dw_fusion {

using desc_t = conv::desc_t;
using prb_t = conv::prb_t;
using alg_t = conv::alg_t;
using dt_conf_t = conv::dt_conf_t;

int doit(const prb_t *prb, res_t *res);

} // namespace conv_dw_fusion

#endif

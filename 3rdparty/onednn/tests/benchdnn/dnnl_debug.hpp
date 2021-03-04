/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED

// clang-format off

#ifndef DNNL_DEBUG_HPP
#define DNNL_DEBUG_HPP

#include "oneapi/dnnl/dnnl.h"

dnnl_data_type_t str2dt(const char *str);
dnnl_format_tag_t str2fmt_tag(const char *str);

/* status */
const char *status2str(dnnl_status_t status);

/* data type */
const char *dt2str(dnnl_data_type_t dt);

/* format */
const char *fmt_tag2str(dnnl_format_tag_t tag);

/* endinge kind */
const char *engine_kind2str(dnnl_engine_kind_t kind);

/* scratchpad mode */
const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode);

#endif

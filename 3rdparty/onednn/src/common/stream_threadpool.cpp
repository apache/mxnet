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

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "oneapi/dnnl/dnnl_threadpool.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "primitive_exec_types.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

dnnl_status_t dnnl_threadpool_interop_stream_create(
        stream_t **stream, engine_t *engine, void *threadpool) {
    bool args_ok = !utils::any_null(stream, engine);
    if (!args_ok) return invalid_arguments;

    auto tp = static_cast<dnnl::threadpool_interop::threadpool_iface *>(
            threadpool);

    return engine->create_stream(stream, tp);
}

dnnl_status_t dnnl_threadpool_interop_stream_get_threadpool(
        dnnl_stream_t stream, void **threadpool) {
    if (utils::any_null(stream, threadpool)) return status::invalid_arguments;
    dnnl::threadpool_interop::threadpool_iface *tp;
    auto status = stream->get_threadpool(&tp);
    if (status == status::success) *threadpool = static_cast<void *>(tp);
    return status;
}

#endif

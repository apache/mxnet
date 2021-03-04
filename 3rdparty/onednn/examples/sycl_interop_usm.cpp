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

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <CL/sycl.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

using namespace dnnl;
using namespace cl::sycl;

class kernel_tag;

void sycl_usm_tutorial(engine::kind engine_kind) {

    engine eng(engine_kind, 0);

    dnnl::stream strm(eng);

    memory::dims tz_dims = {2, 3, 4, 5};
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());
    auto usm_buffer = (float *)malloc_shared(N * sizeof(float),
            sycl_interop::get_device(eng), sycl_interop::get_context(eng));

    memory::desc mem_d(
            tz_dims, memory::data_type::f32, memory::format_tag::nchw);

    memory mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, usm_buffer);

    queue q = sycl_interop::get_queue(strm);
    auto fill_e = q.submit([&](handler &cgh) {
        cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
            int idx = (int)i[0];
            usm_buffer[idx] = (idx % 2) ? -idx : idx;
        });
    });

    auto relu_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);

    auto relu_e = sycl_interop::execute(
            relu, strm, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}}, {fill_e});
    relu_e.wait();

    for (size_t i = 0; i < N; i++) {
        float exp_value = (i % 2) ? 0.0f : i;
        if (usm_buffer[i] != (float)exp_value)
            throw std::string(
                    "Unexpected output, found a negative value after the ReLU "
                    "execution.");
    }

    free((void *)usm_buffer, sycl_interop::get_context(eng));
}

int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        sycl_usm_tutorial(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    return exit_code;
}

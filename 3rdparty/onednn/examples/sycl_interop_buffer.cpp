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

/// @example  sycl_interop_buffer.cpp
/// Annotated version: @ref sycl_interop_buffer_cpp
///
/// @page  sycl_interop_buffer_cpp Getting started on both CPU and GPU with SYCL extensions API
/// Full example text: @ref sycl_interop_buffer.cpp
///
/// This C++ API example demonstrates programming for Intel(R) Processor
/// Graphics with SYCL extensions API in oneDNN.
/// The workflow includes following steps:
///   - Create a GPU or CPU engine. It uses DPC++ as the runtime in this sample.
///   - Create a memory descriptor/object.
///   - Create a SYCL kernel for data initialization.
///   - Access a SYCL buffer via SYCL interoperability interface.
///   - Access a SYCL queue via SYCL interoperability interface.
///   - Execute a SYCL kernel with related SYCL queue and SYCL buffer
///   - Create operation descriptor/operation primitives descriptor/primitive.
///   - Execute the primitive with the initialized memory.
///   - Validate the result through a host accessor.
///
/// @page sycl_interop_buffer_cpp

/// @section sycl_interop_buffer_cpp_headers Public headers
///
/// To start using oneDNN, we must first include the @ref dnnl.hpp
/// header file in the application. We also include CL/sycl.hpp from DPC++ for
/// using SYCL APIs and @ref dnnl_debug.h, which  contains some debugging
/// facilities such as returning a string representation
/// for common oneDNN C types.
/// All C++ API types and functions reside in the `dnnl` namespace, and
/// SYCL API types and functions reside in the `cl::sycl` namespace.
/// For simplicity of the example we import both namespaces.
/// @page sycl_interop_buffer_cpp
/// @snippet  sycl_interop_buffer.cpp Prologue
// [Prologue]

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
// [Prologue]

class kernel_tag;

/// @page sycl_interop_buffer_cpp
/// @section sycl_interop_buffer_cpp_tutorial sycl_interop_buffer_tutorial() function
/// @page sycl_interop_buffer_cpp
void sycl_interop_buffer_tutorial(engine::kind engine_kind) {

    /// @page sycl_interop_buffer_cpp
    /// @subsection sycl_interop_buffer_cpp_sub1 Engine and stream
    ///
    /// All oneDNN primitives and memory objects are attached to a
    /// particular @ref dnnl::engine, which is an abstraction of a
    /// computational device (see also @ref dev_guide_basic_concepts). The
    /// primitives are created and optimized for the device to which they are
    /// attached, and the memory objects refer to memory residing on the
    /// corresponding device. In particular, that means neither memory objects
    /// nor primitives that were created for one engine can be used on
    /// another.
    ///
    /// To create engines, we must specify the @ref dnnl::engine::kind
    /// and the index of the device of the given kind. In this example we use
    /// the first available GPU or CPU engine, so the index for the engine is 0.
    /// This example assumes DPC++ being a runtime. In such case,
    /// during engine creation, an SYCL context is also created and attaches
    /// to the created engine.
    ///
    /// @snippet  sycl_interop_buffer.cpp Initialize engine
    // [Initialize engine]
    engine eng(engine_kind, 0);
    // [Initialize engine]

    /// In addition to an engine, all primitives require a @ref dnnl::stream
    /// for the execution. The stream encapsulates an execution context and is
    /// tied to a particular engine.
    ///
    /// In this example, a stream is created.
    /// This example assumes DPC++ being a runtime. During stream creation,
    /// a SYCL queue is also created and attaches to this stream.
    ///
    /// @snippet  sycl_interop_buffer.cpp Initialize stream
    // [Initialize stream]
    dnnl::stream strm(eng);
    // [Initialize stream]

    /// @subsection  sycl_interop_buffer_cpp_sub2 Wrapping data into oneDNN memory object
    ///
    /// Next, we create a memory object. We need to specify dimensions of our
    /// memory by passing a memory::dims object. Then we create a memory
    /// descriptor with these dimensions, with the dnnl::memory::data_type::f32
    /// data type, and with the dnnl::memory::format_tag::nchw memory format.
    /// Finally, we construct a memory object and pass the memory descriptor.
    /// The library allocates memory internally.
    /// @snippet  sycl_interop_buffer.cpp memory alloc
    //  [memory alloc]
    memory::dims tz_dims = {2, 3, 4, 5};
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());

    memory::desc mem_d(
            tz_dims, memory::data_type::f32, memory::format_tag::nchw);

    memory mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::buffer);
    //  [memory alloc]

    /// @subsection  sycl_interop_buffer_cpp_sub3 Initialize the data executing a custom SYCL kernel
    ///
    /// The underlying SYCL buffer can be extracted from the memory object using
    /// the interoperability interface:
    /// `dnnl::sycl_interop_buffer::get_buffer<T>(const dnnl::memory)`.
    ///
    /// @snippet  sycl_interop_buffer.cpp get sycl buf
    // [get sycl buf]
    auto sycl_buf = sycl_interop::get_buffer<float>(mem);
    // [get sycl buf]

    /// We are going to create an SYCL kernel that should initialize our data.
    /// To execute SYCL kernel we need a SYCL queue.
    /// For simplicity we can construct a stream and extract the SYCL queue from it.
    /// The kernel initializes the data by the `0, -1, 2, -3, ...` sequence: `data[i] = (-1)^i * i`.
    /// @snippet sycl_interop_buffer.cpp sycl kernel exec
    // [sycl kernel exec]
    queue q = sycl_interop::get_queue(strm);
    q.submit([&](handler &cgh) {
        auto a = sycl_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
            int idx = (int)i[0];
            a[idx] = (idx % 2) ? -idx : idx;
        });
    });
    // [sycl kernel exec]

    /// @subsection sycl_interop_buffer_cpp_sub4 Create and execute a primitive
    /// There are three steps to create an operation primitive in oneDNN:
    /// 1. Create an operation descriptor.
    /// 2. Create a primitive descriptor.
    /// 3. Create a primitive.
    ///
    /// Let's create the primitive to perform the ReLU (rectified linear unit)
    /// operation: x = max(0, x). An operation descriptor has no dependency on a
    /// specific engine - it just describes some operation. On the contrary,
    /// primitive descriptors are attached to a specific engine and represent
    /// some implementation for this engine. A primitive object is a realization
    /// of a primitive descriptor, and its construction is usually much
    /// "heavier".
    /// @snippet sycl_interop_buffer.cpp relu creation
    //  [relu creation]
    auto relu_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);
    //  [relu creation]

    /// Next, execute the primitive.
    /// @snippet sycl_interop_buffer.cpp relu exec
    // [relu exec]
    relu.execute(strm, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    strm.wait();
    // [relu exec]
    ///
    ///@note
    ///    With DPC++ runtime, both CPU and GPU have asynchronous execution; However, the user can
    ///    call dnnl::stream::wait() to synchronize the stream and ensure that all
    ///    previously submitted primitives are completed.
    ///

    /// @page sycl_interop_buffer_cpp
    /// @subsection sycl_interop_buffer_cpp_sub5 Validate the results
    ///
    /// Before running validation codes, we need to access the SYCL memory on
    /// the host.
    /// The simplest way to access the SYCL-backed memory on the host is to
    /// construct a host accessor. Then we can directly read and write this data on the host.
    /// However no any conflicting operations are allowed until the host accessor is destroyed.
    /// We can run validation codes on the host accordingly.
    /// @snippet sycl_interop_buffer.cpp Check the results
    // [Check the results]
    auto host_acc = sycl_buf.get_access<access::mode::read>();
    for (size_t i = 0; i < N; i++) {
        float exp_value = (i % 2) ? 0.0f : i;
        if (host_acc[i] != (float)exp_value)
            throw std::string(
                    "Unexpected output, find a negative value after the ReLU "
                    "execution.");
    }
    // [Check the results]
}

/// @section sycl_interop_buffer_cpp_main main() function
///
/// We now just call everything we prepared earlier.
///
/// Because we are using the oneDNN C++ API, we use exceptions to handle
/// errors (see @ref dev_guide_c_and_cpp_apis). The oneDNN C++ API throws
/// exceptions of type @ref dnnl::error, which contains the error status
/// (of type @ref dnnl_status_t) and a human-readable error message accessible
/// through the regular `what()` method.
/// @page sycl_interop_buffer_cpp
/// @snippet sycl_interop_buffer.cpp Main
// [Main]
int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        sycl_interop_buffer_tutorial(engine_kind);
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
// [Main]
/// <b></b>
///
/// Upon compiling and running the example, the output should be just:
///
/// ~~~
/// Example passed.
/// ~~~
///
/// @page sycl_interop_buffer_cpp

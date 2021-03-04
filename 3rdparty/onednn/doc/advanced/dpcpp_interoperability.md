DPC++ Interoperability {#dev_guide_dpcpp_interoperability}
==========================================================

> [API Reference](@ref dnnl_api_sycl_interop)

## Overview

oneDNN may use the DPC++ runtime for CPU and GPU engines to interact with the
hardware. Users may need to use oneDNN with other code that uses DPC++. For
that purpose, the library provides API extensions to interoperate with
underlying SYCL objects. This interoperability API is defined in the
`dnnl_sycl.hpp` header.

One of the possible scenarios is executing a SYCL kernel for a custom operation
not provided by oneDNN. In this case, the library provides all the necessary
API to "seamlessly" submit a kernel, sharing the execution context with oneDNN:
using the same device and queue.

The interoperability API is provided for two scenarios:
- Construction of oneDNN objects based on existing SYCL objects
- Accessing SYCL objects for existing oneDNN objects

The mapping between oneDNN and SYCL objects is provided in the following table:

| oneDNN object         | SYCL object(s)                             |
| :-------------------- | :----------------------------------------- |
| Engine                | `cl::sycl::device` and `cl::sycl::context` |
| Stream                | `cl::sycl::queue`                          |
| Memory (Buffer-based) | `cl::sycl::buffer<uint8_t, 1>`             |
| Memory (USM-based)    | Unified Shared Memory (USM) pointer        |

The table below summarizes how to construct oneDNN objects based on SYCL objects
and how to query underlying SYCL objects for existing oneDNN objects.

| oneDNN object         | API to construct oneDNN object                                                                      | API to access SYCL object(s)                                                                        |
| :-------------------- | :-------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------  |
| Engine                | dnnl::sycl_interop::make_engine(const cl::sycl::device &, const cl::sycl::context &)                | dnnl::sycl_interop::get_device(const engine &) <br> dnnl::sycl_interop::get_context(const engine &) |
| Stream                | dnnl::sycl_interop::make_stream(const engine &, cl::sycl::queue &)                                  | dnnl::sycl_interop::get_queue(const stream &)                                                       |
| Memory (Buffer-based) | dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, cl::sycl::buffer<T, ndims> &) | dnnl::sycl_interop::get_buffer<T, ndims>(const memory &)                                            |
| Memory (USM-based)    | dnnl::memory(const memory::desc &, const engine &, void \*)                                         | dnnl::memory::get_data_handle()                                                                     |

@note Internally, library buffer-based memory objects use 1D `uint8_t` SYCL
buffers; however, the user may initialize and access memory using SYCL buffers
of a different type. In this case, buffers will be reinterpreted to the
underlying type `cl::sycl::buffer<uint8_t, 1>`.

## SYCL Buffers and DPC++ USM Interfaces for Memory Objects

The memory model in SYCL 1.2.1 is based on SYCL buffers. DPC++ further extends
the programming model with a Unified Shared Memory (USM) alternative, which
provides the ability to allocate and use memory in a uniform way on host and
DPC++ devices.

oneDNN supports both programming models. USM is the default and can be used
with the usual oneDNN [memory API](@ref dnnl_api_memory). The buffer-based
programming model requires using the interoperability API.

To construct a oneDNN memory object, use one of the following interfaces:

- dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, sycl_interop::memory_kind kind, void \*handle)

    Constructs a USM-based or buffer-based memory object depending on memory
    allocation kind `kind`. The `handle` could be one of special values
    #DNNL_MEMORY_ALLOCATE or #DNNL_MEMORY_NONE, or it could be a user-provided
    USM pointer. The latter works only when `kind` is
    dnnl::sycl_interop::memory_kind::usm.

- dnnl::memory(const memory::desc &, const engine &, void \*)

    Constructs a USM-based memory object. The call is equivalent to calling the
    function above with with `kind` equal to
    dnnl::sycl_interop::memory_kind::usm.

- dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, cl::sycl::buffer<T, ndims> &)

    Constructs a buffer-based memory object based on a user-provided SYCL
    buffer.

To identify whether a memory object is USM-based or buffer-based,
dnnl::sycl_interop::get_memory_kind() query can be used.

## Handling Dependencies with USM

SYCL queues could be in-order or out-of-order. For out-of-order queues, the
order of execution is defined by the dependencies between SYCL tasks. The
runtime tracks dependencies based on accessors created for SYCL buffers. USM
pointers cannot be used to create accessors and users must handle dependencies
on their own using SYCL events.

oneDNN provides two mechanisms to handle dependencies when USM memory is used:

1. dnnl::sycl_interop::execute() interface

    This interface enables you to pass dependencies between primitives using
    SYCL events. In this case, the user is responsible for passing proper
    dependencies for every primitive execution.

2. In-order oneDNN stream

    oneDNN enables you to create in-order streams when submitted primitives are
    executed in the order they were submitted. Using in-order streams prevents
    possible read-before-write or concurrent read/write issues.


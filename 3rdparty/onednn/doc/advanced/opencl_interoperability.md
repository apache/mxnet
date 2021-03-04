OpenCL Interoperability {#dev_guide_opencl_interoperability}
============================================================

> [API Reference](@ref dnnl_api_ocl_interop)

## Overview

oneDNN uses the OpenCL runtime for GPU engines to interact with the GPU. Users
may need to use oneDNN with other code that uses OpenCL. For that purpose, the
library provides API extensions to interoperate with underlying OpenCL objects.
This interoperability API is defined in the `dnnl_ocl.hpp` header.

The interoperability API is provided for two scenarios:
- Construction of oneDNN objects based on existing OpenCL objects
- Accessing OpenCL objects for existing oneDNN objects

The mapping between oneDNN and OpenCL objects is provided in the following
table:

| oneDNN object        | OpenCL object(s)                |
| :------------------- | :------------------------------ |
| Engine               | `cl_device_id` and `cl_context` |
| Stream               | `cl_command_queue`              |
| Memory               | `cl_mem`                        |

The table below summarizes how to construct oneDNN objects based on OpenCL
objects and how to query underlying OpenCL objects for existing oneDNN objects.

| oneDNN object | API to construct oneDNN object                                   | API to access OpenCL object(s)                                                                    |
| :------------ | :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Engine        | dnnl::ocl_interop::make_engine(cl_device_id, cl_context)         | dnnl::ocl_interop::get_device(const engine &) <br> dnnl::ocl_interop::get_context(const engine &) |
| Stream        | dnnl::ocl_interop::make_stream(const engine &, cl_command_queue) | dnnl::ocl_interop::get_command_queue(const stream &)                                              |
| Memory        | dnnl::memory(const memory::desc &, const engine &, cl_mem)       | dnnl::ocl_interop::get_mem_object(const memory &)                                                 |

@note oneDNN follows retain/release OpenCL semantics when using OpenCL objects
during construction. An OpenCL object is retained on construction and released
on destruction. This ensures that the OpenCL object will not be destroyed while
the oneDNN object stores a reference to it.

@note The access interfaces do not retain the OpenCL object. It is the user's
responsibility to retain the returned OpenCL object if necessary.


DPC++ Backends Support: OpenCL and Level Zero {#dev_guide_dpcpp_backends}
=========================================================================

DPC++ can be used with different backends (e.g., OpenCL and Level Zero). For
more details, see: [The DPC++ Runtime Plugin
Interface](https://github.com/intel/llvm/blob/sycl/sycl/doc/PluginInterface.md).

In oneDNN, engines can be created using two different interfaces: by passing an
index, and by passing a DPC++ device and context. The following table shows what
DPC++ devices are supported (with their backends) by each of the APIs depending
on the engine kind:

| API                                                                                  | `dnnl::engine::kind::cpu`              | `dnnl::engine::kind::gpu`                  |
| ------------------------------------------------------------------------------------ | -------------------------------------- | ------------------------------------------ |
| dnnl::engine(engine::kind, int)                                                      | DPC++ devices: CPU (OpenCL)            | DPC++ devices: GPU (OpenCL and Level Zero) |
| dnnl::sycl_interop::make_engine(const cl::sycl::device &, const cl::sycl::context &) | DPC++ devices: CPU (OpenCL) and host   | DPC++ devices: GPU (OpenCL and Level Zero) |

The host device in DPC++ does not require a backend and is implemented in the
DPC++ runtime directly.

The following sections describe the behavior of oneDNN and the prerequisites
with respect to different backends.

## Library Building

### OpenCL Backend Support

oneDNN is always built with OpenCL backend support, and this support cannot be
disabled at build time. Additional dependencies include OpenCL headers and
OpenCL ICD Loader which are distributed with a DPC++ package and do not require
special handling.

### Level Zero Backend Support

Level Zero backend support is optional. Additional dependencies include [Level
Zero headers](https://github.com/oneapi-src/level-zero/) and should be handled
by the user. Level Zero backend support is enabled implicitly if CMake is able
to find Level Zero headers - in this case CMake prints the corresponding
message:

~~~
-- DPC++ support is enabled (OpenCL and Level Zero)
~~~

## Running

oneDNN builds that were built with OpenCL backend support only cannot be used
with the Level Zero DPC++ backend. This means that the OpenCL DPC++ plugin
must be available on the system.

CPU engines use either a CPU DPC++ device (backed by OpenCL) or a host DPC++
device underneath. GPU engines use a GPU DPC++ device underneath backed by an
OpenCL or Level Zero backend.

For GPU engines, the following rules apply:

- Only one GPU backend is active in oneDNN during an application run. oneDNN
  GPU engines can be created only with the active GPU backend
- If a oneDNN build does not support Level Zero, then the active GPU backend is
  always OpenCL
- When both OpenCL and Level Zero GPU backend support is in place for oneDNN,
  then the active GPU backend is controlled by [SYCL_BE environment
  variable](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md):
    - Empty value (which is the default) enables the Level Zero backend
    - `PI_OPENCL` value corresponds to the OpenCL backend

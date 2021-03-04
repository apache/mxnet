CPU Dispatcher Control {#dev_guide_cpu_dispatcher_control}
==========================================================

oneDNN uses JIT code generation to implement most of its functionality and will
choose the best code based on detected processor features. Sometimes it is
necessary to control which features oneDNN detects. This is sometimes useful for
debugging purposes or for performance exploration.

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`DNNL_ENABLE_JIT_PROFILING`.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_ENABLE_MAX_CPU_ISA     | **ON**, OFF                         | Enables [CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control)

## Run-time Controls

When the feature is enabled at build-time, the `DNNL_MAX_CPU_ISA` environment
variable can be used to limit processor features oneDNN is able to detect to
certain Instruction Set Architecture (ISA) and older instruction sets. It can
also be used to enable ISAs with initial support in the library that are
otherwise disabled by default.

| Environment variable | Value            | Description
| :---                 | :---             | :---
| DNNL_MAX_CPU_ISA     | SSE41            | Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)
|                      | AVX              | Intel Advanced Vector Extensions (Intel AVX)
|                      | AVX2             | Intel Advanced Vector Extensions 2 (Intel AVX2)
|                      | AVX512_MIC       | Intel Advanced Vector Extensions 512 (Intel AVX-512) with AVX512CD, AVX512ER, and AVX512PF extensions
|                      | AVX512_MIC_4OPS  | Intel AVX-512 with AVX512_4FMAPS and AVX512_4VNNIW extensions
|                      | AVX512_CORE      | Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
|                      | AVX512_CORE_VNNI | Intel AVX-512 with Intel Deep Learning Boost (Intel DL Boost)
|                      | AVX512_CORE_BF16 | Intel AVX-512 with Intel DL Boost and bfloat16 support
|                      | **ALL**          | **No restrictions on the above ISAs, but excludes the below ISAs with initial support in the library (default)**
|                      | AVX512_CORE_AMX  | Intel AVX-512 with Intel DL Boost and bfloat16 support and Intel Advanced Matrix Extensions (Intel AMX) with 8-bit integer and bfloat16 support (**initial support**)

@note The ISAs are partially ordered:
* SSE41 < AVX < AVX2,
* AVX2 < AVX512_MIC < AVX512_MIC_4OPS,
* AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16 < AVX512_CORE_AMX.

This feature can also be managed at run-time with the following functions:
* @ref dnnl::set_max_cpu_isa function allows changing the ISA at run-time.
The limitation is that, it is possible to set the value only before the first
JIT-ed function is generated. This limitation ensures that the JIT-ed code
observe consistent CPU features both during generation and execution.
* @ref dnnl::get_effective_cpu_isa function returns the currently used CPU ISA
which is the highest available CPU ISA by default.

Function settings take precedence over environment variables.

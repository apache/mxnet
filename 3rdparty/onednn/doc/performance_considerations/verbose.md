Verbose Mode {#dev_guide_verbose}
========================================================

It is often useful to collect information about how much of an application
runtime is spent executing oneDNN primitives and which of those take
the most time. oneDNN verbose mode enables tracing execution of oneDNN
primitives and collection of basic statistics like execution time and
primitive parameters.

When verbose mode is enabled oneDNN will print out information to `stdout`.
The first lines of verbose information contain the build version and git hash,
if available, as well as CPU and GPU runtimes, and the supported instruction
set architecture.

Each subsequent line of verbose information is formatted as a comma-separated list
containing:
- `dnnl_verbose` marker string
- operation: `create:<cache_hit|cache_miss>` or `exec`
- engine kind: `cpu` or `gpu`
- primitive name: `convolution`, `reorder`, `sum`, etc
- primitive implementation
- propagation: `forward_training`, `forward_inference`, or `backward`
- information about input and output data types and formats
- auxiliary information like algorithm name or number of inputs
- a problem description in [benchdnn format](@ref dev_guide_benchdnn)
- execution time in milliseconds

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`DNNL_VERBOSE`.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_VERBOSE                | **ON**, OFF                         | Enables [verbose mode](@ref dev_guide_verbose)

## Run-time Controls

When the feature is enabled at build-time, the `DNNL_VERBOSE` environment
variable can be used to turn verbose mode on and control the level of verbosity.

| Environment variable | Value            | Description
| :---                 | :---             | :---
| DNNL_VERBOSE         | **0**            | **no verbose output (default)**
|                      | 1                | primitive information at execution
|                      | 2                | primitive information at creation and execution

This feature can also be managed at run-time with the following functions:
* @ref dnnl_set_verbose

The function setting takes precedence over the environment variable.

## Example

~~~sh
DNNL_VERBOSE=1 ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"
~~~

This produces the following output (the line breaks were added to fit the page width):

~~~sh
dnnl_verbose,info,DNNL v1.3.0 (commit d0fc158e98590dfad0165e568ca466876a794597)
dnnl_verbose,info,cpu,runtime:OpenMP
dnnl_verbose,info,cpu,isa:Intel AVX2
dnnl_verbose,info,gpu,runtime:none
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,0.0200195
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,16x16x5x5,0.0251465
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,0.0180664
dnnl_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,,,16,0.0229492
dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_training,src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0390625
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,2x16x7x7,0.173096
~~~

Please see the profiling example [here](@ref dev_guide_verbose), as it uses
DNNL_VERBOSE output to tune oneDNN code to align with
[best practices](@ref dev_guide_inference).

@warning
Verbose mode has non-negligible performance impact especially if the output
rate is high.

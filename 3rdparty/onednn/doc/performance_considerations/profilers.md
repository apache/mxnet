Profiling oneDNN Performance {#dev_guide_profilers}
=================================================

oneDNN uses JIT (just-in-time) code generation based on primitive parameters and
instruction set supported by the system. In order to correctly attribute
performance event information, profilers need to be notified about
address ranges containing JIT-ed code. oneDNN supports two profilers:
VTune(TM) Amplifier and Linux perf.

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`DNNL_ENABLE_JIT_PROFILING`.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_ENABLE_JIT_PROFILING   | **ON**, OFF                         | Enables performance profilers integration

## Run-time Controls

When the feature is enabled at build-time, the `DNNL_JIT_PROFILE` environment
variable can be used to manage integration with performance profilers.

| Environment variable | Value            | Description
| :---                 | :---             | :---
| DNNL_JIT_PROFILE     | **1**            | **Enables VTune Amplifier integration (default)**
|                      | 2                | Enables basic Linux perf integration
|                      | 6                | Enables Linux perf integration with JIT dump output
|                      | 14               | Enables Linux perf integration with JIT dump output and TSC timestamps

Other valid values for `DNNL_JIT_PROFILE` include integer values representing
a combination of flags accepted by @ref dnnl_set_jit_profiling_flags function.

The default setting of the profiling flags is to enable integration with
VTune Amplifier; therefore it does not require any additional setup and works
out of the box. Code integrating oneDNN may override this behavior.

This feature can also be managed at run-time with the following functions:
* @ref dnnl_set_jit_profiling_flags
* @ref dnnl_set_jit_profiling_jitdumpdir

Function settings take precedence over environment variables.

## Example: Profiling with VTune Amplifier

Assuming that environment is set up already.

Collect profiling data:

~~~sh
$ amplxe-cl -collect hotspots -q -no-summary -knob sampling-mode=hw -r dnnl-vtune ./benchdnn --mode=P mb1ic32ih14oc32oh14kh3ph1n"resnet_50:res4a_branch2b*6"
amplxe: Warning: To enable hardware event-base sampling, VTune Amplifier has disabled the NMI watchdog timer. The watchdog timer will be re-enabled after collection completes.
Output template: perf,%engine%,%name%,%desc%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,resnet_50:res4a_branch2b*6,--conv mb1ic32ih14oc32oh14kh3ph1nresnet_50:res4a_branch2b*6,0.0032768,0,2.13525,1.53462,4.32546,0.757561
tests:1 passed:0 skipped:0 mistrusted:0 unimplemented:0 failed:0 listed:0
total perf: min(ms):2.13525 avg(ms):4.32546
~~~

@note You don't need to set `DNNL_JIT_PROFILE` environment variable.

Display top 10 hotspots using command-line interface:

~~~sh
$ amplxe-cl -report hotspots -q -r dnnl-vtune -format csv -csv-delimiter ';' -group-by process,module,function -column 'CPU Time:Self' | head -n 10 | column -t -s';'
Column filter is ON.
Process   Module            Function                                                                                                                           CPU Time
benchdnn  libgomp.so.1.0.0  do_spin                                                                                                                            54.796608
benchdnn  libgomp.so.1.0.0  do_spin                                                                                                                            52.075321
benchdnn  libgomp.so.1.0.0  cpu_relax                                                                                                                          3.979194
benchdnn  libgomp.so.1.0.0  cpu_relax                                                                                                                          3.838870
benchdnn  [Dynamic code]    jit_avx2_conv_fwd_kernel_f32                                                                                                       2.355442
benchdnn  vmlinux           __lock_acquire                                                                                                                     0.801853
benchdnn  vmlinux           do_raw_spin_lock                                                                                                                   0.290672
benchdnn  libdnnl.so.1.1    dnnl::impl::cpu::jit_avx2_convolution_fwd_t::execute_forward(dnnl::impl::exec_ctx_t const&) const::{lambda(intint)#1}::operator()  0.260602
benchdnn  vmlinux           plist_check_prev_next                                                                                                              0.115266
~~~

The JIT-ed function `jit_avx2_conv_fwd_kernel_f32` is shown as belonging to
the `[Dynamic code]` module.

See more examples in the [VTune Amplifier User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/introduction/tutorials-and-samples.html)

## Example: Profiling with Linux perf

The following command instructs oneDNN to enable both jitdump and perfmap
profiling modes and write jitdump files into `.debug` directory in the current
directory by setting environment variable `JITDUMPDIR` to point to the current
directory.

~~~sh
$ JITDUMPDIR=. DNNL_JIT_PROFILE=6 perf record -k1 ./tests/benchdnn/benchdnn --mode=P mb1ic32ih14oc32oh14kh3ph1n"resnet_50:res4a_branch2b*6"
Output template: perf,%engine%,%name%,%desc%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,resnet_50:res4a_branch2b*6,--conv mb1ic32ih14oc32oh14kh3ph1nresnet_50:res4a_branch2b*6,0.0032768,0,0.0131836,248.551,0.0262988,124.599
tests:1 passed:0 skipped:0 mistrusted:0 unimplemented:0 failed:0 listed:0
total perf: min(ms):0.0131836 avg(ms):0.0262988
[ perf record: Woken up 1 times to write data ]
[ perf record: Captured and wrote 0.884 MB perf.data (23102 samples) ]
~~~

The following command injects the information from the jitdump files into the performance data:
~~~sh
$ perf inject -j -i perf.data -o perf.data.j
~~~

The following command displays the top hotspots:
~~~sh
$ perf report -i perf.data.j --stdio | head -n20
# To display the perf.data header info, please use --header/--header-only options.
#
#
# Total Lost Samples: 0
#
# Samples: 23K of event 'cpu-clock:uhH'
# Event count (approx.): 5775500000
#
# Overhead  Command   Shared Object        Symbol
#
    39.33%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8ba
    29.41%  benchdnn  jitted-31475-0.so    [.] jit_avx2_conv_fwd_kernel_f32
    20.49%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d712
     3.47%  benchdnn  libdnnl.so.1.1       [.] dnnl::impl::cpu::jit_avx2_convolution_fwd_t::execute_forward(dnnl::impl::exec_ctx_t const&) const::{lambda(int, int)#1}::operator()
     1.52%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8be
     0.93%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d716
     0.75%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8c5
     0.55%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8c3
     0.46%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d71d
~~~

@note Not every kernel / distribution support displaying detailed profiling
information. Symbol resolution (usually) works as long as the perfmap mode is
enabled, but annotating a JIT-ed functions disassembly, which requires
jitdump, seems to often fail on kernels before 5.x.

See more on the
[Brendan Gregg's excellent perf examples page](http://www.brendangregg.com/perf.html)

# Common Options

**Benchdnn** drivers support a set of options available for every driver.
The following common options are supported:

* --allow-enum-tags-only=`BOOL` -- Instructs the driver to validate format tags
  against the documented tags from `dnnl_format_tag_t` enumeration only.  When
  BOOL is `true` (the default), the only allowed format tags are the ones from
  `dnnl_format_tag_t` enumeration.

* --attr-scratchpad=`MODE` -- Specifies the scratchpad mode to be used for
  benchmarking. MODE values can be `library` (the default) or `user`. Refer to
  [scratchpad primitive attribute](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_scratchpad.html)
  for details.

* --batch=`FILE` -- Instructs the driver to take options and problem descriptors
  from a FILE. If several --batch options are specified, the driver will read
  input files consecutively. Nested inclusion of --batch option is supported.
  The driver searches for a file by extracting a directory where FILE is located
  and then tries to open `dirname(FILE)/FILE`. If file was not found, it tries a
  default path which is `/path_to_benchdnn_binary/inputs/DRIVER/FILE`. If file
  was not found again, an error is reported.

* --canonical=`BOOL` -- Specifies a canonical form of reproducer line to be
  printed. When BOOL equals `false` (the default), the driver prints the minimal
  reproducer line omitting options and problem descriptor entries which values
  are set to their defaults.

* --engine=`ENGINE` -- Specifies an engine kind ENGINE to be used for
  benchmarking. ENGINE values can be `cpu` (the default) or `gpu`.

* --mem-check=`BOOL` -- Instructs the driver to perform a device RAM capability
  check if the problem fits the device. When BOOL is `true` (the default), the
  check is performed.

* --mode=`MODE` -- Specifies **benchdnn** mode to be used for benchmarking. MODE 
  values can be `C` or `c` for correctness testing (the default), `P` or `p` for
  performance testing, `PC` or `pc` for both correctness and performance
  testing, `L` or `l` for listing mode. Refer to
  [modes](benchdnn_general_info.md) for details.

* --reset -- Instructs the driver to reset DRIVER-OPTIONS (not COMMON-OPTIONS!)
  to their default values. The only exception is `--perf-template` option which
  will not be reset.

* --skip-impl=`STR` -- Instructs the driver to return SKIPPED status when the
  implementation name matches STR. STR is a string literal with no spaces. When
  STR is empty (the default), the driver behavior is not modified. STR supports
  several patterns to be matched against through `:` delimiter between patterns.
  E.g. `--skip-impl=ref:gemm`.

* -v`N`, --verbose=`N` -- Specifies the driver verbose level. It prints
  additional information depending on a level N. N is a non-negative integer
  value. The default value is `0`. Refer to [verbose](knobs_verbose.md) for
  details.

The following common options are applicable only for correctness mode:

* --fast-ref-gpu=`BOOL` -- Instructs the driver to use faster reference path
  when doing correctness testing if `--engine=gpu` is specified. When BOOL
  equals `true` (the default), the library best fit CPU implementation is used
  to compute the reference path. Designed to speed up the correctness testing
  for GPU. Currently, the option will make an effect only for the `conv` driver.

The following common options are applicable only for a performance mode:

* --max-ms-per-prb=`N` -- Specifies the limit in milliseconds for performance
  benchmarking set per problem. N is an integer positive number in a range
  [1e2, 6e4]. If a value is out of the range, it will be saturated to range
  board values. The default is `3e3`. This option helps to stabilize the
  performance numbers reported for small problems.

* --fix-times-per-prb=`N` -- Specifies the limit in rounds for performance
  benchmarking set per problem. N is a non-negative integer. When N is set to
  `0` (the default), time criterion is used for benchmarking instead. This
  option is useful for performance profiling, when certain amount of cycles is
  desired.

* --perf-template=`STR` -- Specifies the format of performance report. STR
  values can be `def` (the default), `csv` or a custom set of supported flags.
  Refer to [performance report](knobs_perf_report.md) for details.


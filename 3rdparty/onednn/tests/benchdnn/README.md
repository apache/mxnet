# benchdnn

**benchdnn** is an extended and robust correctness verification and performance
benchmarking tool for the primitives provided by
[oneDNN](https://github.com/oneapi-src/oneDNN). The purpose of the benchmark is
an extended and robust correctness verification of the primitives provided by
oneDNN. **benchdnn** itself is a harness for different primitive-specific
drivers.

## Harness Usage
``` sh
    ./benchdnn --DRIVER [COMMON-OPTIONS] [DRIVER-OPTIONS] PROBLEM-DESCRIPTION
```

where `DRIVER` is one of:
* [binary](doc/driver_binary.md)
* [bnorm](doc/driver_bnorm.md)
* [concat](doc/driver_concat.md)
* [conv](doc/driver_conv.md)
* [deconv](doc/driver_conv.md)
* [eltwise](doc/driver_eltwise.md)
* [ip](doc/driver_ip.md)
* [lnorm](doc/driver_lnorm.md)
* [lrn](doc/driver_lrn.md)
* [matmul](doc/driver_matmul.md)
* [pool](doc/driver_pool.md)
* [reorder](doc/driver_reorder.md)
* [resampling](doc/driver_resampling.md)
* [rnn](doc/driver_rnn.md)
* [shuffle](doc/driver_shuffle.md)
* [softmax](doc/driver_softmax.md)
* [sum](doc/driver_sum.md)

Refer to [`COMMON-OPTIONS`](doc/knobs_common.md) for details on options
supported across all the drivers. Refer to each driver's documentation for
`DRIVER-OPTIONS` and `PROBLEM-DESCRIPTION` definitions, which vary from driver
to driver.

See also [general information](doc/benchdnn_general_info.md) about
**benchdnn**.

## License

**benchdnn** is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Issues and Contributions

We welcome community contributions to **benchdnn** as well as to oneDNN.
If you have any ideas or issues, please submit an issue or pull request. For
clarity, please include ''benchdnn: '' in the title.

## Acknowledgements

This work is inspired by the [benchFFT](http://www.fftw.org/benchfft/) project
developed by Matteo Frigo and Steven G. Johnson as a benchmark for
Discrete Fourier Transform (DFT) implementations.

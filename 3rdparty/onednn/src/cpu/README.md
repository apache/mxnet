oneDNN CPU Implementation
=========================

The source code is organized in a modular way to separate generic code that
does not depend or weakly depends on architecture from architecture-specific
code.
- The generic code is located under `cpu/`;
- The architecture-specific code is put into `cpu/<arch>/` sub-directories.

## Directory structure

```
cpu
├── gemm/               # Generic GEMM implementation (may call <arch>/gemm)
├── rnn/                # Generic RNN implementation (may call <arch>/rnn)
├── x64                 # x64-specific sub-directory
│   ├── gemm/           # x64-specific GEMM implementation
│   ├── jit_utils/      # JIT-related utilities, such as support of profilers
│   ├── rnn/            # JIT-related kernels for rnn primitive
│   ├── xbyak/          # Xbyak sources
│   └── jit_*.*         # x64-specific implementations
├── cpu_engine.hpp      # Basic oneDNN abstractions
├── cpu_lrn_pd.hpp      # Base cpu primitive descriptor classes
├── cpu_lrn_list.cpp    # Implementation lists
├── nchw_pooling.cpp    # Semi-optimized (aka simple) implementations
├── platform.hpp        # Platform-related utility functions
└── ref_eltwise.cpp     # Reference implementations
```

## Target architectures

Currently, the only architecture specific directory is `cpu/x64` which contains
Intel 64 / AMD64 implementations, that mostly use JIT assembler
[Xbyak](https://github.com/herumi/xbyak) to produce highly optimized code.

The architecture specific code can easily access the generic code, but the
opposite should be limited as much as possible. However, sometimes it is
absolutely necessary for generic code to access architecture specific one. For
instance, the list of implementations that live in `cpu/*_list.cpp` should
conditionally include the specific implementations on the corresponding
architecture. Hence, for portability reasons [`cpu/platform.hpp`](platform.hpp)
header file provides a set of helpers macros that could help conditionally
enable or disable parts of code. There the following macros defined:
- `DNNL_X64` is 1 on x64 architecture;
- `DNNL_AARCH64` is 1 on Arm AArch64 architecture;
- `DNNL_PPC64` is 1 on OpenPOWER / IBM Power architecture;
- `DNNL_S390X` is 1 on IBMz / s390x architecture;
- `DNNL_ARCH_GENERIC` is 1 on other platforms.
Only one of the macros above is defined to 1. All others are defined to 0.

Usage example:

``` cpp
#include "cpu/platform.hpp" // IMPORTANT: INCLUDE THIS FILE!

int generic_foo() {
#if DNNL_X64
    return x64_impl_foo();
#else
    return generic_impl_foo();
#endif
}
```

Additionally, there is `DNNL_<ARCH>_ONLY(...)` macro that expands to its
parameters only on the corresponding architectures. Hence, the following
code has the same behavior as the example above:

``` cpp
#include "cpu/platform.hpp" // IMPORTANT: INCLUDE THIS FILE!

int generic_foo() {
    DNNL_X64_ONLY(return x64_impl_foo());
    return generic_impl_foo();
}
```

See more details in [`platform.hpp`](platform.hpp).
Also check `DNNL_TARGET_ARCH` cmake variable.

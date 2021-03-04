API {#dev_guide_c_and_cpp_apis}
==========================================

oneDNN has both **C** and **C++ APIs** available to users for convenience.
There is almost a one-to-one correspondence as far as features are concerned,
so users can choose based on language preference and switch back and forth
in their projects if they desire. Most of the users choose **C++ API** though.

The differences are shown in the table below.

| Features                 | **C API**                                        | **C++ API**
| :-                       | :-                                               | :-
| Minimal standard version | C99                                              | C++11
| Functional coverage      | Full                                             | May require use of the **C API**
| Error handling           | Functions return [status](@ref dnnl_status_t)    | Functions throw [exceptions](@ref dnnl::error)
| Verbosity                | High                                             | Medium
| Implementation           | Completely inside the library                    | Header-based thin wrapper around the **C API**
| Purpose                  | Provide simple API and stable ABI to the library | Improve usability
| Target audience          | Experienced users, FFI                           | Most of the users and framework developers

## Input validation notes

oneDNN performs limited input validation to minimize the performance
overheads. The user application is responsible for sanitizing
inputs passed to the library. Examples of the inputs that may result in
unexpected consequences:
* Not-a-number (NaN) floating point values
* Large `u8` or `s8` inputs may lead to accumulator overflow
* While the `bf16` 16-bit floating point data type has range close to 32-bit
  floating point data type, there is a significant reduction in precision.

As oneDNN API accepts raw pointers as parameters it's the calling code
responsibility to
* Allocate memory and validate the buffer sizes before passing them
to the library
* Ensure that the data buffers do not overlap unless the functionality
explicitly permits in-place computations

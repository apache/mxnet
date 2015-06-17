# MXNet
This is an experimental project to put cxxnet and minerva together, nothing is working yet.

# Guidelines
* Use google c style
* Put module header in [include](include)
  - move them to ```project-name/include``` when we finalized the name
* Depend on [dmlc-core](https://github.com/dmlc/dmlc-core)
* Doxygen comment every function, class and variable for the module headers
  - Ref headers in [dmlc-core/include](https://github.com/dmlc/dmlc-core/tree/master/include/dmlc)
  - Use the same style as dmlc-core
* Try write some use-cases of interface in [test](test)
  - They do not need to link, but need to pass compile
* Minimize dependency, if possible only depend on dmlc-core
* Macro Guard CXX11 code by 
  - Try to make interface compile when c++11 was not avaialable(but with some functionalities pieces missing)
```c++
#include <dmlc/base.h>
#if DMLC_USE_CXX11
  // c++11 code here
#endif
```
* For heterogenous hardware support (CPU/GPU). Hope the GPU-specific component could be isolated easily. That is too say if we use `USE_CUDA` macro to wrap gpu-related code, the macro should not be everywhere in the project.

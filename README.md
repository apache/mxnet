# cxx-minerva

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
  - Use c++11 for threading
```c++
#include <dmlc/base.h>
#if DMLC_USE_CXX11
  // c++11 code here
#endif
```
  - Try to make interface compile when c++11 was not avaialable


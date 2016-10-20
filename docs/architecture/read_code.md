# Read MXNet Code
- All the module interface are listed in [include](../../include), these
  interfaces are heavily documented.
- You read the
  [Doxygen Version](https://mxnet.readthedocs.org/en/latest/doxygen) of the
  document.
- Each module will only depend on other module by the header files in
  [include](../../include).
- The implementation of module is in [src](../../src) folder.
- Each source code only sees the file within its folder,
  [src/common](../../src/common) and [include](../../include).

Most modules are mostly self-contained, with interface dependency on engine.  So
you are free to pick the one you are interested in, and read that part.

# Other resources
* [Doxygen Version of C++ API](https://mxnet.readthedocs.org/en/latest/doxygen) gives a comprehensive document of C++ API.

# Recommended Next Steps

* [Develop and hack MXNet](http://mxnet.io/how_to/develop_and_hack.html)
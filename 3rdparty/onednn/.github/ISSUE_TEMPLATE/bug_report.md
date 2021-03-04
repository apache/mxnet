---
name: Report a bug or a performance issue
about: Use this template to report unexpected behavior
title: ''
labels: 'sighting'
assignees: ''
---

# Summary
Provide a short summary of the issue. Sections below provide guidance on what
factors are considered important to reproduce an issue.

# Version
Report oneDNN version and githash. Version information is printed to stdout
in [verbose mode](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html).

# Environment
oneDNN includes hardware-specific optimizations and may behave
differently on depending on the compiler and build environment. Include
the following information to help reproduce the issue:
* CPU make and model (try `lscpu`; if your `lscpu` does not list CPU flags,
  try running `cat /proc/cpuinfo | grep flags | sort -u`)
* OS version (`uname -a`)
* Compiler version (`gcc --version`)
* CMake version (`cmake --version`)
* CMake output log
* git hash (`git log -1 --format=%H`)

# Steps to reproduce
Please check that the issue is reproducible with the latest revision on
master. Include all the steps to reproduce the issue. 

You can use [verbose mode](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html)
and [benchdnn](https://github.com/oneapi-src/oneDNN/tree/master/tests/benchdnn)
to validate correctness of all primitives the library supports. If this does not
work a short C/C++ program or modified unit tests demonstrating the issue
will greatly help with the investigation.

# Observed behavior
Document behavior you observe. For performance defects, like performance
regressions or a function being slow, provide a log including output generated
by your application in
[verbose mode](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html). 

# Expected behavior
Document behavior you expect.
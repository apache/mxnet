cmake_minimum_required(VERSION 3.13)

# workaround to store CMAKE_CROSSCOMPILING because is getting reset by the project command
if(CMAKE_CROSSCOMPILING)
  set(__CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING})
  set(__CMAKE_CROSSCOMPILING_OVERRIDE ON)
endif()

project(mxnet C CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  # Load config.cmake only if mxnet is not compiled as a dependency of another project
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

if(__CMAKE_CROSSCOMPILING_OVERRIDE)
  set(CMAKE_CROSSCOMPILING ${__CMAKE_CROSSCOMPILING})
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils.cmake)

include(CMakeDependentOption)
#Some things have order. This must be put in front alone
option(USE_CUDA "Build with CUDA support"   ON)
set(MXNET_CUDA_ARCH "Auto" CACHE STRING "Target NVIDIA GPU achitecture.
Format: Auto | Common | All | LIST(ARCH_AND_PTX ...)
- \"Auto\" detects local machine GPU compute arch at runtime.
- \"Common\" and \"All\" cover common and entire subsets of architectures
- ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
- NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal Volta Turing
- NUM: Any number. Only those pairs are currently accepted by NVCC though:
       2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 7.0 7.2 7.5")
option(USE_NCCL "Use NVidia NCCL with CUDA" OFF)
option(USE_OPENCV "Build with OpenCV support" ON)
option(USE_OPENMP "Build with Openmp support" ON)
cmake_dependent_option(USE_CUDNN "Build with cudnn support" ON "USE_CUDA" OFF) # one could set CUDNN_ROOT for search path
cmake_dependent_option(USE_NVTX "Build with nvtx support if found" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_SSE "Build with x86 SSE instruction support" ON "NOT ARM" OFF)
option(USE_F16C "Build with x86 F16C instruction support" ON) # autodetects support if ON
option(USE_LAPACK "Build with lapack support" ON)
option(USE_MKL_IF_AVAILABLE "Use MKL if found" ON)
if(USE_MKL_IF_AVAILABLE AND (NOT APPLE) AND (NOT MSVC) AND (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (NOT CMAKE_CROSSCOMPILING))
  option(USE_MKLDNN "Build with MKL-DNN support" ON)
else()
  option(USE_MKLDNN "Build with MKL-DNN support" OFF)
endif()
if(NOT MSVC)
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" ON)
else()
  option(USE_OPERATOR_TUNING  "Enable auto-tuning of operators" OFF)
endif()
option(USE_GPERFTOOLS "Build with GPerfTools support" OFF)
option(USE_JEMALLOC "Build with Jemalloc support" OFF)
option(USE_DIST_KVSTORE "Build with DIST_KVSTORE support" OFF)
option(USE_PLUGINS_WARPCTC "Use WARPCTC Plugins" OFF)
option(USE_PLUGIN_CAFFE "Use Caffe Plugin" OFF)
option(USE_CPP_PACKAGE "Build C++ Package" OFF)
option(USE_MXNET_LIB_NAMING "Use MXNet library naming conventions." ON)
option(USE_GPROF "Compile with gprof (profiling) flag" OFF)
option(USE_CXX14_IF_AVAILABLE "Build with C++14 if the compiler supports it" OFF)
option(USE_VTUNE "Enable use of Intel Amplifier XE (VTune)" OFF) # one could set VTUNE_ROOT for search path
option(USE_TVM_OP "Enable use of TVM operator build system." OFF)
option(ENABLE_CUDA_RTC "Build with CUDA runtime compilation support" ON)
option(BUILD_CPP_EXAMPLES "Build cpp examples" ON)
option(INSTALL_EXAMPLES "Install the example source files." OFF)
option(USE_SIGNAL_HANDLER "Print stack traces on segfaults." ON)
option(USE_TENSORRT "Enable inference optimization with TensorRT." OFF)
option(USE_ASAN "Enable Clang/GCC ASAN sanitizers." OFF)
option(ENABLE_TESTCOVERAGE "Enable compilation with test coverage metric output" OFF)
option(USE_INT64_TENSOR_SIZE "Use int64_t to represent the total number of elements in a tensor" OFF)
option(BUILD_CYTHON_MODULES "Build cython modules." OFF)
cmake_dependent_option(USE_SPLIT_ARCH_DLL "Build a separate DLL for each Cuda arch (Windows only)." ON "MSVC" OFF)
cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)


message(STATUS "CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING}")
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

if(USE_TVM_OP)
  add_definitions(-DMXNET_USE_TVM_OP=1)
endif()

message(STATUS "CMake version '${CMAKE_VERSION}' using generator '${CMAKE_GENERATOR}'")
project(mxnet C CXX)
if(USE_CUDA)
  cmake_minimum_required(VERSION 3.13.2)  # CUDA 10 (Turing) detection available starting 3.13.2
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER_LOADED AND UNIX AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    message(WARNING "CMAKE_CUDA_COMPILER guessed: ${CMAKE_CUDA_COMPILER}")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  include(CheckCXXCompilerFlag)
  if(USE_CXX14_IF_AVAILABLE)
    check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
    if (SUPPORT_CXX14)
      set(CMAKE_CUDA_STANDARD 14)
    endif()
  endif()
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(UNIX)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
    message(STATUS "Could not find CCache. Consider installing CCache to speed up compilation.")
  endif()
endif()

if(MSVC)
  set(SYSTEM_ARCHITECTURE x86_64)
  enable_language(ASM_MASM)
else()
  execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE SYSTEM_ARCHITECTURE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Distribution" AND UNIX AND NOT APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH $\{ORIGIN\})
  # Enforce DT_PATH instead of DT_RUNPATH
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream;${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")

SET(EXTRA_OPERATORS "" CACHE PATH "EXTRA OPERATORS PATH")

if("$ENV{VERBOSE}" STREQUAL "1")
  message(STATUS " Verbose Makefile ACTIVATED")
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

#Switch off modern thread local for dmlc-core, please see: https://github.com/dmlc/dmlc-core/issues/571#issuecomment-543467484
add_definitions(-DDMLC_MODERN_THREAD_LOCAL=0)
# disable stack trace in exception by default.
add_definitions(-DDMLC_LOG_STACK_TRACE_SIZE=0)
add_definitions(-DDMLC_LOG_FATAL_THROW=1)

if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DMXNET_EXPORTS)
  add_definitions(-DNNVM_EXPORTS)
  add_definitions(-DDMLC_STRICT_CXX11)
  add_definitions(-DNOMINMAX)
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /bigobj")
else()
  include(CheckCXXCompilerFlag)
  if(USE_CXX14_IF_AVAILABLE)
    check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
  endif()
  check_cxx_compiler_flag("-std=c++11"   SUPPORT_CXX11)
  check_cxx_compiler_flag("-std=c++0x"   SUPPORT_CXX0X)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wno-sign-compare")
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang$")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-braced-scalar-init")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_definitions(-DNDEBUG=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
  else()
    add_definitions(-DNDEBUG=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}")
  if(SUPPORT_CXX14)
    add_definitions(-DDMLC_USE_CXX11=1)
    add_definitions(-DDMLC_USE_CXX14=1)
    add_definitions(-DMSHADOW_IN_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif(SUPPORT_CXX11)
    add_definitions(-DDMLC_USE_CXX11=1)
    add_definitions(-DMSHADOW_IN_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(SUPPORT_CXX0X)
    add_definitions(-DDMLC_USE_CXX11=1)
    add_definitions(-DMSHADOW_IN_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  endif()
endif(MSVC)

if(NOT mxnet_LINKER_LIBS)
  set(mxnet_LINKER_LIBS "")
endif(NOT mxnet_LINKER_LIBS)

if(USE_GPROF)
  message(STATUS "Using GPROF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
endif()

if(USE_VTUNE)
  message(STATUS "Using VTUNE")
  if(NOT VTUNE_ROOT)
    set(VTUNE_ROOT /opt/intel/vtune_amplifier_xe_2017)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -pg")
  set(CMAKE_LINK_LIBRARY_FILE_FLAG "${CMAKE_LINK_LIBRARY_FILE_FLAG} -g -pg")
  add_definitions(-DMXNET_USE_VTUNE=1)
  include_directories(${VTUNE_ROOT}/include)
  list(APPEND mxnet_LINKER_LIBS ${VTUNE_ROOT}/lib64/libittnotify.a)
  list(APPEND mxnet_LINKER_LIBS dl)
endif()

if(USE_TENSORRT)
  message(STATUS "Using TensorRT")
  set(ONNX_PATH 3rdparty/onnx-tensorrt/third_party/onnx/build/)
  set(ONNX_TRT_PATH 3rdparty/onnx-tensorrt/build/)

  include_directories(${ONNX_PATH})
  include_directories(3rdparty/onnx-tensorrt/)
  include_directories(3rdparty/)
  include_directories(3rdparty/onnx-tensorrt/third_party/onnx/)
  add_definitions(-DMXNET_USE_TENSORRT=1)
  add_definitions(-DONNX_NAMESPACE=onnx)

  find_package(Protobuf REQUIRED)

  find_library(ONNX_LIBRARY NAMES libonnx.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx library.")
  find_library(ONNX_PROTO_LIBRARY NAMES libonnx_proto.so REQUIRED
          PATHS ${ONNX_PATH}
          DOC "Path to onnx_proto library.")
  find_library(ONNX_TRT_RUNTIME_LIBRARY NAMES libnvonnxparser_runtime.so REQUIRED
          PATHS ${ONNX_TRT_PATH}
          DOC "Path to onnx_proto library.")
  find_library(ONNX_TRT_PARSER_LIBRARY NAMES libnvonnxparser.so REQUIRED
          PATHS ${ONNX_TRT_PATH}
          DOC "Path to onnx_proto library.")

  list(APPEND mxnet_LINKER_LIBS libnvinfer.so ${ONNX_TRT_PARSER_LIBRARY} ${ONNX_TRT_RUNTIME_LIBRARY}
          ${ONNX_PROTO_LIBRARY} ${ONNX_LIBRARY} ${PROTOBUF_LIBRARY})
endif()

# please note that when you enable this, you might run into an linker not being able to work properly due to large code injection.
# you can find more information here https://github.com/apache/incubator-mxnet/issues/15971
if(ENABLE_TESTCOVERAGE)
  message(STATUS "Compiling with test coverage support enabled. This will result in additional files being written to your source directory!")
  find_program( GCOV_PATH gcov )
  if(NOT GCOV_PATH)
    message(FATAL_ERROR "gcov not found! Aborting...")
  endif() # NOT GCOV_PATH

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} --coverage")
  set(GTEST_LIBRARIES "${GTEST_LIBRARIES} --coverage")
  link_libraries(gcov)
endif()

if(USE_MKLDNN)
  # CPU architecture (e.g., C5) can't run on another architecture (e.g., g3).
  if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc /MT")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy /MT")
  endif()

  set(MKLDNN_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
  set(MKLDNN_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
  set(MKLDNN_ARCH_OPT_FLAGS "" CACHE INTERNAL "" FORCE)
  set(MKLDNN_ENABLE_JIT_PROFILING OFF CACHE INTERNAL "" FORCE)
  set(MKLDNN_LIBRARY_TYPE STATIC CACHE INTERNAL "" FORCE)
  set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE INTERNAL "" FORCE)

  if(NOT USE_OPENMP)
      set(MKLDNN_CPU_RUNTIME SEQ CACHE INTERNAL "" FORCE)
  endif()

  add_subdirectory(3rdparty/mkldnn)

  include_directories(3rdparty/mkldnn/include)
  include_directories(${PROJECT_BINARY_DIR}/3rdparty/mkldnn/include)
  add_definitions(-DMXNET_USE_MKLDNN=1)
  list(APPEND mxnet_LINKER_LIBS dnnl)
  set(INSTALL_MKLDNN ON)
endif()

if(USE_CPP_PACKAGE)
    add_definitions(-DMXNET_USE_CPP_PACKAGE=1)
endif()

# Allow Cuda compiles outside of src tree to find things in 'src' and 'include'
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include(cmake/ChooseBlas.cmake)

if(USE_ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
  set(GTEST_LIBRARIES "${GTEST_LIBRARIES} -fsanitize=address")
  list(APPEND mxnet_LINKER_LIBS asan)
endif()

list(APPEND mxnet_LINKER_LIBS ${mshadow_LINKER_LIBS})

foreach(var ${C_CXX_INCLUDE_DIRECTORIES})
    include_directories(${var})
endforeach()

include_directories("include")
include_directories("3rdparty/nvidia_cub")
include_directories("3rdparty/tvm/nnvm/include")
include_directories("3rdparty/tvm/include")
include_directories("3rdparty/dmlc-core/include")
include_directories("3rdparty/dlpack/include")

# commented out until PR goes through
#if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack)
#  add_subdirectory(3rdparty/dlpack)
#endif()

# Prevent stripping out symbols (operator registrations, for example)
if(NOT MSVC AND NOT APPLE)
  set(BEGIN_WHOLE_ARCHIVE -Wl,--whole-archive)
  set(END_WHOLE_ARCHIVE -Wl,--no-whole-archive)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using regular Clang or AppleClang
  set(BEGIN_WHOLE_ARCHIVE -Wl,-force_load)
endif()

if(UNIX)
  find_library(RTLIB rt)
  if(RTLIB)
    list(APPEND mxnet_LINKER_LIBS ${RTLIB})
  endif()
endif()

set(ALT_MALLOC_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

# ---[ gperftools
if(USE_GPERFTOOLS)
  find_package(Gperftools)
  if(GPERFTOOLS_FOUND)
    message(STATUS "Using Gperftools malloc (tcmalloc)")
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${GPERFTOOLS_LIBRARIES})
    set(USE_JEMALLOC 0)
  endif()
endif()

# ---[ jemalloc
if(USE_JEMALLOC)
  if(GPERFTOOLS_FOUND)
    message(ERROR " Only one of USE_JEMALLOC and USE_GPERFTOOLS can be defined at once")
  endif()
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    message(STATUS "Using JEMalloc malloc")
    add_definitions(-DUSE_JEMALLOC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ALT_MALLOC_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALT_MALLOC_FLAGS}")
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    message(STATUS "OpenCV imgcodecs missing")
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND mxnet_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV ${OpenCV_VERSION} found (${OpenCV_CONFIG_PATH})")
  message(STATUS " OpenCV_LIBS=${OpenCV_LIBS}")
  add_definitions(-DMXNET_USE_OPENCV=1)
else(USE_OPENCV)
  message(STATUS "OpenCV Disabled")
  add_definitions(-DMXNET_USE_OPENCV=0)
endif()

# ---[ OpenMP
if(USE_OPENMP)

  function(load_omp)
    # Intel/llvm OpenMP: https://github.com/llvm-mirror/openmp
    set(OPENMP_STANDALONE_BUILD TRUE)
    set(LIBOMP_ENABLE_SHARED TRUE)
    set(CMAKE_BUILD_TYPE Release)
    set(OPENMP_ENABLE_LIBOMPTARGET OFF CACHE BOOL "LLVM OpenMP offloading support")  # Requires CMP0077 CMake 3.13
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmp)
  endfunction()

  # This should build on Windows, but there's some problem and I don't have a Windows box, so
  # could a Windows user please fix?
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmp/CMakeLists.txt
     AND SYSTEM_ARCHITECTURE STREQUAL "x86_64"
     AND NOT CMAKE_BUILD_TYPE STREQUAL "Distribution"
     AND NOT MSVC
     AND NOT CMAKE_CROSSCOMPILING)
    load_omp()
    list(REMOVE_ITEM mxnet_LINKER_LIBS iomp5)
    list(APPEND mxnet_LINKER_LIBS omp)
    if(UNIX)
      list(APPEND mxnet_LINKER_LIBS pthread)
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DMXNET_USE_OPENMP=1)
  else()
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      add_definitions(-DMXNET_USE_OPENMP=1)
    endif()
  endif()
elseif(UNIX AND NOT ANDROID)
  list(APPEND mxnet_LINKER_LIBS pthread)
endif()


# ---[ LAPack
if(USE_LAPACK)
  message("USE_LAPACK is ON")
  add_definitions(-DMXNET_USE_LAPACK=1)
  # In the CMAKE_BUILD_TYPE="Distribution" case, we link against Blas libraries
  # that already provide lapack symbols. Thus -llapack would be wrong.
  if (NOT MSVC AND NOT CMAKE_BUILD_TYPE STREQUAL "Distribution")
    list(APPEND mxnet_LINKER_LIBS lapack)
  endif()
endif()


# ---[ jemalloc
if(USE_JEMALLOC)
  find_package(JeMalloc)
  if(JEMALLOC_FOUND)
    add_definitions(-DUSE_JEMALLOC)
    include_directories(${JEMALLOC_INCLUDE_DIRS})
    set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${JEMALLOC_LIBRARIES})
  endif()
endif()

include(CTest)
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest/googletest")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
#set(GTEST_BOTH_LIBRARIES gtest gtest_main)
set(GTEST_LIBRARIES gtest gtest_main)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)

add_subdirectory(${GTEST_ROOT})
find_package(GTest REQUIRED)

# cudnn detection
if(USE_CUDNN)
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${CUDNN_LIBRARY})
  else()
    set(USE_CUDNN OFF)
  endif()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake)
  add_subdirectory("3rdparty/dmlc-core")
endif()

FILE(GLOB_RECURSE SOURCE "src/*.cc" "src/*.h" "include/*.h")
FILE(GLOB_RECURSE CUDA "src/*.cu" "src/*.cuh")

# add nnvm to source
FILE(GLOB_RECURSE NNVMSOURCE
  3rdparty/tvm/nnvm/src/c_api/*.cc
  3rdparty/tvm/nnvm/src/core/*.cc
  3rdparty/tvm/nnvm/src/pass/*.cc
  3rdparty/tvm/nnvm/src/c_api/*.h
  3rdparty/tvm/nnvm/src/core/*.h
  3rdparty/tvm/nnvm/src/pass/*.h
  3rdparty/tvm/nnvm/include/*.h)
list(APPEND SOURCE ${NNVMSOURCE})

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "3rdparty/tvm/nnvm/*.cc" "plugin/*.cc")
FILE(GLOB_RECURSE GROUP_Include "src/*.h" "3rdparty/tvm/nnvm/*.h" "3rdparty/mshadow/mshadow/*.h" "plugin/*.h")
FILE(GLOB_RECURSE GROUP_CUDA "src/*.cu" "src/*.cuh" "3rdparty/mshadow/mshadow/*.cuh" "plugin/*.cu"
  "plugin/*.cuh" "3rdparty/nvidia_cub/cub/*.cuh")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_Include})
assign_source_group("CUDA" ${GROUP_CUDA})

if(USE_PLUGINS_WARPCTC)
    set(WARPCTC_INCLUDE  "" CACHE PATH "WARPCTC include")
    set(WARPCTC_LIB_DEBUG  "" CACHE FILEPATH "WARPCTC lib")
    set(WARPCTC_LIB_RELEASE  "" CACHE FILEPATH "WARPCTC lib")
    include_directories(SYSTEM ${WARPCTC_INCLUDE})
    list(APPEND mxnet_LINKER_LIBS ${WARPCTC_LIB})
    FILE(GLOB_RECURSE PLUGINS_SOURCE "plugin/warpctc/*.cc" "plugin/warpctc/*.h")
    FILE(GLOB_RECURSE PLUGINS_CUSRC "plugin/warpctc/*.cu")
    list(APPEND SOURCE ${PLUGINS_SOURCE})
    list(APPEND CUDA ${PLUGINS_CUSRC})
endif()

if(USE_OPERATOR_TUNING AND USE_OPENMP)
  add_definitions(-DMXNET_USE_OPERATOR_TUNING=1)
endif()

if(USE_PLUGIN_CAFFE)
  if(NOT USE_CUDA)
    set(CPU_ONLY ON)
    add_definitions(-DCPU_ONLY=1)
  endif()
  if(NOT DEFINED CAFFE_PATH)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/caffe)
      # Need newer FindCUDA.cmake that correctly handles -std=c++11
      cmake_minimum_required(VERSION 3.3)
      set(CAFFE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/caffe)
    else()
      set(CAFFE_PATH $ENV{CAFFE_PATH})
    endif()
  endif()
  list(APPEND CMAKE_MODULE_PATH ${CAFFE_PATH}/cmake)
  include_directories(${CAFFE_PATH}/include)
  include_directories(${CAFFE_PATH}/build/src)
  include_directories(${CMAKE_BINARY_DIR}/caffe/include)
  link_directories(${CAFFE_PATH}/build/lib)
  if(NOT DEFINED CAFFE_PATH)
    message(FATAL_ERROR "Please set CAFFE_PATH to point to the caffe source installation")
  endif()
  FILE(GLOB_RECURSE PLUGINS_SOURCE "plugin/caffe/*.cc" "plugin/caffe/*.h")
  FILE(GLOB_RECURSE PLUGINS_CUSRC "plugin/caffe/*.cu")
  list(APPEND SOURCE ${PLUGINS_SOURCE})
  list(APPEND CUDA ${PLUGINS_CUSRC})
  include_directories(${CMAKE_BINARY_DIR}/include)
  add_definitions(-DMXNET_USE_CAFFE=1)
  list(APPEND mxnet_LINKER_LIBS
    protobuf boost_system boost_thread boost_filesystem
    gflags glog caffe
    ${Caffe_LINKER_LIBS}
)
endif()

if (NOT (EXTRA_OPERATORS STREQUAL ""))
    mxnet_source_group("Extra"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cc")
    mxnet_source_group("Extra\\Cuda"   GLOB_RECURSE "${EXTRA_OPERATORS}/*.cu")
    FILE(GLOB_RECURSE EXTRA_SRC "${EXTRA_OPERATORS}/*.cc")
    FILE(GLOB_RECURSE EXTRA_CUSRC "${EXTRA_OPERATORS}/*.cu")
    list(APPEND SOURCE ${EXTRA_SRC} ${EXTRA_CUSRC})
endif()

if(MSVC)
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endif()

if(USE_CUDA)
  # CUDA_SELECT_NVCC_ARCH_FLAGS is not deprecated, though part of deprecated
  # FindCUDA https://gitlab.kitware.com/cmake/cmake/issues/19199
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/upstream/select_compute_arch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${MXNET_CUDA_ARCH})
  message("-- CUDA: Using the following NVCC architecture flags ${CUDA_ARCH_FLAGS}")
  set(arch_code_list)
  foreach(arch_str ${CUDA_ARCH_FLAGS})
    if((arch_str MATCHES ".*sm_[0-9]+"))
      string( REGEX REPLACE  ".*sm_([0-9]+)" "\\1" arch_code ${arch_str} )
      list(APPEND arch_code_list ${arch_code})
    endif()
  endforeach()

  string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACES "${CUDA_ARCH_FLAGS}")

  find_package(CUDAToolkit REQUIRED cublas cufft cusolver curand
    OPTIONAL_COMPONENTS nvToolsExt nvrtc)

  list(APPEND mxnet_LINKER_LIBS CUDA::cudart CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::curand)
  if(ENABLE_CUDA_RTC)
    if(CUDA_nvrtc_LIBRARY)
      list(APPEND mxnet_LINKER_LIBS CUDA::nvrtc cuda)
      add_definitions(-DMXNET_ENABLE_CUDA_RTC=1)
    else()
      message(FATAL_ERROR "ENABLE_CUDA_RTC=ON, but failed to find NVRTC. CMake will exit." )
    endif()
  endif()
  list(APPEND SOURCE ${CUDA})
  add_definitions(-DMXNET_USE_CUDA=1)

  if(USE_NCCL)
    find_package(NCCL)
    if(NCCL_FOUND)
      include_directories(${NCCL_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${NCCL_LIBRARIES})
      add_definitions(-DMXNET_USE_NCCL=1)
    else()
      add_definitions(-DMXNET_USE_NCCL=0)
      message(WARNING "Could not find NCCL libraries")
    endif()
  endif()
  if(UNIX)
    if(USE_NVTX AND CUDA_nvToolsExt_LIBRARY)
      list(APPEND mxnet_LINKER_LIBS CUDA::nvToolsExt)
      add_definitions(-DMXNET_USE_NVTX=1)
    else()
      message("Building without NVTX support.")
    endif()
  endif()

  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
endif()

# unsupported: if caffe is a subdirectory of mxnet, load its CMakeLists.txt as well
if(USE_PLUGIN_CAFFE)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/caffe)
    add_subdirectory(caffe)
  endif()
endif()


if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF /OPT:ICF")

endif()

# Add cmake targets
add_subdirectory("3rdparty/mshadow")

set(MXNET_INSTALL_TARGETS mxnet)
if(UNIX)
  string(APPEND CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACES}")
  # Create dummy file since we want an empty shared library before linking
  set(DUMMY_SOURCE ${CMAKE_BINARY_DIR}/dummy.c)
  file(WRITE ${DUMMY_SOURCE} "")
  list(APPEND MXNET_INSTALL_TARGETS mxnet_static)
  add_library(mxnet_static STATIC ${SOURCE})
  target_link_libraries(mxnet_static PUBLIC mshadow)
  add_library(mxnet SHARED ${DUMMY_SOURCE})
  target_link_libraries(mxnet PRIVATE ${BEGIN_WHOLE_ARCHIVE} $<TARGET_FILE:mxnet_static> ${END_WHOLE_ARCHIVE})
  target_link_libraries(mxnet PRIVATE mxnet_static)
  target_link_libraries(mxnet_static PUBLIC ${CMAKE_DL_LIBS})
  set_target_properties(mxnet_static PROPERTIES OUTPUT_NAME mxnet)
elseif(MSVC)
  if(USE_CUDA)
    if(MSVC)
      if(USE_SPLIT_ARCH_DLL)
        add_executable(gen_warp tools/windowsbuild/gen_warp.cpp)
        add_library(mxnet SHARED tools/windowsbuild/warp_dll.cpp ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp
                    ${CMAKE_BINARY_DIR}/warp_gen.asm)
        target_link_libraries(mxnet PRIVATE cudart Shlwapi)
        list(GET arch_code_list 0 mxnet_first_arch)
        foreach(arch ${arch_code_list})
          add_library(mxnet_${arch} SHARED ${SOURCE})
          target_link_libraries(mxnet_${arch} PUBLIC mshadow)
          target_compile_options(
            mxnet_${arch}
            PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_${arch}>"
          )
          target_compile_options(
            mxnet_${arch}
            PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_${arch},compute_${arch}>"
          )
          target_compile_options(
            mxnet_${arch} 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
          target_compile_options(
            mxnet_${arch}
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")
        endforeach()

        add_custom_command(
          OUTPUT ${CMAKE_BINARY_DIR}/warp_gen_cpp.cpp ${CMAKE_BINARY_DIR}/warp_gen.asm
          COMMAND gen_warp $<TARGET_FILE:mxnet_${mxnet_first_arch}> WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ DEPENDS $<TARGET_FILE:mxnet_${mxnet_first_arch}>)
      else(USE_SPLIT_ARCH_DLL)
        string(REPLACE ";" " " NVCC_FLAGS_ARCH "${NVCC_FLAGS_ARCH}")
        set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACES}")
        add_library(mxnet SHARED ${SOURCE})
        target_link_libraries(mxnet PUBLIC mshadow)
        target_compile_options(
            mxnet 
            PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MTd -Gy /bigobj>")
        target_compile_options(
            mxnet
            PRIVATE "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-MT -Gy /bigobj>")

      endif(USE_SPLIT_ARCH_DLL)
    else()
      add_library(mxnet SHARED ${SOURCE})
      target_link_libraries(mxnet PUBLIC mshadow)
    endif()
  else()
    add_library(mxnet SHARED ${SOURCE})
    target_link_libraries(mxnet PUBLIC mshadow)
  endif()

endif()

# extension libraries (custom operators, custom subgraphs) are built by default
add_library(customop_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/gemm_lib.cc)
add_library(transposecsr_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposecsr_lib.cc)
add_library(transposerowsp_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/transposerowsp_lib.cc)
add_library(subgraph_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_subgraph/subgraph_lib.cc)
add_library(pass_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_pass/pass_lib.cc)
target_include_directories(customop_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposecsr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(transposerowsp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(subgraph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
target_include_directories(pass_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
if(USE_CUDA)
  add_library(customop_gpu_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/relu_lib.cu)
  target_include_directories(customop_gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
endif()
if(UNIX)
  if (USE_CUDA)
    target_compile_options(customop_gpu_lib PUBLIC -shared)
  endif()
elseif(MSVC)
  target_compile_options(customop_lib PUBLIC /LD)
  target_compile_options(transposecsr_lib PUBLIC /LD)
  target_compile_options(transposerowsp_lib PUBLIC /LD)
  target_compile_options(subgraph_lib PUBLIC /LD)
  target_compile_options(pass_lib PUBLIC /LD)
  set_target_properties(customop_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposecsr_lib PROPERTIES PREFIX "lib")
  set_target_properties(transposerowsp_lib PROPERTIES PREFIX "lib")
  set_target_properties(subgraph_lib PROPERTIES PREFIX "lib")
  set_target_properties(pass_lib PROPERTIES PREFIX "lib")
  if(USE_CUDA)
    target_compile_options(customop_gpu_lib PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>")
    set_target_properties(customop_gpu_lib PROPERTIES PREFIX "lib")
  endif()
endif()

if(USE_DIST_KVSTORE)
  add_subdirectory("3rdparty/ps-lite")
  add_definitions(-DMXNET_USE_DIST_KVSTORE)
  list(APPEND mxnet_LINKER_LIBS pslite)
endif()

if(USE_MKLDNN)
    add_custom_command(TARGET mxnet POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/3rdparty/mkldnn/include/dnnl_config.h  ${CMAKE_SOURCE_DIR}/include/mkldnn/
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_BINARY_DIR}/3rdparty/mkldnn/include/dnnl_version.h  ${CMAKE_SOURCE_DIR}/include/mkldnn/)
endif()

function(BuildTVMOP)
  # scope the variables in BuildTVM.cmake to avoid conflict
  include(cmake/BuildTVM.cmake)
  add_subdirectory("3rdparty/tvm")
endfunction()

if(USE_TVM_OP)
  list(APPEND mxnet_LINKER_LIBS ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm/libtvm_runtime.so)
  BuildTVMOP()
  if(NOT Python3_EXECUTABLE)
    find_package(PythonInterp 3 REQUIRED)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python3 executable")
    if(NOT Python3_EXECUTABLE)
      message(FATAL_ERROR "No python3 interpreter found to build TVM operators")
    endif()
  endif()

  set(TVM_OP_COMPILE_OPTIONS "-o${CMAKE_CURRENT_BINARY_DIR}" "--config" "${CMAKE_CURRENT_BINARY_DIR}/tvmop.conf" "-L" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm")
  if(USE_CUDA)
    set(TVM_OP_COMPILE_OPTIONS "${TVM_OP_COMPILE_OPTIONS}" "--cuda-arch" "\"${CUDA_ARCH_FLAGS}\"")
  endif()

  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/python:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/topi/python:${CMAKE_CURRENT_SOURCE_DIR}/contrib"
    LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm:$ENV{LD_LIBRARY_PATH}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/tvmop/compile.py ${TVM_OP_COMPILE_OPTIONS}
  )
endif()

if(USE_PLUGINS_WARPCTC)
  list(APPEND mxnet_LINKER_LIBS ${WARPCTC_LIB})
endif()

if(MSVC)
  if(USE_SPLIT_ARCH_DLL AND USE_CUDA)
    foreach(arch ${arch_code_list})
      target_link_libraries(mxnet_${arch} PUBLIC ${mxnet_LINKER_LIBS})
      target_link_libraries(mxnet_${arch} PUBLIC dmlc)
    endforeach()
  else()
    target_link_libraries(mxnet PUBLIC ${mxnet_LINKER_LIBS})
    target_link_libraries(mxnet PUBLIC dmlc)
  endif()
else()
  target_link_libraries(mxnet_static PUBLIC ${mxnet_LINKER_LIBS})
  target_link_libraries(mxnet PUBLIC dmlc)
endif()

if(USE_OPENCV AND OpenCV_VERSION_MAJOR GREATER 2)
  add_executable(im2rec "tools/im2rec.cc")
  if(MSVC)
    target_link_libraries(im2rec mxnet)
  else()
    target_link_libraries(im2rec ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE})
  endif()
  target_link_libraries(im2rec
    ${mxnet_LINKER_LIBS}
    ${OpenCV_LIBS}
    dmlc
    )
else()
    message(WARNING "OpenCV_VERSION_MAJOR: ${OpenCV_VERSION_MAJOR}, version 3 with imgcodecs \
    is required for im2rec, im2rec will not be available")
endif()


if(MSVC AND USE_MXNET_LIB_NAMING)
  set_target_properties(mxnet PROPERTIES OUTPUT_NAME "libmxnet")
endif()


include(GNUInstallDirs)
install(TARGETS ${MXNET_INSTALL_TARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# NOTE: Public headers will be installed into ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}, see
#       https://cmake.org/cmake/help/v3.0/variable/CMAKE_INSTALL_PREFIX.html
#       https://cmake.org/cmake/help/v3.0/module/GNUInstallDirs.html

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(USE_MKLDNN)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mkldnn/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mshadow/mshadow/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mshadow)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mxnet)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/nnvm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if (INSTALL_EXAMPLES)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/example  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
endif()

if (USE_SIGNAL_HANDLER)
    add_definitions(-DMXNET_USE_SIGNAL_HANDLER=1)
endif()

# AUTO_INSTALL_DIR -> Optional: specify post-build install direcory
if(AUTO_INSTALL_DIR)
  # ---[ Install Includes
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${AUTO_INSTALL_DIR}/include
    )

  # ---[ Install Examples
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/example ${AUTO_INSTALL_DIR}/example
    )
endif()

if(INSTALL_PYTHON_VERSIONS)
  message(STATUS "Installing for python versions: ${INSTALL_PYTHON_VERSIONS}")
  foreach(version ${INSTALL_PYTHON_VERSIONS})
    set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python${version}/site-packages/mxnet)
    add_custom_command(TARGET mxnet POST_BUILD
      COMMAND mkdir -p ${outdir}
      COMMAND cp -ru ${CMAKE_CURRENT_SOURCE_DIR}/python/mxnet/* ${outdir}
      )
  endforeach()
endif()

if(USE_CPP_PACKAGE)
  add_subdirectory(cpp-package)
endif()

if(BUILD_CPP_EXAMPLES)
  add_subdirectory(example/image-classification/predict-cpp)
endif()
add_subdirectory(tests)

# ---[ Linter target
if(MSVC)
  find_package(PythonInterp)
  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python executable")
endif()
set(LINT_DIRS "include src plugin cpp-package tests")
set(EXCLUDE_PATH "src/operator/contrib/ctc_include")
add_custom_target(mxnet_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DPROJECT_NAME=mxnet -DEXCLUDE_PATH=${EXCLUDE_PATH} -P ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake/lint.cmake)

if(BUILD_CYTHON_MODULES)
  include(cmake/BuildCythonModules.cmake)
  add_cython_modules(3) # Build cython module for python3 if python3 is found
  if(NOT ${PYTHON3_FOUND})
    message(FATAL_ERROR "No python interpreter found to build cython modules")
  endif()
endif()

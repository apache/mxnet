cmake_minimum_required(VERSION 3.0.2)

# workaround to store CMAKE_CROSSCOMPILING because is getting reset by the project command
if(CMAKE_CROSSCOMPILING)
  set(__CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING})
  set(__CMAKE_CROSSCOMPILING_OVERRIDE ON)
endif()

project(mxnet C CXX)

if(__CMAKE_CROSSCOMPILING_OVERRIDE)
  set(CMAKE_CROSSCOMPILING ${__CMAKE_CROSSCOMPILING})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils.cmake)

include(CMakeDependentOption)
#Some things have order. This must be put in front alone
option(USE_CUDA "Build with CUDA support"   ON)
option(USE_OLDCMAKECUDA "Build with old cmake cuda" OFF)
option(USE_NCCL "Use NVidia NCCL with CUDA" OFF)
option(USE_OPENCV "Build with OpenCV support" ON)
option(USE_OPENMP "Build with Openmp support" ON)
cmake_dependent_option(USE_CUDNN "Build with cudnn support" ON "USE_CUDA" OFF) # one could set CUDNN_ROOT for search path
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
option(USE_JEMALLOC "Build with Jemalloc support"   ON)
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

message(STATUS "CMAKE_CROSSCOMPILING ${CMAKE_CROSSCOMPILING}")
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

if(USE_TVM_OP)
  add_definitions(-DMXNET_USE_TVM_OP=1)
endif()

if(USE_CUDA AND NOT USE_OLDCMAKECUDA)
  message(STATUS "CMake version '${CMAKE_VERSION}' using generator '${CMAKE_GENERATOR}'")
  if(
      (
        (${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        OR (${CMAKE_GENERATOR} MATCHES "Xcode.*")
        OR (${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
      ) AND (
        (${CMAKE_VERSION} VERSION_GREATER "3.9.0") OR (${CMAKE_VERSION} VERSION_EQUAL "3.9.0")
      )
    )
    set(FIRST_CUDA TRUE)
    project(mxnet C CXX CUDA)
  else()
    set(FIRST_CUDA FALSE)
    set(USE_OLDCMAKECUDA TRUE)
    project(mxnet C CXX)
  endif()
else()
  project(mxnet C CXX)
endif()


if(MSVC)
  set(SYSTEM_ARCHITECTURE x86_64)
else()
  execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE SYSTEM_ARCHITECTURE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")

SET(EXTRA_OPERATORS "" CACHE PATH "EXTRA OPERATORS PATH")

if("$ENV{VERBOSE}" STREQUAL "1")
  message(STATUS " Verbose Makefile ACTIVATED")
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

#Switch off modern thread local for dmlc-core, please see: https://github.com/dmlc/dmlc-core/issues/571#issuecomment-543467484
add_definitions(-DDMLC_MODERN_THREAD_LOCAL=0)
# disable stack trace in exception by default.
add_definitions(-DDMLC_LOG_STACK_TRACE_SIZE=0)

if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-DMSHADOW_IN_CXX11)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DMXNET_EXPORTS)
  add_definitions(-DNNVM_EXPORTS)
  add_definitions(-DDMLC_STRICT_CXX11)
  add_definitions(-DNOMINMAX)
  if(USE_F16C)
    message("F16C instruction set is not yet supported for MSVC")
  endif()
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /bigobj")
else(MSVC)
  include(CheckCXXCompilerFlag)
  if(USE_CXX14_IF_AVAILABLE)
    check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
  endif()
  check_cxx_compiler_flag("-std=c++11"   SUPPORT_CXX11)
  check_cxx_compiler_flag("-std=c++0x"   SUPPORT_CXX0X)
  # For cross compilation, we can't rely on the compiler which accepts the flag, but mshadow will
  # add platform specific includes not available in other arches
  if(USE_SSE)
    check_cxx_compiler_flag("-msse3"     SUPPORT_MSSE3)
    check_cxx_compiler_flag("-msse2"     SUPPORT_MSSE2)
  else()
    set(SUPPORT_MSSE2 FALSE)
  endif()
  # For cross complication, turn off flag if target device does not support it
  if(USE_F16C)
    # Determine if hardware supports F16C instruction set
    message(STATUS "Determining F16C support")
    include(cmake/AutoDetectF16C.cmake)
  else()
    set(SUPPORT_F16C FALSE)
  endif()
  if(SUPPORT_F16C)
    message(STATUS "F16C enabled")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
  else()
    add_definitions(-DMSHADOW_USE_F16C=0)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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
  if(SUPPORT_MSSE3)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
  elseif(SUPPORT_MSSE2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}")
  if(SUPPORT_CXX14)
    add_definitions(-DDMLC_USE_CXX11=1)
    add_definitions(-DDMLC_USE_CXX14=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif(SUPPORT_CXX11)
    add_definitions(-DDMLC_USE_CXX11=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(SUPPORT_CXX0X)
    add_definitions(-DDMLC_USE_CXX11=1)
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

  add_subdirectory(3rdparty/mkldnn)

  include_directories(3rdparty/mkldnn/include)
  include_directories(${PROJECT_BINARY_DIR}/3rdparty/mkldnn/include)
  add_definitions(-DMXNET_USE_MKLDNN=1)
  list(APPEND mxnet_LINKER_LIBS dnnl)
endif()

# Allow Cuda compiles outside of src tree to find things in 'src' and 'include'
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DMSHADOW_USE_CUDA=1)
  if(FIRST_CUDA AND (NOT USE_OLDCMAKECUDA))
    if(NOT CUDA_TOOLSET)
      set(CUDA_TOOLSET "${CUDA_VERSION_STRING}")
    endif()
  else()
    set(FIRST_CUDA FALSE)
  endif()
  if(USE_NCCL)
    find_package(NCCL)
    if(NCCL_FOUND)
      include_directories(${NCCL_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${NCCL_LIBRARIES})
    else()
      message(WARNING "Could not find NCCL libraries")
    endif()
  endif()
  if(UNIX)
    find_package(NVTX)
    if(NVTX_FOUND)
      include_directories(${NVTX_INCLUDE_DIRS})
      list(APPEND mxnet_LINKER_LIBS ${NVTX_LIBRARIES})
      add_definitions(-DMXNET_USE_NVTX=1)
    else()
      message(WARNING "Could not find NVTX libraries")
    endif()
  endif()
else()
  add_definitions(-DMSHADOW_USE_CUDA=0)
endif()

if(NCCL_FOUND)
  add_definitions(-DMXNET_USE_NCCL=1)
else()
  add_definitions(-DMXNET_USE_NCCL=0)
endif()

if (USE_INT64_TENSOR_SIZE)
  message(STATUS "Using 64-bit integer for tensor size")
  add_definitions(-DMSHADOW_INT64_TENSOR_SIZE=1)
else()
  add_definitions(-DMSHADOW_INT64_TENSOR_SIZE=0)
endif()

include(cmake/ChooseBlas.cmake)
if(USE_CUDA AND FIRST_CUDA)
  include(3rdparty/mshadow/cmake/Utils.cmake)
  include(cmake/FirstClassLangCuda.cmake)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mshadow/cmake)
    include(3rdparty/mshadow/cmake/mshadow.cmake)
    include(3rdparty/mshadow/cmake/Utils.cmake)
    include(3rdparty/mshadow/cmake/Cuda.cmake)
  else()
    include(mshadowUtils)
    include(Cuda)
    include(mshadow)
  endif()
endif()

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
include_directories("3rdparty/mshadow")
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
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmp)
  endfunction()

  find_package(OpenMP REQUIRED)
  # This should build on Windows, but there's some problem and I don't have a Windows box, so
  # could a Windows user please fix?
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmp/CMakeLists.txt
     AND SYSTEM_ARCHITECTURE STREQUAL "x86_64"
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
  if (NOT MSVC)
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
    add_definitions(-DMSHADOW_USE_CUDNN=1)
  else()
    set(USE_CUDNN OFF)
  endif()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake)
  add_subdirectory("3rdparty/dmlc-core")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mshadow/cmake)
  add_subdirectory("3rdparty/mshadow")
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

# add mshadow file
FILE(GLOB_RECURSE MSHADOWSOURCE "3rdparty/mshadow/mshadow/*.h")
FILE(GLOB_RECURSE MSHADOW_CUDASOURCE "3rdparty/mshadow/mshadow/*.cuh")
list(APPEND SOURCE ${MSHADOWSOURCE})
list(APPEND CUDA ${MSHADOW_CUDASOURCE})

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
  if(FIRST_CUDA)
    mshadow_select_nvcc_arch_flags(NVCC_FLAGS_ARCH)
    string(REPLACE ";" " " NVCC_FLAGS_ARCH "${NVCC_FLAGS_ARCH}")
    set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS_ARCH}")
    list(APPEND mxnet_LINKER_LIBS cublas cufft cusolver curand)
    if(ENABLE_CUDA_RTC)
        list(APPEND mxnet_LINKER_LIBS nvrtc cuda)
        add_definitions(-DMXNET_ENABLE_CUDA_RTC=1)
    endif()
    list(APPEND SOURCE ${CUDA})
    add_definitions(-DMXNET_USE_CUDA=1)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  else()
    list(APPEND CUDA_INCLUDE_DIRS ${INCLUDE_DIRECTORIES})
    # define preprocessor macro so that we will not include the generated forcelink header
    if(ENABLE_CUDA_RTC)
      add_definitions(-DMXNET_ENABLE_CUDA_RTC=1)
    endif()
    # Create '.cmake' files for cuda compiles given definitions added thus far
    mshadow_cuda_compile(cuda_objs ${CUDA})
    if(MSVC)
        if(ENABLE_CUDA_RTC)
            FIND_LIBRARY(CUDA_nvrtc_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"  "${CUDA_TOOLKIT_ROOT_DIR}/lib/win32")
            list(APPEND mxnet_LINKER_LIBS ${CUDA_nvrtc_LIBRARY})
            set(CUDA_cuda_LIBRARY "${CUDA_nvrtc_LIBRARY}/../cuda.lib")
            list(APPEND mxnet_LINKER_LIBS ${CUDA_cuda_LIBRARY})
        endif()
        FIND_LIBRARY(CUDA_cufft_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"  "${CUDA_TOOLKIT_ROOT_DIR}/lib/win32")
        list(APPEND mxnet_LINKER_LIBS "${CUDA_cufft_LIBRARY}/../cufft.lib") # For fft operator
        FIND_LIBRARY(CUDA_cusolver_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"  "${CUDA_TOOLKIT_ROOT_DIR}/lib/win32")
        list(APPEND mxnet_LINKER_LIBS "${CUDA_cusolver_LIBRARY}/../cusolver.lib") # For cusolver
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    else(MSVC)
        list(APPEND mxnet_LINKER_LIBS cufft cusolver)
        if(ENABLE_CUDA_RTC)
            list(APPEND mxnet_LINKER_LIBS nvrtc cuda)
        endif()
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    endif()
    list(APPEND SOURCE ${cuda_objs} ${CUDA})
    add_definitions(-DMXNET_USE_CUDA=1)
    if(CUDA_LIBRARY_PATH)
        if(IS_CONTAINER_BUILD)
        # In case of building on a production-like build container which may not have Cuda installed
        if(NOT CMAKE_SYSTEM_HAS_CUDA)
            # Assuming building in a container that doesn't have CUDA installed (ie CPU-only build machine)
            # so use the stub cuda driver shared library
            if(EXISTS ${CUDA_LIBRARY_PATH}/stubs/libcuda.so)
            link_directories(${CUDA_LIBRARY_PATH}/stubs)
            endif()
        endif()
        endif()
    endif()
 endif()
endif()

# unsupported: if caffe is a subdirectory of mxnet, load its CMakeLists.txt as well
if(USE_PLUGIN_CAFFE)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/caffe)
    add_subdirectory(caffe)
  endif()
endif()

if(NOT MSVC)
  # Only add c++11 flags and definitions after cuda compiling
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-DMSHADOW_IN_CXX11)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /EHsc /Gy")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /EHsc /Gy")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL} /OPT:REF /OPT:ICF")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF /OPT:ICF")

endif()

add_library(sample_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/example/extensions/lib_custom_op/gemm_lib.cc)
target_include_directories(sample_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mxnet)
set(MXNET_INSTALL_TARGETS mxnet)
if(UNIX)
  # Create dummy file since we want an empty shared library before linking
  set(DUMMY_SOURCE ${CMAKE_BINARY_DIR}/dummy.c)
  file(WRITE ${DUMMY_SOURCE} "")
  list(APPEND MXNET_INSTALL_TARGETS mxnet_static)
  add_library(mxnet_static STATIC ${SOURCE})
  add_library(mxnet SHARED ${DUMMY_SOURCE})
  target_link_libraries(mxnet PRIVATE ${BEGIN_WHOLE_ARCHIVE} $<TARGET_FILE:mxnet_static> ${END_WHOLE_ARCHIVE})
  target_link_libraries(mxnet PRIVATE mxnet_static)
  target_link_libraries(mxnet_static PUBLIC ${CMAKE_DL_LIBS})
  target_compile_options(sample_lib PUBLIC -shared)
  set_target_properties(mxnet_static PROPERTIES OUTPUT_NAME mxnet)
else()
  add_library(mxnet SHARED ${SOURCE})
  target_compile_options(sample_lib PUBLIC /LD)
  set_target_properties(sample_lib PROPERTIES PREFIX "lib")
endif()

if(USE_CUDA)
  if(FIRST_CUDA AND MSVC)
    target_compile_options(mxnet PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler=-MTd -Gy>")
    target_compile_options(mxnet PUBLIC "$<$<CONFIG:RELEASE>:-Xcompiler=-MT -Gy>")
  endif()
endif()
if(USE_DIST_KVSTORE)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ps-lite/CMakeLists.txt)
    add_subdirectory("3rdparty/ps-lite")
    list(APPEND pslite_LINKER_LIBS pslite protobuf)
    target_link_libraries(mxnet PUBLIC debug ${pslite_LINKER_LIBS_DEBUG})
    target_link_libraries(mxnet PUBLIC optimized ${pslite_LINKER_LIBS_RELEASE})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND mxnet_LINKER_LIBS ${pslite_LINKER_LIBS_DEBUG})
    else()
      list(APPEND mxnet_LINKER_LIBS ${pslite_LINKER_LIBS_RELEASE})
    endif()
    target_link_libraries(mxnet PUBLIC debug ${pslite_LINKER_LIBS_DEBUG})
    target_link_libraries(mxnet PUBLIC optimized ${pslite_LINKER_LIBS_RELEASE})

  else()
    set(pslite_LINKER_LIBS protobuf zmq-static)
  endif()
  add_definitions(-DMXNET_USE_DIST_KVSTORE)
  include_directories(SYSTEM ${pslite_INCLUDE_DIR})
  list(APPEND mxnet_LINKER_LIBS ${pslite_LINKER_LIBS})
endif()

if(USE_TVM_OP)
  list(APPEND mxnet_LINKER_LIBS ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm/libtvm_runtime.so)
  include(cmake/BuildTVM.cmake)
  add_subdirectory("3rdparty/tvm")

  if(NOT Python3_EXECUTABLE)
    find_package(PythonInterp 3 REQUIRED)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python3 executable")
    if(NOT Python3_EXECUTABLE)
      message(FATAL_ERROR "No python3 interpreter found to build TVM operators")
    endif()
  endif()

  set(TVM_OP_COMPILE_OPTIONS "-o${CMAKE_CURRENT_BINARY_DIR}/libtvmop.so" "--config" "${CMAKE_CURRENT_BINARY_DIR}/tvmop.conf")
  if(CUDA_ARCH_BIN)
    set(TVM_OP_COMPILE_OPTIONS "${TVM_OP_COMPILE_OPTIONS}" "--cuda-arch" "${CUDA_ARCH_BIN}")
  endif()
  add_custom_command(TARGET mxnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
          PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/python:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/topi/python:${CMAKE_CURRENT_SOURCE_DIR}/contrib"
          LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm:$ENV{LD_LIBRARY_PATH}
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/tvmop/compile.py ${TVM_OP_COMPILE_OPTIONS}
    )
endif()

target_link_libraries(mxnet PUBLIC ${mxnet_LINKER_LIBS})

if(USE_PLUGINS_WARPCTC)
  target_link_libraries(mxnet PUBLIC debug ${WARPCTC_LIB_DEBUG})
  target_link_libraries(mxnet PUBLIC optimized ${WARPCTC_LIB_RELEASE})
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
    ${pslite_LINKER_LIBS}
    )
else()
    message(WARNING "OpenCV_VERSION_MAJOR: ${OpenCV_VERSION_MAJOR}, version 3 with imgcodecs \
    is required for im2rec, im2rec will not be available")
endif()

target_link_libraries(mxnet PUBLIC dmlc)

if(MSVC AND USE_MXNET_LIB_NAMING)
  set_target_properties(mxnet PROPERTIES OUTPUT_NAME "libmxnet")
endif()

add_subdirectory(tests)

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
  add_cython_modules(2) # Build cython module for python2 if python2 is found
  add_cython_modules(3) # Build cython module for python3 if python3 is found
  if((NOT ${PYTHON2_FOUND}) AND (NOT ${PYTHON3_FOUND}))
    message(FATAL_ERROR "No python interpreter found to build cython modules")
  endif()
endif()

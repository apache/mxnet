cmake_minimum_required(VERSION 2.8.7)

project(mxnet C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
mxnet_option(USE_OPENCV  "Build with OpenCV support" ON)
mxnet_option(USE_OPENMP  "Build with Openmp support" ON)
mxnet_option(USE_CUDNN   "Build with cudnn support"  ON) # one could set CUDNN_ROOT for search path
mxnet_option(USE_CUDA    "Build with CUDA support"   ON)
mxnet_option(USE_DIST_KVSTORE    "Build with DIST_KVSTORE support"   OFF)



include(mshadow/cmake/mshadow.cmake)
include(mshadow/cmake/Utils.cmake)
include(mshadow/cmake/Cuda.cmake)

include_directories("include")
include_directories("mshadow")
include_directories("dmlc-core/include")

if(MSVC)
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-DMSHADOW_IN_CXX11)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DMXNET_EXPORTS)
  set(CMAKE_C_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"   SUPPORT_CXX11)
  check_cxx_compiler_flag("-msse2"       SUPPORT_MSSE2)
  set(CMAKE_C_FLAGS "-O3 -Wall -msse2 -Wno-unknown-pragmas -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
endif(MSVC)

if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  add_definitions(-DMXNET_USE_OPENCV=1)
else(USE_OPENCV)
  message(STATUS "OpenCV Disabled")
  add_definitions(-DMXNET_USE_OPENCV=0)
endif()

if(USE_OPENMP)
 FIND_PACKAGE( OpenMP REQUIRED)
 if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
 endif()
endif()

# cudnn detection
if(USE_CUDNN)
  detect_cuDNN()
  if(HAVE_CUDNN)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND mshadow_LINKER_LIBS ${CUDNN_LIBRARY})
	add_definitions(-DMSHADOW_USE_CUDNN=1)
  endif()
endif()

add_subdirectory("dmlc-core")
if(USE_DIST_KVSTORE)
	add_subdirectory("ps-lite")
endif()

mxnet_source_group("Source"   GLOB_RECURSE "src/*.cc")
mxnet_source_group("Source\\Cuda" GLOB_RECURSE "src/*.cu")

FILE(GLOB_RECURSE SOURCE "src/*.cc")
FILE(GLOB_RECURSE cuda "src/*.cu")

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
  # define preprocessor macro so that we will not include the generated forcelink header
  mshadow_cuda_compile(cuda_objs ${cuda})
  if(MSVC)
    FIND_LIBRARY(CUDA_nvrtc_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"  "${CUDA_TOOLKIT_ROOT_DIR}/lib/win32")
    list(APPEND mshadow_LINKER_LIBS ${CUDA_nvrtc_LIBRARY})
    set(CUDA_cuda_LIBRARY "${CUDA_nvrtc_LIBRARY}/../cuda.lib")
    list(APPEND mshadow_LINKER_LIBS ${CUDA_cuda_LIBRARY})
  else(MSVC)
    list(APPEND mshadow_LINKER_LIBS nvrtc cuda)
  endif()
  list(APPEND SOURCE ${cuda_objs} ${cuda})
endif()


if(NOT MSVC)
  # Only add c++11 flags and definitions after cuda compiling
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-DMSHADOW_IN_CXX11)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

add_library(mxnet SHARED ${SOURCE})
target_link_libraries(mxnet ${mshadow_LINKER_LIBS})
target_link_libraries(mxnet dmlccore)



set_target_properties(mxnet PROPERTIES OUTPUT_NAME "libmxnet")

if(USE_DIST_KVSTORE)
	add_definitions(-DMXNET_USE_DIST_KVSTORE)
	target_link_libraries(mxnet pslite)
	target_link_libraries(mxnet ${pslite_LINKER_LIBS})
	include_directories(SYSTEM ${pslite_INCLUDE_DIR})
endif()

# ---[ Linter target
if(MSVC)
  find_package(PythonInterp 2)
  set(PYTHON2_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 2.x executable")
  find_package(PythonInterp 3)
  set(PYTHON3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 3.x executable")
endif()
set(LINT_DIRS include src scripts python)
add_custom_target(mxnet_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON2_EXECUTABLE=${PYTHON2_EXECUTABLE} -DPYTHON3_EXECUTABLE=${PYTHON3_EXECUTABLE} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DPROJECT_NAME=mxnet -P ${PROJECT_SOURCE_DIR}/dmlc-core/cmake/lint.cmake)
